//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MODialect/Transforms/ShapeInferencePassesCommon.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace MO;
using namespace M::MO::ShapeInferencePassCommon;

namespace M::MO {

#define GEN_PASS_DEF_HOISTPARAMETEREXPRESSIONS

#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

struct RewriteParamExprTypes : public RewritePattern {
  RewriteParamExprTypes(
      MLIRContext *context,
      llvm::DenseMap<Region *,
                     llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
          &&regionToExprToParam)
      : RewritePattern(mlir::Pattern::MatchAnyOpTypeTag(), 1, context),
        regionToExprToParam(regionToExprToParam) {}
  mutable llvm::DenseMap<Region *,
                         llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
      regionToExprToParam;

  /// Check if `ty` contains parameter expression inside its tensor shape, and
  /// if it contains expression not tracked by `regionToExprToParam`, declare a
  /// new parameter for that expression, and add the mapping to
  /// `regionToExprToParam`.
  ///
  /// Return true if `ty` contains _any_ parameter expression though will ignore
  /// any shapes with unknown dimension (e.g. `?`).
  bool collectComplexParams(DeviceAwareRewriter &rewriter, Operation *op,
                            Type ty) const {
    // Bundle types are containers of tensor types; their element shapes can
    // carry `KGEN::ParamOperatorAttr` dims just like top-level tensor types.
    // Recurse so the same `kgen.param.declare` aliases get emitted, and the
    // subsequent `recursivelyReplaceElementsIn` rewrites the bundle's element
    // types consistently with their consumers.
    if (auto bundleTy = dyn_cast<MO::BundleType>(ty)) {
      bool found = false;
      for (Type elemTy : bundleTy.getElementTypes())
        if (collectComplexParams(rewriter, op, elemTy))
          found = true;
      return found;
    }

    auto graphOp = op->getParentOfType<GraphOp>();

    bool found = false;
    auto tensorTy = dyn_cast<MO::TensorType>(ty);
    if (!tensorTy)
      return false;
    auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(tensorTy.getShapeAttr());
    if (!shapeAttr)
      return false;
    for (TypedAttr dimValue : shapeAttr.getValues()) {
      auto paramOp = dyn_cast<KGEN::ParamOperatorAttr>(dimValue);
      if (!paramOp)
        continue;
      found = true;
      // Check if the parameter has already been rewritten.
      Region *parentRegion = op->getParentRegion();
      auto &map = regionToExprToParam[parentRegion];
      if (auto [it, inserted] = map.insert({paramOp, {}}); inserted) {
        // If not, add substitution with a new unique name.
        std::string exprName = graphOp.getNewParamName("param_expr_");
        it->second = exprName;
        // Declare the new parameters with values of complex expression.
        auto dimType = ShapeAttr::getDimType(op->getContext());
        rewriter.create<KGEN::ParamDeclareOp>(
            op->getLoc(), KGEN::ParamDeclAttr::get(exprName, dimType), paramOp);
      }
    }
    return found;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &origRewriter) const override {
    if (isa<KGEN::ParamDeclareOp>(op))
      return origRewriter.notifyMatchFailure(op, "skip parameter declarations");
    // Skip true control-flow ops (mo.output, mo.while, mo.yield, mo.rebind)
    // whose types are pinned by the enclosing graph's signature or whose
    // regions carry non-tensor block arguments.
    //
    // `mo.call` and `mo.if` are exceptions and ARE rewritten:
    //  - mo.call is just a symbol call whose result types can be freely
    //    rewritten (consumers pick up the new types via the output rebinds).
    //  - mo.if has no region block arguments (`NoRegionArguments`) and its
    //    results are not the graph signature, so a composite dim in an
    //    mo.if result type (e.g. `mul_no_wrap(batch, seq)`) can be aliased
    //    like any other op: the alias param is declared in the enclosing
    //    region (where the expression's leaves live), and
    //    `recursivelyReplaceElementsIn` rewrites the branch bodies and the
    //    `mo.yield` operand types to match. Without this, the un-aliased
    //    `ParamExpr` survives to MOGG kernel lowering and asserts in
    //    `getTensorDynamicDimensions`.
    // TODO(GEX-3580): Revisit whether mo.call should implement
    // MOControlOpInterface at all; dropping it would remove that exception.
    if (isa<MO::MOControlOpInterface>(op) && !isa<MO::CallOp, MO::IfOp>(op))
      return origRewriter.notifyMatchFailure(op, "skip control flow");

    // Don't rewrite individual ops inside mo.parallel bodies -- the body
    // must remain kernel-only (no rebinds).  The parallel op itself is
    // rewritten as a whole, and
    // `AttrTypeReplacer::recursivelyReplaceElementsIn` walks the parallel op's
    // region to keep block-arg and body-op stored types consistent with the
    // rewritten bundle element types.
    if (op->getParentOfType<MO::ParallelOp>())
      return origRewriter.notifyMatchFailure(op, "skip parallel body ops");

    DeviceAwareRewriter rewriter(op, origRewriter);

    llvm::DenseSet<Value> rewriteInputs;
    bool containsComplexExprs = false;
    for (auto [operand, ty] :
         llvm::zip(op->getOperands(), op->getOperandTypes())) {
      bool foundComplex = collectComplexParams(rewriter, op, ty);
      if (foundComplex && isa<MO::TensorType>(ty)) {
        // Tensor operand: schedule an input rebind from the original
        // operator-attr type to the alias type, and trigger overall
        // pattern application.
        containsComplexExprs = true;
        rewriteInputs.insert(operand);
      }
      // For `MO::BundleType` operands no further code is needed: the call
      // to `collectComplexParams` above has already recursed into the
      // bundle's element tensor types and declared the alias
      // `kgen.param.declare` ops; `AttrTypeReplacer` later rewrites the
      // bundle's element types in place. We don't insert an input rebind
      // (`MO::RebindOp` is tensor-only).
    }
    for (Type resType : op->getResultTypes())
      if (collectComplexParams(rewriter, op, resType))
        containsComplexExprs = true;

    // Also detect complex expressions reachable through the op's
    // regions -- block-arg types and nested op result types -- so the
    // existing `recursivelyReplaceElementsIn` rewrite (which already
    // walks into regions) is triggered.  Required for `mo.parallel`
    // when its outer types are alias-clean but the body still
    // references the un-aliased form (e.g. block arg rebound to an
    // aliased dim before yield); the `MOControlOpInterface` early-out
    // above keeps `mo.while` / `mo.yield` / `mo.output` etc. out of this
    // path (`mo.if` is handled like a normal op, see above).
    op->walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments())
        if (collectComplexParams(rewriter, op, arg.getType()))
          containsComplexExprs = true;
    });
    op->walk([&](Operation *nested) {
      if (nested == op)
        return;
      for (Type ty : nested->getResultTypes())
        if (collectComplexParams(rewriter, op, ty))
          containsComplexExprs = true;
    });

    if (!containsComplexExprs)
      return rewriter.notifyMatchFailure(op, "no complex expressions");

    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
      Region *parentRegion = op->getParentRegion();
      if (auto it = regionToExprToParam[parentRegion].find(paramOp);
          it != regionToExprToParam[parentRegion].end()) {
        auto dimType = ShapeAttr::getDimType(op->getContext());
        return KGEN::ParamDeclRefAttr::get(it->second, dimType);
      }
      return paramOp;
    });

    // Store result types so we can rebind back to them at the end.
    llvm::SmallVector<Type> resTypes(op->getResultTypes());

    rewriter.setInsertionPoint(op);

    // Create rebinding of inputs to ones with the complex expressions renamed.
    llvm::DenseMap<Value, Value> inputRemap;
    for (Value inp : rewriteInputs) {
      inputRemap[inp] = rewriter.create<MO::RebindOp>(
          op->getLoc(), replacer.replace(inp.getType()), inp);
    }

    // Do substitutions in the op.
    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/true);

    // Rebind outputs to original types.
    rewriter.setInsertionPointAfter(op);
    for (auto [res, ty] : llvm::zip(op->getResults(), resTypes)) {
      if (isa<MO::TensorType>(ty)) {
        auto rebind = rewriter.create<MO::RebindOp>(op->getLoc(), ty, res);
        rewriter.replaceAllUsesExcept(res, rebind, rebind);
      }
    }

    // For `mo.if`, the region terminators (`mo.yield`) feed the op's results
    // along the region-branch edge, so each yield operand type must equal the
    // (now-rewritten) result type. `recursivelyReplaceElementsIn` only rewrites
    // values produced *inside* the regions; a branch that yields a value
    // defined outside the `mo.if` (or whose expression is in a different
    // canonical form than the one aliased above) keeps its original
    // expression-typed value, which would break the control-flow contract.
    // Rebind every mismatching yield operand to the matching result type.
    if (auto ifOp = dyn_cast<MO::IfOp>(op)) {
      TypeRange resultTys = ifOp.getResultTypes();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        auto yieldOp = cast<MO::YieldOp>(region->front().getTerminator());
        rewriter.setInsertionPoint(yieldOp);
        llvm::SmallVector<Value> newYields(yieldOp.getOperands().begin(),
                                           yieldOp.getOperands().end());
        bool changed = false;
        for (auto [i, ty] : llvm::enumerate(resultTys)) {
          if (isa<MO::TensorType>(ty) && newYields[i].getType() != ty) {
            newYields[i] = rewriter.create<MO::RebindOp>(yieldOp.getLoc(), ty,
                                                         newYields[i]);
            changed = true;
          }
        }
        if (changed)
          rewriter.modifyOpInPlace(yieldOp,
                                   [&] { yieldOp->setOperands(newYields); });
      }
    }

    // Use rebound inputs with the operation.
    llvm::SmallVector<Value> inputs;
    for (Value inp : op->getOperands()) {
      if (auto it = inputRemap.find(inp); it != inputRemap.end())
        inputs.push_back(it->second);
      else
        inputs.push_back(inp);
    }
    rewriter.modifyOpInPlace(op, [&] { op->setOperands(inputs); });

    return LogicalResult::success();
  }
};

struct HoistParameterExpressionsPass
    : public M::MO::impl::HoistParameterExpressionsBase<
          HoistParameterExpressionsPass> {
  using HoistParameterExpressionsBase::HoistParameterExpressionsBase;

  /// Look for all parameter declarations of this form:
  ///     kgen.param.declare name = <expr>
  /// where `expr` is a parameter expression, and return mapping from all such
  /// `name` to `expr`.
  static llvm::DenseMap<Region *,
                        llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
  collectExistingExprToParamMap(GraphOp graphOp) {
    llvm::DenseMap<Region *,
                   llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>>
        regionToExprToParam;
    graphOp.walk([&](KGEN::ParamDeclareOp paramDeclOp) {
      Region *parentRegion = paramDeclOp->getParentRegion();
      auto &map = regionToExprToParam[parentRegion];
      if (auto paramExpr =
              dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue())) {
        map.insert({paramExpr, paramDeclOp.getName().str()});
      }
    });
    return regionToExprToParam;
  }

  void runOnOperation() override {
    // High level view:
    // 1. find all KGEN::ParamOperatorAttrs in tensor types.
    // 2. replace KGEN::ParamOperatorAttrs with a KGEN::ParamDeclRefAttr
    // referencing a new parameter
    // 3. generate a high-level alias declaration mapping from
    // KGEN::ParamDeclRefAttr to KGEN::ParamOperatorAttrs
    // 4. rebind all inputs and outputs into new inputs with parameter aliases
    auto *ctx = &getContext();

    RewritePatternSet patterns{ctx};
    patterns.add<RewriteParamExprTypes>(
        ctx, collectExistingExprToParamMap(getOperation()));
    patterns.add<FoldConsecutiveRebinds>(ctx);
    // Needed for folding away (newly introduced) redundant rebinds (with
    // identical input/output types); the alternative is to add an entire
    // canonicalization pass after this pass.
    RebindOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
