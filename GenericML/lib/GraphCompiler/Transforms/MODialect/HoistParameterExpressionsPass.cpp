//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "ShapeInferencePassesCommon.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace M;
using namespace MO;
using namespace M::MO::ShapeInferencePassCommon;

namespace M::MO {

#define GEN_PASS_DEF_HOISTPARAMETEREXPRESSIONS

#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

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
    auto graphOp = op->getParentOfType<GraphOp>();

    bool found = false;
    auto tensor_ty = dyn_cast<MO::TensorType>(ty);
    if (!tensor_ty)
      return false;
    if (tensor_ty.hasUnknownShape())
      return false;
    auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(tensor_ty.getShapeAttr());
    if (!shapeAttr)
      return false;
    for (TypedAttr dimValue : shapeAttr.getValues()) {
      auto paramOp = dyn_cast<KGEN::ParamOperatorAttr>(dimValue);
      if (!paramOp)
        continue;
      found = true;
      // Check if the parameter has already been rewritten.
      Region *parentRegion = op->getParentRegion();
      if (!regionToExprToParam.contains(parentRegion)) {
        regionToExprToParam[parentRegion] =
            llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>();
      }
      if (!regionToExprToParam[parentRegion].contains(paramOp)) {
        // If not, add substitution with a new unique name.
        std::string expr_name = graphOp.getNewParamName("param_expr_");
        regionToExprToParam[parentRegion][paramOp] = expr_name;
        // Declare the new parameters with values of complex expression.
        rewriter.create<KGEN::ParamDeclareOp>(
            op->getLoc(),
            KGEN::ParamDeclAttr::get(expr_name, rewriter.getIndexType()),
            paramOp);
      }
    }
    return found;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &origRewriter) const override {
    DeviceAwareRewriter rewriter(op, origRewriter);

    // Skip parameter declarations and control flow.
    if (isa<KGEN::ParamDeclareOp, MO::MOControlOpInterface>(op)) {
      return rewriter.notifyMatchFailure(
          op, "don't rewrite declarations and output");
    }

    llvm::DenseSet<Value> rewriteInputs;
    bool containsComplexExprs = false;
    for (auto [operand, ty] :
         llvm::zip(op->getOperands(), op->getOperandTypes())) {
      if (collectComplexParams(rewriter, op, ty)) {
        rewriteInputs.insert(operand);
        containsComplexExprs = true;
      }
    }
    for (Type resType : op->getResultTypes())
      if (collectComplexParams(rewriter, op, resType))
        containsComplexExprs = true;

    if (!containsComplexExprs)
      return rewriter.notifyMatchFailure(op, "no complex expressions");

    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamOperatorAttr paramOp) -> Attribute {
      Region *parentRegion = op->getParentRegion();
      if (regionToExprToParam.contains(parentRegion)) {
        if (auto it = regionToExprToParam[parentRegion].find(paramOp);
            it != regionToExprToParam[parentRegion].end())
          return KGEN::ParamDeclRefAttr::get(it->second,
                                             rewriter.getIndexType());
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
      if (!regionToExprToParam.contains(parentRegion)) {
        regionToExprToParam[parentRegion] =
            llvm::DenseMap<KGEN::ParamOperatorAttr, std::string>();
      }
      if (auto paramExpr =
              dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue())) {
        regionToExprToParam[parentRegion].insert(
            {paramExpr, paramDeclOp.getName().str()});
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
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
