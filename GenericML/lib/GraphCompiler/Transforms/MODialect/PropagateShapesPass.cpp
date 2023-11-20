//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/SetOperations.h"

using namespace M;
using namespace MO;

namespace M::MO {

#define GEN_PASS_DEF_PROPAGATESHAPES

#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h.inc"

} // namespace M::MO

namespace {

bool isParamOperatorPure(KGEN::ParamOperatorAttr paramOperator) {
  if (paramOperator.getOpcode() == KGEN::POC::Apply)
    return false;
  return true;
}

bool isParamDeclOpPure(KGEN::ParamDeclareOp paramDeclOp) {
  if (auto paramOperatorAttr =
          dyn_cast<KGEN::ParamOperatorAttr>(paramDeclOp.getValue()))
    return isParamOperatorPure(paramOperatorAttr);
  return true;
}

// Assume `useDefGraph` is constructed for `graphOp`, remove all the dead
// `kgen.param.declare` ops (basically aggressive DCE applied to
// `kgen.param.declare`s).
void removeDeadParamDeclares(GraphOp graphOp,
                             const KGEN::ParameterUseDefGraph &useDefGraph) {
  // Invariants:
  // Any op that has been on `worklist`
  // 1. is in `processed`
  // 2. must be alive
  DenseSet<Operation *> processed;
  SmallVector<Operation *> worklist;
  auto pushOntoWorklist = [&](Operation *op) {
    if (auto [_, inserted] = processed.insert(op); inserted)
      worklist.push_back(op);
  };

  // Initialize worklist
  graphOp.walk([&](Operation *op) {
    // Conservatively collect seeding ops which must be alive -- the only
    // exceptions are `kgen.param.declare` that are pure (they _might_ not
    // need to be alive), everything else must be alive.
    auto paramDeclOp = dyn_cast<KGEN::ParamDeclareOp>(op);
    if (paramDeclOp && isParamDeclOpPure(paramDeclOp))
      return;
    else
      pushOntoWorklist(op);
  });

  // Process the worklist and find more ops that must be alive.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // `op` is considered alive, so all of the parameter it uses must be
    // alive, so the definition ops of those parameters must be alive.
    for (KGEN::ParamDeclRefAttr usedParamDeclRef :
         useDefGraph.opUses.lookup(op)) {
      const KGEN::ParamDefinition &usedParamDef =
          useDefGraph.defs.at(usedParamDeclRef.getName());
      pushOntoWorklist(usedParamDef.defOp);
    }
  }

  // Remove all the `kgen.param.declarle` ops that do not have to be alive.
  graphOp.walk([&](KGEN::ParamDeclareOp paramDeclOp) {
    if (!processed.contains(paramDeclOp))
      paramDeclOp.erase();
  });
}

void scheduleOpByUsedParams(Operation *, KGEN::ParameterUseDefGraph &,
                            DenseSet<Operation *> &);
/// For each op inside `block` (including nested blocks within), make sure any
/// depended ops (SSA or parameter dependencies) appear before the op.
/// `scheduledOps` is used to track ops which already satisfy this property.
void scheduleBlock(Block *block, KGEN::ParameterUseDefGraph &useDefGraph,
                   DenseSet<Operation *> &scheduledOps) {
  mlir::sortTopologically(block);
  for (Operation &op : block->getOperations())
    scheduleOpByUsedParams(&op, useDefGraph, scheduledOps);
}

/// For all parameters used by `op` (i.e., used in the output shape), make
/// sure their defining `kgen.param.declare` appear before `op`, by moving
/// them if not already in `scheduledOps`. Also puts `op` into `scheduledOps`.
void scheduleOpByUsedParams(Operation *op,
                            KGEN::ParameterUseDefGraph &useDefGraph,
                            DenseSet<Operation *> &scheduledOps) {
  // Skip if already scheduled.
  if (!scheduledOps.insert(op).second)
    return;

  for (KGEN::ParamDeclRefAttr declRefAttr : useDefGraph.opUses.lookup(op)) {
    Operation *paramDefOp =
        useDefGraph.defs.lookup(declRefAttr.getName()).defOp;
    // Avoid bringing graph op into scheduling because it may declare input
    // parameters used inside the graph.
    if (!isa_and_nonnull<GraphOp>(paramDefOp) &&
        !scheduledOps.contains(paramDefOp)) {
      paramDefOp->moveBefore(op);
      scheduleOpByUsedParams(paramDefOp, useDefGraph, scheduledOps);
    }
  }

  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      scheduleBlock(&block, useDefGraph, scheduledOps);
}

/// Handle output op and graph result parameters. This is special in that
/// 1. We need to manually check for the binding parameters, because they
/// never materialize kgen.param.decl (the result params are declared in the
/// graph signature).
/// 2. We need to be careful about mapping graph result parameters.
void handleOutputBindings(GraphOp graphOp,
                          KGEN::ParameterEvaluator &evaluator) {
  // First we collect the declared parameter names so we can check for them.
  SmallPtrSet<StringAttr, 4> resDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getResultParams())
    resDeclNames.insert(decl.getName());
  SmallPtrSet<StringAttr, 4> argDeclNames;
  for (KGEN::ParamDeclAttr decl : graphOp.getInputParams())
    argDeclNames.insert(decl.getName());

  // Check if `attr` can appear as a dimension attribute in graph output.
  auto isValidOutputDim = [&](Attribute dimAttr) {
    // Don't allow any reference to parameters declared inside the graph.
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(dimAttr))
      return argDeclNames.contains(declRef.getName());

    if (isa<IntegerAttr>(dimAttr))
      return true;

    return false;
  };

  // Check if `attr` can appear as a dimension attribute in graph output.
  auto isValidOutputShape = [&](Attribute attr) {
    // Don't allow any reference to parameters declared inside the graph.
    if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr))
      return argDeclNames.contains(declRef.getName());

    if (auto shapeAttr = dyn_cast<MOSH::ShapeAttr>(attr))
      return llvm::all_of(shapeAttr.getValues(), isValidOutputDim);

    return false;
  };

  auto outputOp = graphOp.getOutputOp();
  for (auto [bindingParam, resultParam] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParams())) {
    // First we resolve the binding parameter.
    Attribute resolvedBindingParam =
        evaluator.getReboundAttribute(bindingParam);

    // Output parameter could be either a shape or dimension, we check to make
    // sure the resolved parameter is valid.
    if ((isa<ShapeType>(bindingParam.getType()) &&
         isValidOutputShape(resolvedBindingParam)) ||
        isValidOutputDim(resolvedBindingParam)) {
      evaluator.setParameterValue(resultParam.getName(), resolvedBindingParam);
    } else {
      // Otherwise we still map the result parameter to itself so replacer
      // won't complain.
      evaluator.setParameterValue(resultParam,
                                  KGEN::ParamDeclRefAttr::get(resultParam));
    }
  }
}

/// Update the signature's result declarations and output bindings to account
/// for any output parameter simplification (e.g., maybe an old output
/// parameter `D1` is deduced to be the same as an input parameter `D0`).
void updateGraphSignature(GraphOp graphOp) {
  SmallPtrSet<StringAttr, 4> resParamNames;
  for (Type resTy : graphOp.getResultTypes())
    if (auto resTensorTy = dyn_cast<MO::TensorType>(resTy))
      llvm::set_union(resParamNames, resTensorTy.getUsedParamNames());

  auto outputOp = graphOp.getOutputOp();
  SmallVector<KGEN::ParamDeclAttr> newResultParams{};
  SmallVector<TypedAttr> newBindingParams{};
  for (auto [bindingParam, resParamDecl] :
       llvm::zip(outputOp.getParameters(), graphOp.getResultParams())) {
    if (resParamNames.contains(resParamDecl.getName())) {
      newResultParams.emplace_back(resParamDecl);
      newBindingParams.emplace_back(bindingParam);
    }
  }

  outputOp.setParameters(newBindingParams);

  // Update the result parameters.
  auto newResultParamsAttr =
      KGEN::ParamDeclArrayAttr::get(graphOp->getContext(), newResultParams);
  NamedAttrList attrs = graphOp->getAttrDictionary();
  attrs.set(graphOp.getResultParamsAttrName(), newResultParamsAttr);

  SignatureType signature = KGEN::SignatureType::remapToSignature(
      graphOp.getInputParams(), newResultParamsAttr, graphOp.getFunctionType());
  attrs.set(graphOp.getSignatureAttrName(), TypeAttr::get(signature));
  graphOp->setAttrs(attrs.getDictionary(graphOp->getContext()));
}

// Some expression attribute can be simplified as long as there are extra
// assertions emitted to legalize the simplification. However, not all such
// assertions can be folded away statically. In order to still benefit from
// the simplified expression and avoid having to provide a runtime primitive
// for the original expression, we try to expand/decompose these expressions
// into smaller primitives we already support, and try to simplify those
// smaller primitives as much as possible.
//
// Return true if expansion happened; the resulting ops from the expansion
// will be pushed onto `worklist` in correct order, to be processed later.
bool tryExpandDeclOpExpr(KGEN::ParamDeclareOp declOp,
                         KGEN::ParameterEvaluator &evaluator,
                         std::deque<Operation *> &worklist, GraphOp graphOp) {
  // Only try to expand expression that applies an mlir op.
  auto operatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(
      evaluator.getReboundAttribute(declOp.getValue()));
  if (!operatorAttr || operatorAttr.getOpcode() != KGEN::POC::Apply)
    return false;

  ArrayRef<TypedAttr> operands = operatorAttr.getOperands();
  auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(operands.front());
  if (!mlirOpAttr)
    return false;

  // If expandable, insert and collect the expanded ops.
  OpBuilder builder(declOp);
  Location loc = declOp.getLoc();
  SmallVector<Operation *> expandedOps;

  // Create an op of this form:
  //   kgen.param.declare declPrefix... : resTy = <poc(operands...)>
  auto createDeclToKgenExpr = [&](StringRef declPrefix, Type resTy,
                                  KGEN::POC poc, ArrayRef<TypedAttr> operands)
      -> std::pair<KGEN::ParamDeclareOp, KGEN::ParamDeclRefAttr> {
    auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
    KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
    auto declOp = builder.create<KGEN::ParamDeclareOp>(loc, declAttr, exprAttr);
    auto declRefAttr = KGEN::ParamDeclRefAttr::get(declOp.getParamDecl());
    expandedOps.push_back(declOp);
    return {declOp, declRefAttr};
  };

  // Create an op of this form:
  //   kgen.param.declare declPrefix... : resTy =
  //    <apply(..."mosh.assert", errorMsg, condAttr)>
  auto createAssert = [&](StringRef declPrefix, StringRef errorMsg,
                          TypedAttr condAttr)
      -> std::pair<KGEN::ParamDeclareOp, KGEN::ParamDeclRefAttr> {
    MLIRContext *ctx = condAttr.getContext();
    Type chainType = MOSH::ChainType::get(ctx);

    // Build a new operand list which starts with MLIROpAttr.
    auto signature = SignatureType::get(ctx, {condAttr.getType()}, chainType);
    NamedAttribute msgAttr(
        StringAttr::get(ctx, "message"),
        StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
    auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
    auto opNameAttr = StringAttr::get(ctx, MOSH::AssertOp::getOperationName());
    SmallVector<TypedAttr> applyOperands{
        KGEN::MLIROpAttr::get(opNameAttr, dictAttr, signature), condAttr};

    return createDeclToKgenExpr(declPrefix, chainType, KGEN::POC::Apply,
                                applyOperands);
  };

  // Check to see if the mlir op expression can be expanded.
  if (mlirOpAttr.getName() == MOSH::BroadcastOp::getOperationName()) {
    MLIRContext *ctx = declOp.getContext();
    Type boolType = IntegerType::get(ctx, 1);
    auto intOneAttr = IntegerAttr::get(IndexType::get(ctx), 1);

    auto shapeAttr1 =
        dyn_cast<MOSH::ShapeAttr>(evaluator.getReboundAttribute(operands[1]));
    auto shapeAttr2 =
        dyn_cast<MOSH::ShapeAttr>(evaluator.getReboundAttribute(operands[2]));
    if (shapeAttr1 && shapeAttr2) {
      // Ensure rank(shapeAttr1) >= rank(shapeAttr2)
      if (shapeAttr1.getRank() < shapeAttr2.getRank())
        std::swap(shapeAttr1, shapeAttr2);

      SmallVector<TypedAttr> resDims;
      size_t rankDiff = shapeAttr1.getRank() - shapeAttr2.getRank();
      for (size_t axis = 0; axis < shapeAttr1.getRank(); axis++) {
        if (axis < rankDiff) {
          resDims.push_back(shapeAttr1.getValues()[axis]);
        } else {
          auto dimAttr1 = shapeAttr1.getValues()[axis];
          auto dimAttr2 = shapeAttr2.getValues()[axis - rankDiff];
          resDims.push_back(KGEN::ParamOperatorAttr::get(KGEN::POC::Max,
                                                         {dimAttr1, dimAttr2}));

          // Insert assertions
          auto [dimAttr1IsOneDeclOp, dimAttr1IsOneDeclRef] =
              createDeclToKgenExpr("dim_1_is_one", boolType, KGEN::POC::EQ,
                                   {dimAttr1, intOneAttr});
          auto [dimAttr2IsOneDeclOp, dimAttr2IsOneDeclRef] =
              createDeclToKgenExpr("dim_2_is_one", boolType, KGEN::POC::EQ,
                                   {dimAttr2, intOneAttr});
          auto [dimsEqualDeclOp, dimsEqualDeclRef] = createDeclToKgenExpr(
              "dims_equal", boolType, KGEN::POC::EQ, {dimAttr1, dimAttr2});
          auto [dimsCompatibleDeclOp, dimsCompatibleDeclRef] =
              createDeclToKgenExpr("dims_compatible", boolType, KGEN::POC::Or,
                                   {dimAttr1IsOneDeclRef, dimAttr2IsOneDeclRef,
                                    dimsEqualDeclRef});
          createAssert("dims_assert", "dimensions are not broadcastable",
                       dimsCompatibleDeclRef);
        }
      }
      declOp.setValueAttr(MOSH::ShapeAttr::get(resDims, shapeAttr1.getType()));
      expandedOps.push_back(declOp);
    }
  }

  if (!expandedOps.empty()) {
    // The first expanded op must be processed next in the worklist, so we
    // push the expanded ops onto the worklist in reversed order.
    for (Operation *op : llvm::reverse(expandedOps))
      worklist.push_back(op);
    return true;
  }

  return false;
}

} // namespace

struct PropagateShapesPass
    : public MO::impl::PropagateShapesBase<PropagateShapesPass> {
  using PropagateShapesBase::PropagateShapesBase;

  /// For each op inside `graphOp` (including nested blocks within), make sure
  /// any depended ops (SSA or parameter dependencies) appear before the op.
  void topoSortWithParameters(GraphOp graphOp) {
    // Construct parameter use-def graph
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();
    KGEN::ParameterUseDefGraph useDefGraph(graphOp.getRegion());
    useDefGraph.calculate(paramCache);

    // Sort the kgen.param.declare ops based on parameter use-def chains.
    DenseSet<Operation *> scheduledOps;
    scheduleBlock(graphOp.getBodyBlock(), useDefGraph, scheduledOps);
  }

  /// Return the depth of a KGEN param operator, where
  /// - <add(D0, 1)> has depth 1
  /// - <add(mul(D0, 2), 1)> has depth 2
  size_t getParamAttrDepth(KGEN::ParamOperatorAttr paramOperator) {
    if (!paramAttrToDepth.contains(paramOperator)) {
      size_t depth = 0;
      for (TypedAttr operand : paramOperator.getOperands()) {
        if (auto nestedParamOperator =
                dyn_cast<KGEN::ParamOperatorAttr>(operand)) {
          depth = std::max(depth, getParamAttrDepth(nestedParamOperator));
        }
      }
      depth += 1;
      paramAttrToDepth[paramOperator] = depth;
    }
    return paramAttrToDepth.at(paramOperator);
  }

  bool canParamOperatorBeNested(KGEN::ParamOperatorAttr paramOperator) {
    if (!isParamOperatorPure(paramOperator))
      return false;

    // A heuristic to avoid having super nested KGEN expression which cannot be
    // folded away. `kMaxDepth` should be picked such that, folding for
    // practical models should only require at most this amount of expression
    // nesting.
    constexpr size_t kMaxDepth = 2;
    if (getParamAttrDepth(paramOperator) > kMaxDepth)
      return false;

    // Be conservative and only allow nesting for expressions shape inference
    // care about folding, this mainly keeps the IR easy to read (also helps in
    // case we ever want to expose the `!mo.tensor` symbolic shapes to users).
    switch (paramOperator.getOpcode()) {
    case KGEN::POC::Add:
    case KGEN::POC::Max:
    case KGEN::POC::Mul:
      [[fallthrough]];
    case KGEN::POC::MulNuw:
      return true;
    default:
      return false;
    }
  }

  void runOnOperation() override {
    // Clear state variables (in case the pass instance is reused)
    paramAttrToDepth.clear();

    GraphOp graphOp = getOperation();
    topoSortWithParameters(graphOp);

    // There are 3 special cases for adding param mappings:
    // 1. Input parameters of the graph, e.g., `mo.graph<D0, Sh1 -> ...>`
    // 2. Output param decls in an op that could not be resolved by
    // shape inference, e.g., `mo.reshape<() -> D0>`.
    // 3. Mappings to selected expressions, to avoid extremely large expressions
    // that cannot be folded, e.g., `max(D42, max(D772, 42, add(...)), ...))`.
    // TODO this limits folding as well, we need a better approach.
    //
    // In these cases, we map the declared parameters to themselves, so that
    // evaluator won't panic when looking them up.
    KGEN::ParameterEvaluator evaluator;
    auto mapDeclaredParamToItself = [&](KGEN::ParamDeclAttr paramDecl) {
      evaluator.setParameterValue(paramDecl,
                                  KGEN::ParamDeclRefAttr::get(paramDecl));
    };
    llvm::for_each(graphOp.getInputParams(), mapDeclaredParamToItself);
    graphOp.walk([&](ParamDeclarationInterface paramDeclItf) {
      llvm::for_each(paramDeclItf.getOutputParamDecls(),
                     mapDeclaredParamToItself);
    });

    // TODO(#9008) inline inputs to `mosh.assert`
    // A replacer to help replacing parameters in an op with its evaluated form.
    // TODO(#26115) we should switch to replacing manually via
    // `evaluator.getReboundAttribute(shapeAttr)` for each op, since we only
    // care about specific region of an op, and to avoid having 2 caches -- 1 in
    // replacer and 1 in evaluator.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
      Attribute resolvedAttr = evaluator.getReboundAttribute(declRef);
      return resolvedAttr;
    });

    DenseMap<KGEN::ParamOperatorAttr, KGEN::ParamDeclRefAttr> exprToDeclRef;

    // A stack of ops to be "evaluated", i.e., the op in the back is the one to
    // be processed next.
    std::deque<Operation *> worklist;
    graphOp.walk([&](Operation *op) {
      // Skip the graph op itself since it'll be handled at the very end, after
      // we process all the ops inside the graph.
      if (!isa<GraphOp>(op))
        worklist.push_front(op);
    });

    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      // Use accumulated param-to-expression mapping to simplify the op first.
      // This is needed to help DCE get rid of the decls with simplified exprs.
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceTypes=*/true);

      // Every single `kgen.param.declare x = <expr>` should add a mapping from
      // `x` to some attribute in `evaluator`.
      if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
        // If expansion happened, no need to process `op` anymore, instead
        // continue to process the resulting ops from the expansion.
        if (tryExpandDeclOpExpr(declOp, evaluator, worklist, graphOp))
          continue;

        KGEN::ParamDeclAttr declAttr = declOp.getParamDecl();
        auto reboundDeclAttr =
            cast<TypedAttr>(evaluator.getReboundAttribute(declOp.getValue()));

        auto paramOperatorAttr =
            dyn_cast<KGEN::ParamOperatorAttr>(reboundDeclAttr);
        // If the param expression cannot be nested, we either map the decl to
        // itself or to a previous decl that has the same expression. The goal
        // here is maximize textual equality for expression, e.g.,
        // - `D0` and `D0` for decls that map to themselves or other decls.
        // - `add(D0, 2)` and `add(D0, 2)` for decls that map to expressions.
        // This enables more simplifications.
        if (paramOperatorAttr && !canParamOperatorBeNested(paramOperatorAttr)) {
          // CSE on the fly for expression that cannot be nested, which helps
          // the following scenario, where we'd otherwise lose textual equality
          // for `D0` and `D1`:
          //   kgen.param.declare D0 = <...nested_expr_e1...>
          //   kgen.param.declare D1 = <...nested_expr_e1...>
          //   static.broadcast_to(...) : <[D0], f32> -> <[D1], f32>
          if (auto existingDeclRef = exprToDeclRef.lookup(paramOperatorAttr)) {
            evaluator.setParameterValue(declAttr, existingDeclRef);

            // Else we remember this decl has this expression.
          } else {
            auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
            if (isParamOperatorPure(paramOperatorAttr))
              exprToDeclRef.try_emplace(paramOperatorAttr, declRefAttr);
            mapDeclaredParamToItself(declAttr);
          }

        } else {
          // The param expression can be nested, so we just record it as is.
          evaluator.setParameterValue(declAttr, reboundDeclAttr);
        }
      }
    }

    handleOutputBindings(graphOp, evaluator);
    replacer.replaceElementsIn(graphOp, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);
    updateGraphSignature(graphOp);

    // DCE on `kgen.param.declare`s for better readability/debuggability.
    auto &paramCache = getAnalysis<KGEN::ParameterCollector::Analysis>();
    KGEN::ParameterUseDefGraph useDefGraph(graphOp.getRegion());
    useDefGraph.calculate(paramCache);
    removeDeadParamDeclares(graphOp, useDefGraph);
  }

  // State variables used by the pass
  DenseMap<KGEN::ParamOperatorAttr, size_t> paramAttrToDepth;
};
