//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Analysis/MODialect/ParametricTopologyAnalysis.h"
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

/// Handle output op and graph result parameters. This is special in that
/// 1. We need to manually check for the binding parameters, because they
/// never materialize kgen.param.decl (the result params are declared in the
/// graph signature).
/// 2. We need to be careful about mapping graph result parameters.
///
/// NOTE this assumes we've run shape propagation for all the ops within a
/// `mo.graph`.
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
    // NOTE the binding parameter has already been rebound.
    // Output parameter could be either a shape or dimension, we check to make
    // sure the resolved parameter is valid.
    if ((isa<ShapeType>(bindingParam.getType()) &&
         isValidOutputShape(bindingParam)) ||
        isValidOutputDim(bindingParam)) {
      evaluator.setParameterValue(resultParam.getName(), bindingParam);
    }
    // Otherwise `KGEN::ParameterEvaluator` will map the result parameter to
    // itself by default, meaning it won't change in the signature.
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

/// A class that helps `PropagateShapesPass` handle ops that could be
/// decomposed when the shape parameters they use have been simplified
/// already. For instance, in order to simplify `mosh.broadcast`, we often need
/// its two input shapes to be simplified first (e.g., `[D0, 1]` instead of
/// `Sh0`). However, the simplification happens _during_ `PropagateShapesPass`,
/// so we use this `OpDecomposer` to decompose ops on the fly, and letting
/// `PropagateShapesPass` to further simplify the decomposed ops.
class OpDecomposer {
  // Create an op of the following form and track it in `decomposedOps`.
  //   kgen.param.declare declPrefix... = <poc(operands...)>
  //
  // Return a parameter decl ref to the newly created parameter `declPrefix...`.
  KGEN::ParamDeclRefAttr createDeclToKgenExpr(StringRef declPrefix,
                                              KGEN::POC poc,
                                              ArrayRef<TypedAttr> operands) {
    auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
    return createDeclToKgenAttr(declPrefix, exprAttr);
  };

  // Create a new kgen declaration from the given attribute and return a
  // parameter decl ref to the newly created parameter.
  KGEN::ParamDeclRefAttr createDeclToKgenAttr(StringRef declPrefix,
                                              TypedAttr attr) {
    Type resTy = attr.getType();
    KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
    auto declOp = builder.create<KGEN::ParamDeclareOp>(loc, declAttr, attr);
    auto declRefAttr = KGEN::ParamDeclRefAttr::get(declOp.getParamDecl());
    decomposedOps.push_back(declOp);
    return declRefAttr;
  };

  // Create an op of the following form and track it in `decomposedOps`.
  //   kgen.param.declare declPrefix... =
  //    <apply(..."mosh.assert", errorMsg, condAttr)>
  //
  // Return a parameter decl ref to the newly created parameter `declPrefix...`.
  KGEN::ParamDeclRefAttr createAssert(StringRef declPrefix, StringRef errorMsg,
                                      TypedAttr condAttr) {
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

    return createDeclToKgenExpr(declPrefix, KGEN::POC::Apply, applyOperands);
  };

  // State that stays invariant across calls.
  OpBuilder builder;

  // State that might change across calls.
  GraphOp graphOp;
  Location loc;
  SmallVector<Operation *> decomposedOps;

public:
  /// Create an OpDecomposer with given MLIRContext.
  OpDecomposer(MLIRContext *ctx) : builder(ctx), loc(builder.getUnknownLoc()) {}

  /// Try decomposing the given Operation if possible. Return a list of
  /// decomposed ops (ordered by SSA and parametric dependencies) if successful,
  /// else return an empty vector. The input operation won't be modified.
  SmallVector<Operation *> tryDecomposeOp(Operation *op) {
    // Reset transient state.
    graphOp = op->getParentOfType<GraphOp>();
    loc = op->getLoc();
    decomposedOps.clear();

    if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
      tryDecomposeParamDeclOp(declOp);

    } else if (auto invokeShapeFuncOp = dyn_cast<InvokeShapeFuncOp>(op)) {
      tryDecomposeInvokeShapeFuncOp(invokeShapeFuncOp);
    }

    return decomposedOps;
  }

private:
  // Some parameter declaration can be simplified as long as there are extra
  // assertions emitted to legalize the simplification. However, not all such
  // assertions can proven to be trivial statically. In order to still benefit
  // from the expression simplification and avoid having to provide a runtime
  // primitive for the original expression, we try to decompose these
  // expressions into smaller primitives we already support, and try to simplify
  // those smaller primitives as much as possible.
  void tryDecomposeParamDeclOp(KGEN::ParamDeclareOp declOp) {
    // Only try to decompose expression that applies an mlir op.
    auto operatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(declOp.getValue());
    if (!operatorAttr || operatorAttr.getOpcode() != KGEN::POC::Apply)
      return;

    ArrayRef<TypedAttr> operands = operatorAttr.getOperands();
    auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(operands.front());
    if (!mlirOpAttr)
      return;

    // Check to see if the mlir op expression can be decomposed.
    if (mlirOpAttr.getName() == MOSH::BroadcastOp::getOperationName()) {
      MLIRContext *ctx = declOp.getContext();
      auto intOneAttr = IntegerAttr::get(IndexType::get(ctx), 1);

      auto shapeAttr1 = dyn_cast<MOSH::ShapeAttr>(operands[1]);
      auto shapeAttr2 = dyn_cast<MOSH::ShapeAttr>(operands[2]);
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
            // Insert assertions
            KGEN::ParamDeclRefAttr dimAttr1IsOne = createDeclToKgenExpr(
                "dim_1_is_one", KGEN::POC::EQ, {dimAttr1, intOneAttr});
            KGEN::ParamDeclRefAttr dimAttr2IsOne = createDeclToKgenExpr(
                "dim_2_is_one", KGEN::POC::EQ, {dimAttr2, intOneAttr});
            KGEN::ParamDeclRefAttr dimsEqual = createDeclToKgenExpr(
                "dims_equal", KGEN::POC::EQ, {dimAttr1, dimAttr2});
            KGEN::ParamDeclRefAttr dimsCompatible =
                createDeclToKgenExpr("dims_compatible", KGEN::POC::Or,
                                     {dimAttr1IsOne, dimAttr2IsOne, dimsEqual});
            createAssert("dims_assert", "dimensions are not broadcastable",
                         dimsCompatible);

            // def bcast(D1, D2):
            //     assert `D1 == D2 or D1 == 1 or D2 == 1`
            //     if D1 == 1:
            //         return D2
            //     else:
            //         return D1
            TypedAttr newResDim = KGEN::ParamOperatorAttr::get(
                KGEN::POC::Cond, {dimAttr1IsOne, dimAttr2, dimAttr1});
            resDims.push_back(newResDim);
          }
        }
        auto newDeclOp = cast<KGEN::ParamDeclareOp>(declOp->clone());
        newDeclOp.setValueAttr(
            MOSH::ShapeAttr::get(resDims, shapeAttr1.getType()));
        decomposedOps.push_back(newDeclOp);
      }
    }
  }

  // Assume `invokeShapeFuncOp` applies the shape function of `MO::BroadcastTo`,
  // try decomposing the shape function call into `kgen.param.declare`
  // primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeBroadcastToShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    auto inputTensorVal = cast<TensorValue>(invokeShapeFuncOp.getOperand(0));
    auto inputShape =
        dyn_cast<MOSH::ShapeAttr>(inputTensorVal.getType().getShapeAttr());
    auto targetShape = tryGetShapeAttr(invokeShapeFuncOp.getOperand(1));
    if (!inputShape || !targetShape)
      return {};

    int inputRank = inputShape.getRank();
    int outputRank = targetShape.getRank();
    // Illegal broadcast, let runtime handle this error.
    if (inputRank > outputRank)
      return {};

    MLIRContext *ctx = invokeShapeFuncOp->getContext();
    auto intOneAttr = IntegerAttr::get(IndexType::get(ctx), 1);
    for (int i = 0; i < inputRank; i++) {
      int inputAxis = inputRank - i - 1;
      int outputAxis = outputRank - i - 1;
      TypedAttr inputDim = inputShape.getValues()[inputAxis];
      TypedAttr outputDim = targetShape.getValues()[outputAxis];

      // Insert assertions
      KGEN::ParamDeclRefAttr inputDimIsOne = createDeclToKgenExpr(
          "input_dim_is_one", KGEN::POC::EQ, {inputDim, intOneAttr});
      KGEN::ParamDeclRefAttr dimsEqual = createDeclToKgenExpr(
          "dims_equal", KGEN::POC::EQ, {inputDim, outputDim});
      KGEN::ParamDeclRefAttr dimsCompatible = createDeclToKgenExpr(
          "dims_compatible", KGEN::POC::Or, {inputDimIsOne, dimsEqual});
      createAssert("dims_assert",
                   "[broadcast_to] input dim must be either 1 or equal to "
                   "corresponding output dim starting from the rightmost dim",
                   dimsCompatible);
    }
    return targetShape;
  }

  /// If target shape contains a `-1`, infer it based on the invariant that
  /// `inputShapeAttr` and `targetShapeAttr` represent equal number of elements.
  ///
  /// Example:
  ///   inputShapeAttr = [3, D4]
  ///   targetShapeAttr = [D0, -1, 42]
  ///   returns [D0, RS0, 42]
  /// with the following inserted by `rewriter`
  ///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
  ShapeAttr resolveReshapeTargetShape(ShapeAttr inputShapeAttr,
                                      ShapeAttr targetShapeAttr) {
    auto indexType = builder.getType<IndexType>();
    // Find index of `-1` dimension (if any), and collect all other dimensions.
    SmallVector<TypedAttr> targetNonNegOneDims;
    std::optional<size_t> maybeTargetNegOneDimIdx = std::nullopt;
    for (auto [idx, targetDim] : llvm::enumerate(targetShapeAttr.getValues())) {
      auto intDim = dyn_cast<IntegerAttr>(targetDim);
      if (intDim && intDim.getInt() == -1) {
        // Detected multiple negative one, gives up simplifying this reshape
        // function call and let runtime handle the error for now.
        if (maybeTargetNegOneDimIdx.has_value())
          return {};
        maybeTargetNegOneDimIdx = idx;
      } else {
        // TODO: we are using the assumption that all non-integer attribute
        // (e.g., `D0` or `mul(D0, D1)`) will be non-negative. To make this
        // completely safe, we'll need to extend and leverage the range
        // analysis here.
        targetNonNegOneDims.push_back(targetDim);
      }
    }

    // no literal `-1` dimension, nothing to resolve
    if (!maybeTargetNegOneDimIdx.has_value())
      return targetShapeAttr;

    // Infer the actual value of `-1` dimension based on other dimensions
    KGEN::ParamDeclRefAttr numerator = createDeclToKgenAttr(
        "reshape_dyndim_numerator", inputShapeAttr.getSymbolicNumElements());

    // Add a 1 to the shapes to multiply to ensure the denominator is valid
    auto mulDims = targetNonNegOneDims;
    auto one = IntegerAttr::get(indexType, 1);
    mulDims.emplace_back(one);

    KGEN::ParamDeclRefAttr denominator = createDeclToKgenExpr(
        "reshape_dyndim_denominator", KGEN::POC::MulNuw, mulDims);

    // Avoid division by 0, occurs for empty tensors in which case the -1 dim
    // being 0 would always work. There is ambiguity here in allowed values but
    // in practice probably not important.
    // TODO: think about removing the max(...) as it is very hard to remove
    // TODO: without range-based analysis (and even then it is hard).
    KGEN::ParamDeclRefAttr denominatorNormalized =
        createDeclToKgenExpr("reshape_dyndim_denominator_normalized",
                             KGEN::POC::Max, {one, denominator});
    KGEN::ParamDeclRefAttr resolvedDim =
        createDeclToKgenExpr("reshape_dyndim_resolved", KGEN::POC::Div,
                             {numerator, denominatorNormalized});

    SmallVector<TypedAttr> resolvedDims = targetNonNegOneDims;
    resolvedDims.insert(resolvedDims.begin() + maybeTargetNegOneDimIdx.value(),
                        resolvedDim);

    return ShapeAttr::get(resolvedDims, targetShapeAttr.getType());
  }

  // Assume `invokeShapeFuncOp` applies the shape function of `MO::ReshapeOp`,
  // try decomposing the shape function call into `kgen.param.declare`
  // primitives if possible, return null otherwise.
  ShapeAttr tryDecomposeReshapeShape(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Make sure both input and target shapes are ranked symbolic shapes.
    auto inputShape = dyn_cast<ShapeAttr>(
        cast<TensorType>(invokeShapeFuncOp.getOperand(0).getType())
            .getShapeAttr());
    ShapeAttr targetShape = tryGetShapeAttr(invokeShapeFuncOp.getOperand(1));
    if (!inputShape || !targetShape)
      return {};

    // Make sure we are able to resolve any `-1` (inferred dimension) in the
    // target shape.
    ShapeAttr outputShape = resolveReshapeTargetShape(inputShape, targetShape);
    if (!outputShape)
      return {};

    // Insert assertions
    KGEN::ParamDeclRefAttr inputNumeElems = createDeclToKgenAttr(
        "input_num_elems", inputShape.getSymbolicNumElements());
    KGEN::ParamDeclRefAttr outputNumeElems = createDeclToKgenAttr(
        "output_num_elems", outputShape.getSymbolicNumElements());

    KGEN::ParamDeclRefAttr numElemsEqual = createDeclToKgenExpr(
        "num_elems_equal", KGEN::POC::EQ, {inputNumeElems, outputNumeElems});
    createAssert("reshape_assert",
                 "[reshape] input and output number of elements must match",
                 numElemsEqual);

    return outputShape;
  }

  // Treat certain fallback shape function call as special intrinsics and
  // effectively symbolicizing them _during_ shape propagation.
  //
  // TODO this should eventually become general symbolicization applied to a
  // single `InvokeShapeFuncOp`, however we can't do it yet since we need to
  // symbolicize in batch in order to minimize the overhead.
  void tryDecomposeInvokeShapeFuncOp(InvokeShapeFuncOp invokeShapeFuncOp) {
    // Dispatch on certain shape function names.
    StringRef shapeFuncName = invokeShapeFuncOp.getShapeFuncName();
    ShapeAttr outputShape;
    if (shapeFuncName == "broadcast_to_shape") {
      outputShape = tryDecomposeBroadcastToShape(invokeShapeFuncOp);

    } else if (shapeFuncName == "reshape_shape") {
      outputShape = tryDecomposeReshapeShape(invokeShapeFuncOp);
    }

    // Create new declaration of the shape function call's output parameter if
    // decomposition succeeded.
    if (outputShape) {
      for (auto [outputDecl, outputDim] :
           llvm::zip(invokeShapeFuncOp.getOutputParamDecls(),
                     outputShape.getValues())) {
        auto declOp =
            builder.create<KGEN::ParamDeclareOp>(loc, outputDecl, outputDim);
        decomposedOps.push_back(declOp);
      }
    }
  }
};

} // namespace

struct PropagateShapesPass
    : public MO::impl::PropagateShapesBase<PropagateShapesPass> {
  using PropagateShapesBase::PropagateShapesBase;

  /// For each op inside `graphOp` (including nested blocks within), make sure
  /// any depended ops (SSA or parameter dependencies) appear before the op.
  void topoSortWithParameters(GraphOp graphOp) {
    ParametricTopologyAnalysis analysis(graphOp);
    analysis.topologicallySortAllOpsWithin(&graphOp.getRegion());
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
    constexpr size_t kMaxDepth = 4;
    auto size = getParamAttrDepth(paramOperator);
    if (size > kMaxDepth)
      return false;

    return true;
  }

  /// Track `attr` as definitely non-negative.
  void markDefinitelyNonNegative(TypedAttr attr) {
    definitelyNonNegativeAttrs.insert(attr);
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// non-positive, and return false we can't claim anything.
  bool isDefinitelyNonPositive(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNonPositive();

    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr || exprAttr.getNumOperands() != 2)
      return false;

    // If we have an integer operand, it'll be `rhs` because the order is
    // canonicalized by attribute type.
    TypedAttr lhs = exprAttr.getOperand(0);
    TypedAttr rhs = exprAttr.getOperand(1);

    // mul(non_negative_expr, -1) must be non-positive
    if (llvm::is_contained({KGEN::POC::Mul, KGEN::POC::MulNuw},
                           exprAttr.getOpcode())) {
      auto intAttr = dyn_cast<IntegerAttr>(rhs);
      if (intAttr && intAttr.getInt() == -1 && isDefinitelyNonNegative(lhs))
        return true;
    }

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// non-negative, and return false we can't claim anything.
  bool isDefinitelyNonNegative(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNonNegative();

    if (definitelyNonNegativeAttrs.contains(attr))
      return true;

    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return false;

    // If operands are all non-negative, maximum result will preserve the
    // property.
    if (exprAttr.getOpcode() == KGEN::POC::Max) {
      return llvm::all_of(exprAttr.getOperands(), [&](TypedAttr operand) {
        return isDefinitelyNonNegative(operand);
      });
    }

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// negative, and return false we can't claim anything.
  bool isDefinitelyNegative(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isNegative();

    return false;
  }

  /// A conservative predicate that returns true iff `attr` is provable to be
  /// positive, and return false we can't claim anything.
  bool isDefinitelyPositive(TypedAttr attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getValue().isStrictlyPositive();

    return false;
  }

  /// This function serves as shape inference system's extension to KGEN
  /// folders. It leverages things like range information to enable more
  /// expression folding, and applies it only to `opcode(operands)`, assuming
  /// `operands` have been folded as much as possible.
  ///
  /// Return null if we can't apply any special folding.
  TypedAttr tryFoldExprAttr(KGEN::POC opcode, ArrayRef<TypedAttr> operands) {
    // Only fold certain binary expression for now.
    if (operands.size() != 2)
      return {};

    // If we have an integer operand, it'll be `rhs` because the order is
    // canonicalized by attribute type.
    TypedAttr lhs = operands[0];
    TypedAttr rhs = operands[1];
    MLIRContext *ctx = lhs.getContext();

    if (opcode == KGEN::POC::Max) {
      // TODO: need a pattern matcher real bad...
      // <max(add(A, B), B)>
      // `lhs` and `rhs` are the operands to the max
      auto maxAddOptimization = [&](TypedAttr maxLHS,
                                    TypedAttr maxRHS) -> TypedAttr {
        // Assume add operand is on the lhs.
        auto lhsOp = dyn_cast<M::KGEN::ParamOperatorAttr>(maxLHS);
        if (!lhsOp || lhsOp.getOpcode() != KGEN::POC::Add)
          return {};

        if (lhsOp.getNumOperands() != 2)
          return {};

        // ensure add operands are parameters
        TypedAttr addLHS = lhsOp.getOperand(0);
        TypedAttr addRHS = lhsOp.getOperand(1);

        // If both params `A` and `B` are non negative, then `add(A, B) >= B`
        // and the max of both sides is `add(A, B)`
        if (!isDefinitelyNonNegative(addLHS) ||
            !isDefinitelyNonNegative(addRHS))
          return {};

        // Trick to handle `<max(add(A, B), B)>` and `<max(add(A, B), A)>`
        if (addLHS == rhs || addRHS == maxRHS)
          return maxLHS;

        return {};
      };
      // Check with both permutations
      if (auto maybeMaxAddOptimization = maxAddOptimization(lhs, rhs))
        return maybeMaxAddOptimization;
      if (auto maybeMaxAddOptimization = maxAddOptimization(rhs, lhs))
        return maybeMaxAddOptimization;

      // max(non_negative_expr, non_positive_expr) = non_negative_expr
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return lhs;

      // max(non_positive_expr, non_negative_expr) = non_negative_expr
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return rhs;

    } else if (opcode == KGEN::POC::Min) {
      // min(non_negative_expr, non_positive_expr) = non_positive_expr
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return rhs;

      // min(non_positive_expr, non_negative_expr) = non_positive_expr
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return lhs;

    } else if (opcode == KGEN::POC::LT) {
      // NOTE we have to put this logic here because for some reason the KGEN
      // folder doesn't consider it valid to fold things like
      // `lt(9223372036854775807, 100)`, maybe due to potential overflow? But we
      // know this is safe as far as shape inference is concerned.
      auto intLhs = dyn_cast<IntegerAttr>(lhs);
      auto intRhs = dyn_cast<IntegerAttr>(rhs);
      if (intLhs && intRhs) {
        bool isLessThan = intLhs.getValue().slt(intRhs.getValue());
        return BoolAttr::get(ctx, isLessThan);
      }

      // lt(non_negative_expr, non_positive_expr) = false
      if (isDefinitelyNonNegative(lhs) && isDefinitelyNonPositive(rhs))
        return BoolAttr::get(ctx, false);

      // lt(non_positive_expr, non_negative_expr) = true
      if (isDefinitelyNonPositive(lhs) && isDefinitelyNonNegative(rhs))
        return BoolAttr::get(ctx, true);

    } else if (opcode == KGEN::POC::EQ) {
      MLIRContext *ctx = lhs.getContext();
      // eq(non_negative_expr, negative_expr) = false
      if ((isDefinitelyNonNegative(lhs) && isDefinitelyNegative(rhs)) ||
          (isDefinitelyNonNegative(rhs) && isDefinitelyNegative(lhs)))
        return BoolAttr::get(ctx, false);

      // eq(non_positive_expr, positive_expr) = false
      if ((isDefinitelyNonPositive(lhs) && isDefinitelyPositive(rhs)) ||
          (isDefinitelyNonPositive(rhs) && isDefinitelyPositive(lhs)))
        return BoolAttr::get(ctx, false);
    }

    return {};
  }

  /// This function serves as shape inference system's extension to KGEN
  /// folders. It leverages things like range information to enable more
  /// expression folding, and applies it recursively to all expression nested
  /// inside `attr`.
  TypedAttr tryFoldExprAttr(TypedAttr attr) {
    auto exprAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!exprAttr)
      return attr;

    KGEN::POC opcode = exprAttr.getOpcode();
    SmallVector<TypedAttr> foldedOperands;
    for (TypedAttr operand : exprAttr.getOperands())
      foldedOperands.push_back(tryFoldExprAttr(operand));

    if (TypedAttr foldedAttr = tryFoldExprAttr(opcode, foldedOperands))
      return foldedAttr;

    // Otherwise fallback to KGEN's folding hooks.
    return KGEN::ParamOperatorAttr::get(opcode, foldedOperands);
  }

  /// If `attr` is a `mosh.assert false`, return the error message associated
  /// with it, else return null.
  static StringAttr tryGetAssertFalseErrMsg(TypedAttr attr) {
    // Check it's an expression attribute.
    auto paramOperatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
    if (!paramOperatorAttr)
      return {};

    // Check it's a `mosh.assert`.
    auto mlirOpAttr =
        dyn_cast<KGEN::MLIROpAttr>(paramOperatorAttr.getOperand(0));
    if (!mlirOpAttr ||
        mlirOpAttr.getName() != MOSH::AssertOp::getOperationName())
      return {};

    // Check assert condition is a static false.
    if (auto cond = dyn_cast<BoolAttr>(paramOperatorAttr.getOperand(1));
        cond && cond.getValue() == false)
      return dyn_cast<StringAttr>(mlirOpAttr.getAttrs().get("message"));

    return {};
  }

  /// Try decomposing `op` with `opDecomposer`. If successful, replace `op` with
  /// the decomposed ops in the graph, and push the decomposed ops onto
  /// `worklist` so they will be processed next immediately.
  ///
  /// Return whether the decomposition was successful.
  static LogicalResult
  tryDecomposeAndReplaceOp(OpDecomposer &opDecomposer, Operation *op,
                           std::deque<Operation *> &worklist) {

    // Try decompose the op, and check if it was successful.
    SmallVector<Operation *> decomposedOps = opDecomposer.tryDecomposeOp(op);
    if (decomposedOps.empty())
      return failure();

    OpBuilder builder(op);
    // Replace the original op with decomposed ops in correct order.
    for (Operation *decomposedOp : decomposedOps)
      builder.insert(decomposedOp);
    op->erase();

    // The first decomposed op must be processed next in the worklist, so we
    // push the decomposed ops onto the worklist in reversed order.
    for (Operation *decomposedOp : llvm::reverse(decomposedOps))
      worklist.push_back(decomposedOp);

    return success();
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
    // evaluator won't panic when looking them up. Note that (1) and (2) are
    // handled by default in `KGEN::ParameterEvaluator`, and (3) is handled in
    // the worklist loop below.
    KGEN::ParameterEvaluator evaluator;
    auto markIntParamDeclAsNonNegative = [&](KGEN::ParamDeclAttr paramDecl) {
      if (paramDecl.getType().isIntOrIndex())
        markDefinitelyNonNegative(KGEN::ParamDeclRefAttr::get(paramDecl));
    };
    // The integer input parameters to graph are dynamic dimensions, so they
    // must be non-negative.
    llvm::for_each(graphOp.getInputParams(), markIntParamDeclAsNonNegative);

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

    // NOTE think of this as a stack machine that also decomposes "pseudo
    // instructions".
    OpDecomposer opDecomposer(graphOp.getContext());
    while (!worklist.empty()) {
      Operation *op = worklist.back();
      worklist.pop_back();

      // Use accumulated param-to-expression mapping to simplify the op first.
      // This is needed to help DCE get rid of the decls with simplified exprs.
      replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                 /*replaceLocs=*/true, /*replaceTypes=*/true);

      // If decomposition happened, no need to process `op` anymore, instead
      // continue to process the decomposed ops which were pushed onto worklist.
      if (succeeded(tryDecomposeAndReplaceOp(opDecomposer, op, worklist)))
        continue;

      // Every single `kgen.param.declare x = <expr>` should add a mapping from
      // `x` to some attribute in `evaluator`.
      if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
        KGEN::ParamDeclAttr declAttr = declOp.getParamDecl();
        auto foldedRhsAttr = cast<TypedAttr>(declOp.getValue());

        // Apply shape-inference specific folding
        foldedRhsAttr = tryFoldExprAttr(foldedRhsAttr);
        declOp.setValueAttr(foldedRhsAttr);

        auto paramOperatorAttr =
            dyn_cast<KGEN::ParamOperatorAttr>(foldedRhsAttr);
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
            declOp.setValueAttr(existingDeclRef);

            // Else we remember this decl has this expression, and map the
            // parameter to itself since we don't want to inline it.
          } else {
            auto declRefAttr = KGEN::ParamDeclRefAttr::get(declAttr);
            if (isParamOperatorPure(paramOperatorAttr))
              exprToDeclRef.try_emplace(paramOperatorAttr, declRefAttr);
            evaluator.setParameterValue(declAttr.getName(), declRefAttr);
          }

        } else {
          // The param expression can be nested, so we just record it as is.
          evaluator.setParameterValue(declAttr, foldedRhsAttr);
        }

        // Emit compile time error for any `mosh.assert false`
        if (StringAttr errMsg = tryGetAssertFalseErrMsg(foldedRhsAttr)) {
          op->emitError(errMsg.strref());
          return signalPassFailure();
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
  DenseSet<TypedAttr> definitelyNonNegativeAttrs;
};
