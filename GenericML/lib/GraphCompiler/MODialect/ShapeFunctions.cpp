//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/Support/Assert.h"
#include "KGEN/KGENDialect/KGENOps.h"

namespace M::MO {

Var ParamExprBuilder::createVar(TypedAttr attr) {
  auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr);
  if (!declRef)
    declRef = this->createDeclToKgenAttr(attr, "var");
  return Var(declRef, this);
}

Var ParamExprBuilder::createVar(int64_t n) {
  return createVar(IntegerAttr::get(builder.getIndexType(), n));
}

KGEN::ParamDeclRefAttr ParamExprBuilder::createDeclToKgenExpr(
    KGEN::POC poc, ArrayRef<TypedAttr> operands, StringRef declPrefix) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(poc, operands);
  return createDeclToKgenAttr(exprAttr, declPrefix);
};

KGEN::ParamDeclRefAttr
ParamExprBuilder::createDeclToKgenAttr(TypedAttr attr, StringRef declPrefix) {
  Type resTy = attr.getType();
  KGEN::ParamDeclAttr declAttr = graphOp.getNewParamOfType(declPrefix, resTy);
  auto declOp = builder.create<KGEN::ParamDeclareOp>(loc, declAttr, attr);
  auto declRefAttr = KGEN::ParamDeclRefAttr::get(declOp.getParamDecl());
  createdOps->push_back(declOp);
  return declRefAttr;
};

Var ParamExprBuilder::createCond(const Var &condV, const Var &thenV,
                                 const Var &elseV) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(
      KGEN::POC::Cond, {condV.getAttr(), thenV.getAttr(), elseV.getAttr()});
  return createVar(createDeclToKgenAttr(exprAttr, "cond"));
};

ShapeAttr ParamExprBuilder::getShapeAttr(ArrayRef<TypedAttr> dimAttrs) {
  return ShapeAttr::get(dimAttrs, ShapeType::get(graphOp.getContext()));
}

void ParamExprBuilder::createAssert(const Var &condVar, StringRef errorMsg) {
  TypedAttr condAttr = condVar.getAttr();
  MLIRContext *ctx = condAttr.getContext();
  Type chainType = MOSH::ChainType::get(ctx);

  // Build a new operand list which starts with MLIROpAttr.
  auto signature = SignatureType::get(ctx, {condAttr.getType()}, chainType);
  NamedAttribute msgAttr(StringAttr::get(ctx, "message"),
                         StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
  auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
  auto opNameAttr = StringAttr::get(ctx, MOSH::AssertOp::getOperationName());
  SmallVector<TypedAttr> applyOperands{
      KGEN::MLIROpAttr::get(opNameAttr, dictAttr, signature), condAttr};

  (void)createDeclToKgenExpr(KGEN::POC::Apply, applyOperands, "assert");
};

/// Try converting `attr` to a constant integer.
std::optional<int64_t> tryGetConstantInt(TypedAttr attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt();
  return std::nullopt;
}

//----------------------------------------------------------------------------//
//------------------------Shape functions begin-------------------------------//

// NOTE these helpers will eventually be shared between shape inference and
// GraphAPI shape propagation.
//
// These helpers have pretty uniform input types, where
// - `GraphOp` is used for new parameter name generation
// - `Location` is used for newly created instructions
// - All the `TypedAttr`s are parameteric representation of the op's input

/// Materialize the shape logic for `MO::BatchMatmulOp`, assuming given
ShapeMaterializeResult materializeBatchMatmulShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr lhsShapeAttr,
                                                   TypedAttr rhsShapeAttr) {
  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr lhsShape = dyn_cast<ShapeAttr>(lhsShapeAttr);
  ShapeAttr rhsShape = dyn_cast<ShapeAttr>(rhsShapeAttr);
  if (!lhsShape || !rhsShape)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = lhsShape.getRank();
  if (rhsShape.getRank() != inputRank || inputRank <= 2) {
    return ShapeMaterializeResult::error(
        "[batch_matmul] input ranks must match and exceed 2");
  }

  SmallVector<TypedAttr> resDims;
  // Check batch dimensions are equal and collect them into `resDims`.
  for (size_t batchDim = 0; batchDim < inputRank - 2; batchDim++) {
    Var lhsDim = b.createVar(lhsShape[batchDim]);
    Var rhsDim = b.createVar(rhsShape[batchDim]);
    b.createAssert(lhsDim == rhsDim,
                   "[batch_matmul] inputs batch dimensions must match");
    resDims.push_back(lhsDim.getAttr());
  }

  // Check inner dimensions are equal
  Var lhsInnerDim = b.createVar(lhsShape[inputRank - 1]);
  Var rhsInnerDim = b.createVar(rhsShape[inputRank - 2]);
  b.createAssert(lhsInnerDim == rhsInnerDim,
                 "[batch_matmul] inputs inner dimensions must match");

  // Collect outer dims.
  resDims.push_back(lhsShape[inputRank - 2]);
  resDims.push_back(rhsShape[inputRank - 1]);
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

/// Materialize the shape logic for `MO::BroadcastToOp`.
ShapeMaterializeResult materializeBroadcastToShape(GraphOp graphOp,
                                                   Location loc,
                                                   TypedAttr inputShapeAttr,
                                                   TypedAttr targetShapeAttr) {
  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  int inputRank = inputShape.getRank();
  int outputRank = targetShape.getRank();
  if (inputRank > outputRank) {
    return ShapeMaterializeResult::error(
        "[batch_matmul] input ranks must match and exceed 2");
  }

  for (int i = 0; i < inputRank; i++) {
    int inputAxis = inputRank - i - 1;
    int outputAxis = outputRank - i - 1;
    Var inputDim = b.createVar(inputShape[inputAxis]);
    Var outputDim = b.createVar(targetShape[outputAxis]);

    // Insert assertions
    b.createAssert(inputDim == 1 || inputDim == outputDim,
                   "[broadcast_to] input dim must be either 1 or equal to "
                   "corresponding output dim starting from the rightmost dim");
  }
  return ShapeMaterializeResult::success(std::move(ops), targetShape);
}

/// Materialize the shape logic for `MO::ConcatOp`.
ShapeMaterializeResult
materializeConcatShape(GraphOp graphOp, Location loc, TypedAttr axisAttr,
                       ArrayRef<TypedAttr> inputShapeAttrs) {
  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Check for constant axis
  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  // Check for statically-ranked input shapes.
  SmallVector<ShapeAttr> inputShapes;
  for (TypedAttr inputShapeAttr : inputShapeAttrs) {
    if (auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      inputShapes.push_back(inputShape);
    } else {
      // Give up on any dyn-ranked input shape
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  // `MO::ConcatOp` verifier ensures there's at least 1 input, and ranks of
  // inputs all match.
  ShapeAttr firstInputShape = inputShapes.front();
  auto restInputShapes = ArrayRef(inputShapes).drop_front();
  int64_t inputRank = firstInputShape.getRank();

  // Normalize axis and make sure it's in bound.
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= axis) {
    return ShapeMaterializeResult::error(
        "[concat] normalized axis must be within range [0, input_rank)");
  }

  // Construct output shape and make sure input shapes are compatible.
  SmallVector<TypedAttr> resDims(firstInputShape.getValues());
  Var concatDimSum = b.createVar(resDims[axis]);
  for (ShapeAttr inputShape : restInputShapes) {
    for (auto [idx, firstInputDimAttr, inputDimAttr] :
         llvm::enumerate(firstInputShape.getValues(), inputShape.getValues())) {
      if (static_cast<int64_t>(idx) != axis) {
        Var firstInputDim = b.createVar(firstInputDimAttr);
        Var inputDim = b.createVar(inputDimAttr);
        b.createAssert(
            firstInputDim == inputDim,
            "[concat] input shapes must match except at concat axis");
      }
    }
    Var inputAxisDim = b.createVar(inputShape[axis]);
    concatDimSum = concatDimSum + inputAxisDim;
  }
  resDims[axis] = concatDimSum.getAttr();
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

/// Materialize the shape logic for `MO::RangeOp`.
ShapeMaterializeResult materializeRangeShape(GraphOp graphOp, Location loc,
                                             TypedAttr startAttr,
                                             TypedAttr stopAttr,
                                             TypedAttr stepAttr) {
  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  Var start = b.createVar(startAttr);
  Var stop = b.createVar(stopAttr);
  Var step = b.createVar(stepAttr);

  // Check step != 0
  b.createAssert(step != 0, "[range] step must be non-zero");

  // Check start/stop are in bounds
  b.createAssert(start <= stop || step < 0,
                 "[range] requires (start <= stop) for positive step");
  b.createAssert(stop <= start || step > 0,
                 "[range] requires (stop <= start) for negative step");

  // Compute output shape by simulating a ceiling-div -- think the length of
  // range(0, 3, 2).
  Var diff = stop - start;
  Var offset = b.createCond(step > 0, step - 1, step + 1);
  Var outDim = (diff + offset) / step;
  ShapeAttr outputShape = b.getShapeAttr(outDim);
  return ShapeMaterializeResult::success(std::move(ops), outputShape);
}

/// Materialize the shape logic for `MO::ReshapeOp`.
///
/// Specifically, if target shape contains a `-1`, infer it based on the
/// invariant that `inputShapeAttr` and `targetShapeAttr` represent equal number
/// of elements.
///
/// Example:
///   inputShape = [3, D4]
///   targetShape = [D0, -1, 42]
///   returns [D0, RS0, 42]
/// with the following operation accumulated in `b`
///   kgen.param.declare RS0 = <div(mul(3, D4), mul(D0, 42)>
ShapeMaterializeResult materializeReshapeShape(GraphOp graphOp, Location loc,
                                               TypedAttr inputShapeAttr,
                                               TypedAttr targetShapeAttr) {
  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Make sure both input and target shapes are ranked symbolic shapes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr targetShape = dyn_cast<ShapeAttr>(targetShapeAttr);
  if (!inputShape || !targetShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Find index of `-1` dimension (if any), and collect all other dimensions.
  SmallVector<TypedAttr> targetNonNegOneDims;
  std::optional<size_t> maybeTargetNegOneDimIdx = std::nullopt;
  for (auto [idx, targetDim] : llvm::enumerate(targetShape.getValues())) {
    auto intDim = dyn_cast<IntegerAttr>(targetDim);
    if (intDim && intDim.getInt() == -1) {
      // Detected multiple negative one, gives up and let runtime handle the
      // error via a shape function call.
      if (maybeTargetNegOneDimIdx.has_value()) {
        return ShapeMaterializeResult::error(
            "[reshape] multiple -1 detected in target shape");
      }

      maybeTargetNegOneDimIdx = idx;
    } else {
      // TODO: we are using the assumption that all non-integer attribute
      // (e.g., `D0` or `mul(D0, D1)`) will be non-negative. To make this
      // completely safe, we'll need to extend and leverage the range
      // analysis here.
      targetNonNegOneDims.push_back(targetDim);
    }
  }

  // resolve the `-1` in output shape if needed.
  ShapeAttr outputShape = targetShape;
  if (maybeTargetNegOneDimIdx.has_value()) {
    // Now infer the actual value of `-1` dimension based on other dimensions
    Var numerator = b.createVar(inputShape.getSymbolicNumElements());

    // Add a 1 to the shapes to multiply to ensure the denominator is valid
    auto mulDims = targetNonNegOneDims;
    mulDims.emplace_back(IntegerAttr::get(numerator.getAttr().getType(), 1));
    Var denominator =
        b.createVar(KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw, mulDims));

    // Make sure we check for 0 denominator
    b.createAssert(denominator != 0,
                   "[reshape] cannot infer dimension when a specified output "
                   "dimension is 0");

    Var resolvedDim = numerator / denominator;
    SmallVector<TypedAttr> resolvedDims = targetNonNegOneDims;
    resolvedDims.insert(resolvedDims.begin() + maybeTargetNegOneDimIdx.value(),
                        resolvedDim.getAttr());
    outputShape = b.getShapeAttr(resolvedDims);
  }

  // Insert assertions
  Var inputNumeElems = b.createVar(inputShape.getSymbolicNumElements());
  Var outputNumeElems = b.createVar(outputShape.getSymbolicNumElements());
  b.createAssert(inputNumeElems == outputNumeElems,
                 "[reshape] input and output number of elements must match");
  return ShapeMaterializeResult::success(std::move(ops), outputShape);
}

ShapeMaterializeResult materializeSplitIthOutputShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr splitSizesAttr, TypedAttr axisAttr, size_t i) {
  // Make sure input is ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr splitSizes = dyn_cast<ShapeAttr>(splitSizesAttr);
  if (!inputShape || !splitSizes)
    return ShapeMaterializeResult::nonMaterializable();

  // Check for i
  if (i >= splitSizes.getValues().size()) {
    return ShapeMaterializeResult::error(
        "[split] output index must be within range [0, len(split_sizes))");
  }

  // Check for constant axis
  std::optional<int64_t> axisOr = tryGetConstantInt(axisAttr);
  if (!axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t axis = axisOr.value();

  // Normalize axis and check it's in range.
  size_t inputRank = inputShape.getRank();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[split] normalized axis must be within range [0, rank)");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Check for sum of splitSizes.
  Var splitSizesSum = b.createVar(0);
  for (TypedAttr splitSize : splitSizes.getValues())
    splitSizesSum = splitSizesSum + b.createVar(splitSize);
  b.createAssert(
      splitSizesSum == b.createVar(inputShape[axis]),
      "[split] sum of split sizes must match input dimension at split axis");

  // Create output shape.
  SmallVector<TypedAttr> outDims(inputShape.getValues());
  outDims[axis] = splitSizes[i];
  ShapeAttr outputShape = b.getShapeAttr(outDims);
  return ShapeMaterializeResult::success(std::move(ops), outputShape);
}

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

// TODO move the following into a new interface.

// Assume `inputs` are staticized inputs for a `MO::BroadcastToOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeBroadcastToShapeImpl(ArrayRef<TypedAttr> inputs,
                                DictionaryAttr kgenParams, GraphOp graphOp,
                                Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr targetShape = inputs[1];
  return materializeBroadcastToShape(graphOp, loc, inputShape, targetShape);
}

// Assume `inputs` are staticized inputs for a `MO::ReshapeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeReshapeShapeImpl(ArrayRef<TypedAttr> inputs,
                            DictionaryAttr kgenParams, GraphOp graphOp,
                            Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr targetShape = inputs[1];
  return materializeReshapeShape(graphOp, loc, inputShape, targetShape);
}

// Assume `inputs` are staticized inputs for a `MO::RangeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeRangeShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                          GraphOp graphOp, Location loc) {
  TypedAttr startAttr = inputs[0];
  TypedAttr stopAttr = inputs[1];
  TypedAttr stepAttr = inputs[2];
  return materializeRangeShape(graphOp, loc, startAttr, stopAttr, stepAttr);
}

// Assume `inputs` are staticized inputs for a `MO::BatchMatmulOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeBatchMatmulShapeImpl(ArrayRef<TypedAttr> inputs,
                                DictionaryAttr kgenParams, GraphOp graphOp,
                                Location loc) {
  auto lhsShape = inputs[0];
  auto rhsShape = inputs[1];
  return materializeBatchMatmulShape(graphOp, loc, lhsShape, rhsShape);
}

// Assume `inputs` are staticized inputs for a `MO::ConcatOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeConcatShapeImpl(ArrayRef<TypedAttr> inputs,
                           DictionaryAttr kgenParams, GraphOp graphOp,
                           Location loc) {
  TypedAttr axis = inputs[0];
  ArrayRef<TypedAttr> inputShapes = inputs.drop_front();
  return materializeConcatShape(graphOp, loc, axis, inputShapes);
}

// Assume `inputs` are staticized inputs for a `MO::SplitiOp`, try
// materializing the its shape logic for
static ShapeMaterializeResult
materializeSplitIthOutputShapeImpl(ArrayRef<TypedAttr> inputs,
                                   DictionaryAttr kgenParams, GraphOp graphOp,
                                   Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr splitSizes = inputs[1];
  TypedAttr axis = inputs[2];

  GENERICML_ASSERT(kgenParams)
      << "kgenParams must be available for split_ith_output_shape";

  auto intAttr = dyn_cast_or_null<IntegerAttr>(kgenParams.get("output_idx"));
  GENERICML_ASSERT(intAttr) << "kgenParams must contain an 'output_idx' "
                               "integer for split_ith_output_shape";

  return materializeSplitIthOutputShape(graphOp, loc, inputShape, splitSizes,
                                        axis, intAttr.getInt());
}

std::optional<MaterializeShapeFuncTy>
tryGetShapeMaterializeFunc(StringRef shapeFuncName) {
  // A hard-coded map from shape function name to how their shape logic can be
  // materialized.
  static DenseMap<StringRef, MaterializeShapeFuncTy> nameToFunc = {
      {"batch_matmul_shape", materializeBatchMatmulShapeImpl},
      {"broadcast_to_shape", materializeBroadcastToShapeImpl},
      {"concat_shape", materializeConcatShapeImpl},
      {"range_shape", materializeRangeShapeImpl},
      {"reshape_shape", materializeReshapeShapeImpl},
      {"split_ith_output_shape", materializeSplitIthOutputShapeImpl},
  };

  auto iter = nameToFunc.find(shapeFuncName);
  if (iter == nameToFunc.end())
    return std::nullopt;
  return iter->getSecond();
}

} // namespace M::MO
