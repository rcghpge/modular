//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"

#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/Support/Assert.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include <optional>
#include <utility>

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

Var ParamExprBuilder::createMax(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Max, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(createDeclToKgenAttr(exprAttr, "max"));
}

Var ParamExprBuilder::createMin(const Var &lhs, const Var &rhs) {
  auto exprAttr = KGEN::ParamOperatorAttr::get(KGEN::POC::Min, lhs.getAttr(),
                                               rhs.getAttr());
  return createVar(createDeclToKgenAttr(exprAttr, "min"));
}

ShapeAttr ParamExprBuilder::getShapeAttr(ArrayRef<TypedAttr> dimAttrs) {
  return ShapeAttr::get(dimAttrs, ShapeType::get(graphOp.getContext()));
}

void ParamExprBuilder::createAssert(const Var &condVar, StringRef errorMsg) {
  TypedAttr condAttr = condVar.getAttr();
  MLIRContext *ctx = condAttr.getContext();
  Type chainType = MO::ChainType::get(ctx);

  // Build a new operand list which starts with MLIROpAttr.
  auto signature = SignatureType::get(ctx, {condAttr.getType()}, chainType);
  NamedAttribute msgAttr(StringAttr::get(ctx, "message"),
                         StringAttr::get(errorMsg, KGEN::StringType::get(ctx)));
  auto dictAttr = DictionaryAttr::get(ctx, {msgAttr});
  auto opNameAttr = StringAttr::get(ctx, MO::AssertOp::getOperationName());
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

/// Compute the output spatial dimension for a sliding window op (e.g.,
/// convolution or pooling).
static Var getSlidingWindowOutDim(const Var &inputDim, const Var &filterDim,
                                  const Var &dilation, const Var &stride,
                                  const Var &padding, bool ceilMode = false) {
  Var numerator = inputDim + padding - (dilation * (filterDim - 1) + 1);
  if (ceilMode)
    return (numerator + stride - 1) / stride + 1;
  return numerator / stride + 1;
}

// Compute the size of a range (assuming validity of input values).
static Var computeRangeSize(const Var &start, const Var &stop,
                            const Var &step) {
  // We need to be careful with offset and negatives, think range(0, 3, 2) and
  // range(0, -3, -2);
  ParamExprBuilder &b = start.getBuilder();
  Var diff = stop - start;
  Var offset = b.createCond(step > 0, step - 1, step + 1);
  Var size = (diff + offset) / step;
  return size;
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

ShapeMaterializeResult
materializeConvShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterShapeAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     TypedAttr numGroupsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  // No need to check for `numGroupsAttr` because it's just an integer.
  if (!inputShape || !filterShape || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank < 3) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank >= 3)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (input_rank == filter_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(strides) == len(dilations) == input_rank "
        "- 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[convolution] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  //   input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);
  Var filterChannels = b.createVar(filterShape[inputRank - 2]);
  Var outputChannels = b.createVar(filterShape[inputRank - 1]);
  Var numGroups = b.createVar(numGroupsAttr);

  // More checks on channels and numGroups.
  b.createAssert(inputChannels == (numGroups * filterChannels),
                 "[convolution] requires (input_channels == num_groups * "
                 "filter_channels)");
  b.createAssert(
      (outputChannels % numGroups) == 0,
      "[convolution] output_channels must be divisible by num_groups");

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filterShape[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding);
    b.createAssert(outputSpatialDim > 0,
                   "[convolution] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(outputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult materializeConvTransposeShape(
    GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
    TypedAttr filterShapeAttr, TypedAttr stridesAttr, TypedAttr dilationsAttr,
    TypedAttr paddingsAttr, TypedAttr outputPaddingsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filterShape = dyn_cast<ShapeAttr>(filterShapeAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  auto outPaddings = dyn_cast<ShapeAttr>(outputPaddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  // No need to check for `numGroupsAttr` because it's just an integer.
  if (!inputShape || !filterShape || !strides || !dilations || !paddings ||
      !outPaddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank != 4 && inputRank != 5) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == 4 or 5)");
  }
  if (inputRank != filterShape.getRank()) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (input_rank == kernel_rank)");
  }
  if (strides.getSize() != inputRank - 2 ||
      dilations.getSize() != inputRank - 2) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  // input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  //
  // BUT since ConvTranspose is the inverse of Conv w.r.t. the input, the output
  // shape would have layout [batch_size, ...spatial_dims..., filter_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);
  Var filterChannels = b.createVar(filterShape[inputRank - 2]);
  Var outputChannels = b.createVar(filterShape[inputRank - 1]);

  // Check input and kernel channels.
  b.createAssert(inputChannels == outputChannels,
                 "[conv_transpose] requires input channel to match output "
                 "channel in kernel");

  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());

  // Compute a specific output spatial dimension.
  auto computeOutDim = [](const Var &inputDim, const Var &filterDim,
                          const Var &dilation, const Var &stride,
                          const Var &prePad, const Var &postPad,
                          const Var &outPad) -> Var {
    return stride * (inputDim - 1) + outPad + ((filterDim - 1) * dilation + 1) -
           prePad - postPad;
  };

  // Compute and collect output spatial dimensions;
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filterShape[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePad = b.createVar(paddings[2 * axis]);
    Var postPad = b.createVar(paddings[2 * axis + 1]);
    Var outPad = b.createVar(outPaddings[axis]);

    Var outputSpatialDim =
        computeOutDim(inputSpatialDim, filterSpatialDim, dilation, stride,
                      prePad, postPad, outPad);
    b.createAssert(outputSpatialDim > 0,
                   "[conv_transpose] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(filterChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult materializePadShape(GraphOp graphOp, Location loc,
                                           TypedAttr inputShapeAttr,
                                           TypedAttr paddingsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure input shape is ranked and paddings is a list of symbolic integers
  // with known length.
  if (!inputShape || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (2 * inputRank != paddings.getSize()) {
    return ShapeMaterializeResult::error(
        "[pad] paddings shape must be (input_rank, 2)");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);
    Var outputDim = inputDim + prePadding + postPadding;
    resDims.push_back(outputDim.getAttr());
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult
materializePoolShape(GraphOp graphOp, Location loc, TypedAttr inputShapeAttr,
                     TypedAttr filterAttr, TypedAttr stridesAttr,
                     TypedAttr dilationsAttr, TypedAttr paddingsAttr,
                     bool ceilMode) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto filter = dyn_cast<ShapeAttr>(filterAttr);
  auto strides = dyn_cast<ShapeAttr>(stridesAttr);
  auto dilations = dyn_cast<ShapeAttr>(dilationsAttr);
  auto paddings = dyn_cast<ShapeAttr>(paddingsAttr);
  // Make sure both input and target shapes are ranked symbolic shapes.
  if (!inputShape || !filter || !strides || !dilations || !paddings) {
    return ShapeMaterializeResult::nonMaterializable();
  }

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  // NOTE it's a trivial extension to enable this on arbitrary ranks, but I'm
  // keeping it the same as the mojo shape function for now.
  if (inputRank != 4) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (input_rank == 4)");
  }

  size_t filterSize = filter.getSize();
  if (filterSize != inputRank - 2 ||
      !llvm::all_equal({filterSize, dilations.getSize(), strides.getSize()})) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(filter) == len(strides) == len(dilations) == "
        "input_rank - 2)");
  }
  if (paddings.getSize() != 2 * (inputRank - 2)) {
    return ShapeMaterializeResult::error(
        "[pooling] requires (len(paddings) == 2 * (input rank - 2))");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // NOTE
  // - input and output have layout [batch_size, ...spatial_dims...,
  // input_channels]
  // - filter has layout [...spatial_dims..., filter_channels, output_channels]
  Var batchSize = b.createVar(inputShape[0]);
  Var inputChannels = b.createVar(inputShape[inputRank - 1]);

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> resDims;
  resDims.push_back(batchSize.getAttr());
  for (size_t axis = 0; axis < inputRank - 2; axis++) {
    Var inputSpatialDim = b.createVar(inputShape[axis + 1]);
    Var filterSpatialDim = b.createVar(filter[axis]);
    Var dilation = b.createVar(dilations[axis]);
    Var stride = b.createVar(strides[axis]);
    Var prePadding = b.createVar(paddings[2 * axis]);
    Var postPadding = b.createVar(paddings[2 * axis + 1]);

    Var outputSpatialDim =
        getSlidingWindowOutDim(inputSpatialDim, filterSpatialDim, dilation,
                               stride, prePadding + postPadding, ceilMode);
    b.createAssert(outputSpatialDim > 0,
                   "[pooling] output spatial dim must be positive");
    resDims.push_back(outputSpatialDim.getAttr());
  }

  // Create output shape
  resDims.push_back(inputChannels.getAttr());
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult materializeGatherShape(GraphOp graphOp, Location loc,
                                              TypedAttr inputShapeAttr,
                                              TypedAttr indicesShapeAttr,
                                              TypedAttr axisAttr) {
  // Make sure input and indices shapes are ranked, and axis is constant.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  auto axisOr = tryGetConstantInt(axisAttr);
  if (!inputShape || !indicesShape || !axisOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();

  // Normalize constant and make sure it's in bound.
  size_t inputRank = inputShape.getRank();
  int64_t axis = axisOr.value();
  axis = axis < 0 ? axis + inputRank : axis;
  if (axis < 0 || inputRank <= static_cast<size_t>(axis)) {
    return ShapeMaterializeResult::error(
        "[gather] normalized axis must be within range [0, input_rank)");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output dimensions.
  // Note the output shape is essentially replacing the `axis` dimension of
  // `input_shape` with `indices_shape`, i.e.,
  //   output_shape := input_shape[:axis] + indices_shape + input_shape[axis+1:]
  SmallVector<TypedAttr> resDims;
  // input_shape[:axis]
  for (int64_t idx = 0; idx < axis; idx++)
    resDims.push_back(inputShape[idx]);
  // indices_shape
  llvm::append_range(resDims, indicesShape.getValues());
  // input_shape[axis+1:]
  for (size_t idx = axis + 1; idx < inputRank; idx++)
    resDims.push_back(inputShape[idx]);

  // Create and return output shape.
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult materializeGatherNdShape(GraphOp graphOp, Location loc,
                                                TypedAttr inputShapeAttr,
                                                TypedAttr indicesShapeAttr,
                                                int64_t batchDims) {
  // Make sure input and indices shapes are ranked.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr indicesShape = dyn_cast<ShapeAttr>(indicesShapeAttr);
  if (!inputShape || !indicesShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks
  size_t inputRank = inputShape.getRank();
  size_t indicesRank = indicesShape.getRank();
  if (inputRank < 1 || indicesRank < 1) {
    return ShapeMaterializeResult::error(
        "[gather_nd] input_rank and indices_rank must be >= 1");
  }

  // Size of the indexing dimension must be a constant
  auto idxSizeOr = tryGetConstantInt(indicesShape[indicesRank - 1]);
  if (!idxSizeOr.has_value())
    return ShapeMaterializeResult::nonMaterializable();
  int64_t idxSize = idxSizeOr.value();

  // More shape checks.
  if (idxSize < 1 || inputRank - batchDims < static_cast<size_t>(idxSize)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] index size must be within range [1, input_rank - "
        "batch_dims]");
  }
  if (batchDims >= static_cast<int64_t>(indicesRank)) {
    return ShapeMaterializeResult::error(
        "[gather_nd] requires (batch_dims < indices_rank)");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output dimensions.
  //   output_shape = input_shape[:batch_dims]
  //                  + indices_shape[batch_dims:-1]
  //                  + input_shape[batch_dims + indices_shape[-1]:]
  SmallVector<TypedAttr> resDims;

  // input_shape[:batch_dims]
  for (int64_t idx = 0; idx < batchDims; idx++)
    resDims.push_back(inputShape[idx]);
  // indices_shape[batch_dims:-1]
  llvm::append_range(resDims,
                     indicesShape.getValues().slice(batchDims).drop_back());
  // input_shape[batch_dims + indices_shape[-1]:]
  for (size_t idx = batchDims + idxSize; idx < inputRank; idx++)
    resDims.push_back(inputShape[idx]);

  // Create and return output shape.
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

  Var outDim = computeRangeSize(start, stop, step);
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

/// Materialize the shape logic for `MO::SliceOp`.
ShapeMaterializeResult materializeSliceShape(GraphOp graphOp, Location loc,
                                             TypedAttr inputShapeAttr,
                                             TypedAttr startsAttr,
                                             TypedAttr stopsAttr,
                                             TypedAttr stepsAttr) {
  // Make sure input shape is ranked, and bounds are list of symbolic integers
  // with known sizes.
  ShapeAttr inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  ShapeAttr starts = dyn_cast<ShapeAttr>(startsAttr);
  ShapeAttr stops = dyn_cast<ShapeAttr>(stopsAttr);
  ShapeAttr steps = dyn_cast<ShapeAttr>(stepsAttr);
  if (!inputShape || !starts || !stops || !steps)
    return ShapeMaterializeResult::nonMaterializable();

  size_t inputRank = inputShape.getRank();
  if (starts.getSize() != inputRank ||
      !llvm::all_equal({starts.getSize(), stops.getSize(), steps.getSize()})) {
    return ShapeMaterializeResult::error(
        "Input rank must match the number of elements in starts/stops/steps "
        "bounds");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output spatial dimensions.
  SmallVector<TypedAttr> outDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inDim = b.createVar(inputShape[axis]);
    Var start = b.createVar(starts[axis]);
    Var stop = b.createVar(stops[axis]);
    Var step = b.createVar(steps[axis]);

    // Check step != 0
    b.createAssert(step != 0, "[slice] step must be non-zero");

    // Compute the min/max for clamping start/end
    Var idxMin = b.createCond(step > 0, b.createVar(0), b.createVar(-1));
    Var idxMax = b.createCond(step > 0, inDim, inDim - 1);

    // Normalize and clamp start/stop values.
    start = b.createCond(start < 0, start + inDim, start);
    start = b.createMax(idxMin, b.createMin(idxMax, start));
    stop = b.createCond(stop < 0, stop + inDim, stop);
    stop = b.createMax(idxMin, b.createMin(idxMax, stop));

    // Make sure start and stop don't cross each other.
    b.createAssert(step < 0 || start <= stop,
                   "[slice] normalized stop cannot be smaller than start for "
                   "positive step");
    b.createAssert(step > 0 || start >= stop,
                   "[slice] normalized start cannot be smaller than stop for "
                   "negative step");

    Var outDim = computeRangeSize(start, stop, step);
    outDims.push_back(outDim.getAttr());
  }

  ShapeAttr outputShape = b.getShapeAttr(outDims);
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

ShapeMaterializeResult materializeTileShape(GraphOp graphOp, Location loc,
                                            TypedAttr inputShapeAttr,
                                            TypedAttr repeatsAttr) {
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  auto repeats = dyn_cast<ShapeAttr>(repeatsAttr);
  // Make sure input shape is ranked and repeats is a list of symbolic integers
  // with known length.
  if (!inputShape || !repeats)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks.
  size_t inputRank = inputShape.getRank();
  if (inputRank != repeats.getSize()) {
    return ShapeMaterializeResult::error(
        "[tile] requires (len(repeats) == input_rank)");
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    Var inputDim = b.createVar(inputShape[axis]);
    Var repeat = b.createVar(repeats[axis]);
    Var outputDim = inputDim * repeat;
    resDims.push_back(outputDim.getAttr());
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

ShapeMaterializeResult materializeTransposeShape(GraphOp graphOp, Location loc,
                                                 TypedAttr inputShapeAttr,
                                                 TypedAttr permutationsAttr) {
  // Make sure input shape is ranked.
  auto inputShape = dyn_cast<ShapeAttr>(inputShapeAttr);
  if (!inputShape)
    return ShapeMaterializeResult::nonMaterializable();

  // Preliminary rank checks.
  auto permutations = dyn_cast<ShapeAttr>(permutationsAttr);
  size_t inputRank = inputShape.getRank();
  if (inputRank != permutations.getSize()) {
    return ShapeMaterializeResult::error(
        "[transpose] permutation size must match input rank");
  }

  // Make sure permutations is a list of constant integers with known length.
  SmallVector<int64_t> permInts;
  for (TypedAttr permAttr : permutations.getValues()) {
    if (auto permInt = tryGetConstantInt(permAttr)) {
      permInts.push_back(permInt.value());
    } else {
      return ShapeMaterializeResult::nonMaterializable();
    }
  }

  // Create builder and track generated ops.
  SmallVector<Operation *> ops;
  ParamExprBuilder b(graphOp, loc, &ops);

  // Compute and collect output dimensions.
  SmallVector<TypedAttr> resDims;
  for (size_t axis = 0; axis < inputRank; axis++) {
    // Retrieve and validate the constant permutation index.
    int64_t perm = permInts[axis];
    if (perm < 0 || inputRank <= static_cast<size_t>(perm)) {
      return ShapeMaterializeResult::error(
          "[transpose] each permutation must be within range [0, rank)");
    }
    // Retrieve the output dimension.
    TypedAttr outputDim = inputShape[perm];
    resDims.push_back(outputDim);
  }

  // Create output shape
  ShapeAttr resShape = b.getShapeAttr(resDims);
  return ShapeMaterializeResult::success(std::move(ops), resShape);
}

//------------------------Shape functions end---------------------------------//
//----------------------------------------------------------------------------//

/// Try to unpack the given attribute as a `KGEN::ParamOperatorAttr` that is
/// representing a `mo.assert`, and return the `(condAttr, errorMsgAttr)` pair.
static std::optional<std::pair<TypedAttr, StringAttr>>
tryGetAssertCondAndMsg(TypedAttr attr) {
  // Check it's an expression attribute.
  auto paramOperatorAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr);
  if (!paramOperatorAttr)
    return std::nullopt;

  // Check it's a `mo.assert`.
  auto mlirOpAttr = dyn_cast<KGEN::MLIROpAttr>(paramOperatorAttr.getOperand(0));
  if (!mlirOpAttr || mlirOpAttr.getName() != MO::AssertOp::getOperationName())
    return std::nullopt;

  // Package the condition attribute and error message.
  TypedAttr cond = paramOperatorAttr.getOperand(1);
  auto msgAttr = mlirOpAttr.getAttrs().getAs<StringAttr>("message");
  return std::make_pair(cond, msgAttr);
}

StringAttr tryGetAssertFalseErrMsg(TypedAttr attr) {
  if (auto condAndMsg = tryGetAssertCondAndMsg(attr)) {
    TypedAttr cond = condAndMsg.value().first;
    StringAttr errMsg = condAndMsg.value().second;

    // Check assert condition is a static false.
    if (auto boolAttr = dyn_cast<BoolAttr>(cond);
        boolAttr && boolAttr.getValue() == false) {
      return errMsg;
    }
  }
  return {};
}

ErrorOr<ShapeAttr> propagateShape(ArrayRef<Operation *> shapeOps,
                                  ShapeAttr targetShape) {

  // Setting things up
  KGEN::ParameterEvaluator evaluator;
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](KGEN::ParamDeclRefAttr declRef) -> Attribute {
    Attribute resolvedAttr = evaluator.getReboundAttribute(declRef);
    return resolvedAttr;
  });

  // Process individual ops and propagate by simple substitution.
  for (Operation *op : shapeOps) {

    // Apply accumulated substitutions to this op.
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/true, /*replaceTypes=*/true);

    if (auto declOp = dyn_cast<KGEN::ParamDeclareOp>(op)) {
      // Process kgen.param.declare LHS = <RHS>
      KGEN::ParamDeclAttr lhsAttr = declOp.getParamDecl();
      TypedAttr rhsAttr = declOp.getValue();

      // Check for assert.
      if (auto condAndMsg = tryGetAssertCondAndMsg(rhsAttr)) {
        TypedAttr cond = condAndMsg.value().first;
        StringAttr errMsg = condAndMsg.value().second;

        // Error if condition didn't fold to true.
        if (auto boolAttr = dyn_cast<BoolAttr>(cond);
            !boolAttr || !boolAttr.getValue()) {
          return Error(errMsg.str());
        }
      }

      // Otherwise remember this substitution.
      evaluator.setParameterValue(lhsAttr, rhsAttr);

    } else {
      // Error if not a `kgen.param.declare`
      return Error("Unexpected op type: " + op->getName().getStringRef());
    }
  }

  // Apply accumulated substitutions to target shape.
  auto resultShape = cast<ShapeAttr>(replacer.replace(targetShape));
  return resultShape;
}

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

// Assume `inputs` are staticized inputs for a `MO::GatherOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeGatherShapeImpl(ArrayRef<TypedAttr> inputs,
                           DictionaryAttr kgenParams, GraphOp graphOp,
                           Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr indicesShape = inputs[1];
  TypedAttr axis = inputs[2];
  return materializeGatherShape(graphOp, loc, inputShape, indicesShape, axis);
}

// Assume `inputs` are staticized inputs for a `MO::GatherNdOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeGatherNdShapeImpl(ArrayRef<TypedAttr> inputs,
                             DictionaryAttr kgenParams, GraphOp graphOp,
                             Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr indicesShape = inputs[1];

  GENERICML_ASSERT(kgenParams)
      << "kgenParams must be available for split_ith_output_shape";
  auto intAttr = dyn_cast_or_null<IntegerAttr>(kgenParams.get("batch_dims"));
  GENERICML_ASSERT(intAttr) << "kgenParams must contain an 'batch_dims' "
                               "integer for gather_nd_shape";

  return materializeGatherNdShape(graphOp, loc, inputShape, indicesShape,
                                  intAttr.getInt());
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

// Assume `inputs` are staticized inputs for a `MO::ConvOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeConvShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                         GraphOp graphOp, Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr filterShape = inputs[1];
  TypedAttr strides = inputs[2];
  TypedAttr dilations = inputs[3];
  TypedAttr paddings = inputs[4];
  TypedAttr numGroup = inputs[5];
  return materializeConvShape(graphOp, loc, inputShape, filterShape, strides,
                              dilations, paddings, numGroup);
}
// Assume `inputs` are staticized inputs for a `MO::ConvTransposeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeConvTransposeShapeImpl(ArrayRef<TypedAttr> inputs,
                                  DictionaryAttr kgenParams, GraphOp graphOp,
                                  Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr filterShape = inputs[1];
  TypedAttr strides = inputs[2];
  TypedAttr dilations = inputs[3];
  TypedAttr paddings = inputs[4];
  TypedAttr outPaddings = inputs[5];
  return materializeConvTransposeShape(graphOp, loc, inputShape, filterShape,
                                       strides, dilations, paddings,
                                       outPaddings);
}

// Assume `inputs` are staticized inputs for a Pooling op that rounds _up_ when
// computing output shapes, try materializing the its shape logic.
static ShapeMaterializeResult
materializePoolCeilShapeImpl(ArrayRef<TypedAttr> inputs,
                             DictionaryAttr kgenParams, GraphOp graphOp,
                             Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr filter = inputs[1];
  TypedAttr strides = inputs[2];
  TypedAttr dilations = inputs[3];
  TypedAttr paddings = inputs[4];
  return materializePoolShape(graphOp, loc, inputShape, filter, strides,
                              dilations, paddings, /*ceilMode=*/true);
}

// Assume `inputs` are staticized inputs for a padding op, try materializing the
// its shape logic.
static ShapeMaterializeResult
materializePadShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                        GraphOp graphOp, Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr paddings = inputs[1];
  return materializePadShape(graphOp, loc, inputShape, paddings);
}

// Assume `inputs` are staticized inputs for a Pooling op that rounds _down_
// when computing output shapes, try materializing the its shape logic.
static ShapeMaterializeResult
materializePoolFloorShapeImpl(ArrayRef<TypedAttr> inputs,
                              DictionaryAttr kgenParams, GraphOp graphOp,
                              Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr filter = inputs[1];
  TypedAttr strides = inputs[2];
  TypedAttr dilations = inputs[3];
  TypedAttr paddings = inputs[4];
  return materializePoolShape(graphOp, loc, inputShape, filter, strides,
                              dilations, paddings, /*ceilMode=*/false);
}

// Assume `inputs` are staticized inputs for a `MO::SliceOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeSliceShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                          GraphOp graphOp, Location loc) {
  TypedAttr input = inputs[0];
  TypedAttr starts = inputs[1];
  TypedAttr stops = inputs[2];
  TypedAttr steps = inputs[3];
  return materializeSliceShape(graphOp, loc, input, starts, stops, steps);
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

// Assume `inputs` are staticized inputs for a `MO::TileOp`, try materializing
// the its shape logic.
static ShapeMaterializeResult
materializeTileShapeImpl(ArrayRef<TypedAttr> inputs, DictionaryAttr kgenParams,
                         GraphOp graphOp, Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr repeats = inputs[1];
  return materializeTileShape(graphOp, loc, inputShape, repeats);
}

// Assume `inputs` are staticized inputs for a `MO::TransposeOp`, try
// materializing the its shape logic.
static ShapeMaterializeResult
materializeTransposeShapeImpl(ArrayRef<TypedAttr> inputs,
                              DictionaryAttr kgenParams, GraphOp graphOp,
                              Location loc) {
  TypedAttr inputShape = inputs[0];
  TypedAttr permutations = inputs[1];
  return materializeTransposeShape(graphOp, loc, inputShape, permutations);
}

std::optional<MaterializeShapeFuncTy>
tryGetShapeMaterializeFunc(StringRef shapeFuncName) {
  // A hard-coded map from shape function name to how their shape logic can be
  // materialized.
  static DenseMap<StringRef, MaterializeShapeFuncTy> nameToFunc = {
      {"batch_matmul_shape", materializeBatchMatmulShapeImpl},
      {"broadcast_to_shape", materializeBroadcastToShapeImpl},
      {"concat_shape", materializeConcatShapeImpl},
      {"conv_shape", materializeConvShapeImpl},
      {"conv_transpose_shape", materializeConvTransposeShapeImpl},
      {"gather_shape", materializeGatherShapeImpl},
      {"gather_nd_shape", materializeGatherNdShapeImpl},
      {"pad_shape", materializePadShapeImpl},
      {"pool_shape", materializePoolFloorShapeImpl},
      {"pool_shape_ceil", materializePoolCeilShapeImpl},
      {"range_shape", materializeRangeShapeImpl},
      {"reshape_shape", materializeReshapeShapeImpl},
      {"slice_shape", materializeSliceShapeImpl},
      {"split_ith_output_shape", materializeSplitIthOutputShapeImpl},
      {"tile_shape", materializeTileShapeImpl},
      {"transpose_shape", materializeTransposeShapeImpl},
  };

  auto iter = nameToFunc.find(shapeFuncName);
  if (iter == nameToFunc.end())
    return std::nullopt;
  return iter->getSecond();
}

} // namespace M::MO
