//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect operations that are analogue with MO.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOOps.h"
#include "GraphCompiler/Support/Dims.h"
#include "Support/AssertStream.h"
#include "Support/ML/RangeUtils.h"
#include <optional>

using namespace M;
using namespace RMO;

using MO::TensorType;
using MO::TensorValue;

//===----------------------------------------------------------------------===//
// RMODialect op support
//===----------------------------------------------------------------------===//

/// Create a new attribute representing a static integer dimension.
static TypedAttr createStaticDimensionType(MO::GraphOp op, int64_t dimSize) {
  return ShapeAttr::getDim(op.getContext(), dimSize);
}

/// Create a new parameter representing a parameterized dimension.
static TypedAttr
createDynamicDimensionType(MO::GraphOp graphOp,
                           SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls,
                           const std::string &dimPrefix = "RMO_D") {
  KGEN::ParamDeclAttr paramDecl = graphOp.getNewDimParam(dimPrefix);
  newParamDecls.emplace_back(paramDecl);
  return KGEN::ParamDeclRefAttr::get(paramDecl);
}

static TensorType
getRankedParamTensorType(int64_t rank, DType dtype, MO::GraphOp graphOp,
                         SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls,
                         DeviceRefAttr deviceRef) {
  SmallVector<TypedAttr> newShape;
  for (int64_t i = 0; i < rank; ++i)
    newShape.push_back(createDynamicDimensionType(graphOp, newParamDecls));
  TensorType outputTy =
      TensorType::get(graphOp->getContext(), newShape, dtype, deviceRef);
  return outputTy;
}

/// Given an optional set of output values, and an index, return the dtype
/// of the underlying tensor at the index.
///
/// Assumes we are dealing with TensorType.
///
/// If `values` is not present, return the fallbackDType.
static DType getDType(std::optional<ValueRange> values, size_t index,
                      DType fallbackDType) {
  if (!values.has_value())
    return fallbackDType;

  return cast<TensorType>(values.value()[index].getType()).getDtype();
}

/// Given two tensor types which are meant to represent the same shape
/// return a new type which returns a more "refined type" using information
/// from both existing types
TensorType RMO::refineShapeWithDType(TensorType ty1, TensorType ty2,
                                     DType dtype) {
  // TODO(AIPIPE-312): Re-enable once the graph API propagates devices.
  // ASSERT_STREAM(ty1.getDeviceRef() == ty2.getDeviceRef(),
  //               << " tried to refine types with different devices");
  // Refine each dimension individually
  ASSERT_STREAM(ty1.getRank() == ty2.getRank(),
                << " tried to refine types with different ranks");
  SmallVector<TypedAttr> newDims;
  auto refineDim = [&ty1, &ty2](size_t index) {
    if (ty1.isIntegerDim(index) && ty2.isIntegerDim(index))
      ASSERT_STREAM(
          ty1.getIntDim(index) == ty2.getIntDim(index),
          << " trying to refine two tensors whose integer dimensions don't "
             "match.");

    // Prefer integers over anything else
    if (ty1.isIntegerDim(index))
      return ty1.getDim(index);
    if (ty2.isIntegerDim(index))
      return ty2.getDim(index);

    // Otherwise both are parameters, just favor the first
    return ty1.getDim(index);
  };
  for (size_t i = 0, e = ty1.getRank(); i < e; ++i)
    newDims.push_back(refineDim(i));

  return TensorType::get(ty1.getContext(), newDims, dtype,
                         /*deviceRef=*/ty1.getDeviceRef(), ty1.getMetadata());
}

// Return a TensorType with refined shape information given ty1 and ty2 which
// represent the same underlying shape. Returned type will have a DType matching
// ty1.
TensorType RMO::refineShape(TensorType ty1, TensorType ty2) {
  return refineShapeWithDType(ty1, ty2, ty1.getDtype());
}

//===----------------------------------------------------------------------===//
// RMODialect MO analogue ops
//===----------------------------------------------------------------------===//

void MOShapeOfOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // We should match the dtype of the outputs if we have that information
  DType outDType = getDType(rmoResults, 0, DType::si64);
  TensorValue input = cast<TensorValue>(operands.getInput());

  int64_t rank = input.getType().getRank();
  TensorType outType = TensorType::get(
      context, SmallVector<TypedAttr>{createStaticDimensionType(graphOp, rank)},
      outDType, input.getType().getDeviceRef());

  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moResultTypes.push_back(outType);
  moOutputParamDecls.push_back({});
}

void MOBroadcastShapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  DType outDType =
      getDType(rmoResults, 0,
               cast<TensorType>(operands.getInputX().getType()).getDtype());
  DeviceRefAttr dev = getDevice(operands.getInputX());

  // Inputs are rank-1 shape tensors; their lengths represent the ranks of the
  // underlying data tensors. Rebind each to a fresh parametric-length rank-1
  // tensor (matching the MO op's natural signature). The broadcast result's
  // rank is `max(input lengths)` — when both inputs have static lengths we
  // propagate the concrete value so downstream consumers can determine their
  // output rank statically.
  SmallVector<KGEN::ParamDeclAttr> dimXParamDecls, dimYParamDecls;
  TensorType xType =
      getRankedParamTensorType(1, outDType, graphOp, dimXParamDecls, dev);
  TensorType yType =
      getRankedParamTensorType(1, outDType, graphOp, dimYParamDecls, dev);
  moOperandTypes.push_back(xType);
  moOperandTypes.push_back(yType);
  moInputParamDecls.push_back(std::move(dimXParamDecls));
  moInputParamDecls.push_back(std::move(dimYParamDecls));

  auto xLen = cast<TensorType>(operands.getInputX().getType()).getNumElements();
  auto yLen = cast<TensorType>(operands.getInputY().getType()).getNumElements();
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType outType;
  if (xLen.has_value() && yLen.has_value()) {
    outType = TensorType::get(
        context, SmallVector<int64_t>{std::max(*xLen, *yLen)}, outDType, dev);
  } else {
    outType =
        getRankedParamTensorType(1, outDType, graphOp, outputParamDecls, dev);
  }
  moResultTypes.push_back(outType);
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOReshapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  DeviceRefAttr dev = getDevice(operands.getInput());

  // If there already is an output value use the rank information below
  TensorValue outputValue;
  if (rmoResults.has_value())
    outputValue = cast<TensorValue>(rmoResults.value()[0]);

  // First input is not important, always the same
  DType outDType = cast<TensorType>(operands.getInput().getType()).getDtype();
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});

  // Second operand must always be rank 1, size equal to the output rank
  SmallVector<KGEN::ParamDeclAttr> newShapeParamDecls;
  DType newShapeDType =
      cast<TensorType>(operands.getNewShape().getType()).getDtype();

  if (outputValue && outputValue.getType().hasRank()) {
    int64_t outputRank = outputValue.getType().getRank();
    TensorType newShapeTy = TensorType::get(
        context, SmallVector<int64_t>{outputRank}, newShapeDType, dev);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back({});
  } else {
    TensorType newShapeTy = getRankedParamTensorType(1, newShapeDType, graphOp,
                                                     newShapeParamDecls, dev);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back(std::move(newShapeParamDecls));
  }

  // Get output rank from the shape operand's static numElements, or fall
  // back to the existing output value's rank when available.
  TensorType shapeType = cast<TensorType>(operands.getNewShape().getType());
  std::optional<int64_t> newRank = shapeType.getNumElements();
  if (!newRank.has_value() && outputValue)
    newRank = outputValue.getType().getRank();
  ASSERT_STREAM(newRank.has_value(),
                << "reshape output rank cannot be inferred");

  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(getRankedParamTensorType(*newRank, outDType, graphOp,
                                                   outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOBroadcastToOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // First input is not important, always the same
  DType outDType = cast<TensorType>(operands.getInput().getType()).getDtype();
  DeviceRefAttr dev = getDevice(operands.getInput());
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});

  // Second operand must always be rank-1
  SmallVector<KGEN::ParamDeclAttr> newShapeParamDecls;
  TensorValue bcastShape = cast<TensorValue>(operands.getNewShape());
  if (bcastShape.getType().hasStaticShape()) {
    moOperandTypes.push_back(bcastShape.getType());
    moInputParamDecls.push_back({});
  } else {
    TensorType newShapeTy = getRankedParamTensorType(
        1, bcastShape.getType().getDtype(), graphOp, newShapeParamDecls, dev);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back(std::move(newShapeParamDecls));
  }

  // Output rank comes from the shape operand's static numElements, or falls
  // back to the RMO result's rank when available.
  std::optional<int64_t> outputRank = bcastShape.getType().getNumElements();
  if (!outputRank.has_value() && rmoResults.has_value())
    outputRank = cast<TensorType>((*rmoResults)[0].getType()).getRank();
  ASSERT_STREAM(outputRank.has_value(),
                << "broadcast_to output rank cannot be inferred");

  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(getRankedParamTensorType(
      *outputRank, outDType, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOCastOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // If no dtype is given, recast to the current type.
  // This is useless but so is the default inferred type builder! Define a
  // useful inferred type builder below.
  TensorValue input = cast<TensorValue>(operands.getInput());
  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());

  // Operand remains the same
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  moResultTypes.push_back(input.getType().cloneWith(outputDType));
  moOutputParamDecls.push_back({});
}

void MOCastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
}

static TensorType inferReduceOpReturnType(
    TensorValue input, TensorValue axisValue, DType outputDtype,
    MO::GraphOp graphOp,
    SmallVectorImpl<KGEN::ParamDeclAttr> &outputParamDecls) {
  MLIRContext *ctx = input.getContext();
  TensorType inputTy = input.getType();
  DeviceRefAttr dev = inputTy.getDeviceRef();

  std::optional<int64_t> maybeAxis = tryGetConstantInt(axisValue);
  // Unknown axis => ranked dynamic output.
  if (!maybeAxis.has_value())
    return getRankedParamTensorType(inputTy.getRank(), outputDtype, graphOp,
                                    outputParamDecls, dev);

  int64_t axis = normalizeAxis(*maybeAxis, inputTy.getRank());
  SmallVector<TypedAttr> outputShape(inputTy.getTypedShapeAttr().getValues());
  if (axis >= 0 && axis < int64_t(inputTy.getRank()))
    outputShape[axis] = ShapeAttr::getDim(ctx, 1);
  return TensorType::get(
      MOSH::ShapeAttr::get(outputShape, MOSH::ShapeType::get(ctx)), outputDtype,
      dev);
}

static void inferMOTypesReduceImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue axis, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls,
    bool returnIndices) {

  DType outDType = getDType(
      rmoResults, 0, returnIndices ? DType::ui64 : input.getType().getDtype());

  // input type remains the same
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // Axis must be a scalar
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Infer output type
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType outputTy =
      inferReduceOpReturnType(input, axis, outDType, graphOp, outputParamDecls);
  moResultTypes.push_back(outputTy);
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

#define DEFINE_REDUCE_INFER_MO_TYPES(ClassName, ReturnIndices)                 \
  void ClassName::inferMOTypesImpl(                                            \
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,             \
      std::optional<ValueRange> rmoResults,                                    \
      SmallVectorImpl<Type> &moOperandTypes,                                   \
      SmallVectorImpl<Type> &moResultTypes,                                    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) { \
    inferMOTypesReduceImpl(                                                    \
        context, graphOp, cast<TensorValue>(operands.getInput()),              \
        cast<TensorValue>(operands.getAxis()), rmoResults, moOperandTypes,     \
        moResultTypes, moInputParamDecls, moOutputParamDecls, ReturnIndices);  \
  }

DEFINE_REDUCE_INFER_MO_TYPES(MOMeanOp, false)
DEFINE_REDUCE_INFER_MO_TYPES(MOReduceMulOp, false)
DEFINE_REDUCE_INFER_MO_TYPES(MOReduceAddOp, false)
DEFINE_REDUCE_INFER_MO_TYPES(MOReduceMinOp, false)
DEFINE_REDUCE_INFER_MO_TYPES(MOReduceMaxOp, false)
DEFINE_REDUCE_INFER_MO_TYPES(MOArgMinOp, true)
DEFINE_REDUCE_INFER_MO_TYPES(MOArgMaxOp, true)
#undef DEFINE_REDUCE_INFER_MO_TYPES

void RMO::inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices, TensorValue axis,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // Input type is unchanged
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // Updates and indices must have the same shape, but updates have the
  // same datatype as the input
  auto updatesTy = cast<MO::TensorType>(updates.getType());
  auto indicesTy = cast<MO::TensorType>(indices.getType());
  auto newUpdatesTy =
      refineShapeWithDType(updatesTy, indicesTy, updatesTy.getDtype());
  auto newIndicesTy =
      refineShapeWithDType(updatesTy, indicesTy, indicesTy.getDtype());

  moOperandTypes.push_back(newUpdatesTy);
  moOperandTypes.push_back(newIndicesTy);
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  // Axis must be a scalar
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // The output of scatter has the same type as the input
  moResultTypes.push_back(input.getType());
  moOutputParamDecls.push_back({});
}

void RMO::inferMOTypesElemwiseUnaryImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  auto inputTy = cast<MO::TensorType>(input.getType());

  // Expected input type
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // Expected output type
  moResultTypes.push_back(inputTy);
  moOutputParamDecls.push_back({});
}

void RMO::inferMOTypesUnaryExplicitOutputTypeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  auto inputTy = cast<MO::TensorType>(input.getType());
  ASSERT_STREAM(rmoResults.has_value(),
                << "inferMOTypesUnaryExplicitOutputTypeImpl must have explicit "
                   "output type");
  auto outputTy = cast<TensorType>((*rmoResults)[0].getType());

  // Expected input type
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // Expected output type
  moResultTypes.push_back(outputTy);
  moOutputParamDecls.push_back({});
}

static void inferConvMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filter, TensorValue strides, TensorValue dilations,
    TensorValue paddings, TensorValue outputPaddings, TensorValue numGroups,
    bool isTranspose, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  DeviceRefAttr dev = input.getType().getDeviceRef();

  // input / filter types remain the same.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filter.getType());
  moInputParamDecls.push_back({});

  // strides / dilations / paddings / output_paddings are rank-1 per the op's
  // typed constraints (MO_Rank1IndexTensor); their MO types match their RMO
  // types verbatim.
  moOperandTypes.push_back(strides.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(dilations.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(paddings.getType());
  moInputParamDecls.push_back({});

  if (isTranspose) {
    moOperandTypes.push_back(outputPaddings.getType());
    moInputParamDecls.push_back({});
  } else {
    // num_groups must be a scalar.
    moOperandTypes.push_back(
        numGroups.getType().cloneWith(SmallVector<TypedAttr>{}));
    moInputParamDecls.push_back({});
  }

  // Infer output type: output rank = input rank.
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType inputTy = input.getType();
  moResultTypes.push_back(getRankedParamTensorType(
      inputTy.getRank(), outputDtype, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOConvOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  inferConvMOTypesImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                       cast<TensorValue>(operands.getFilter()),
                       cast<TensorValue>(operands.getStrides()),
                       cast<TensorValue>(operands.getDilations()),
                       cast<TensorValue>(operands.getPaddings()), nullptr,
                       cast<TensorValue>(operands.getNumGroups()), false,
                       rmoResults, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);
}

void MOConvTransposeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  inferConvMOTypesImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                       cast<TensorValue>(operands.getFilter()),
                       cast<TensorValue>(operands.getStrides()),
                       cast<TensorValue>(operands.getDilations()),
                       cast<TensorValue>(operands.getPaddings()),
                       cast<TensorValue>(operands.getOutputPaddings()), nullptr,
                       true, rmoResults, moOperandTypes, moResultTypes,
                       moInputParamDecls, moOutputParamDecls);
}

static void inferMOTypesPoolImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filterShape, TensorValue strides, TensorValue dilations,
    TensorValue paddings, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // input type remains the same.
  DeviceRefAttr dev = input.getType().getDeviceRef();
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // filter_shape / strides / dilations / paddings are rank-1 per the op's
  // typed constraints (MO_Rank1IndexTensor); their MO types match their RMO
  // types verbatim.
  moOperandTypes.push_back(filterShape.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(strides.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(dilations.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(paddings.getType());
  moInputParamDecls.push_back({});

  // Infer output type: output rank = input rank.
  // Output is always 4D (for now), but MO verifier requires input-rank ==
  // output-rank.
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType inputTy = input.getType();
  moResultTypes.push_back(getRankedParamTensorType(
      inputTy.getRank(), outputDtype, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

#define DEFINE_POOL_INFER_MO_TYPES(ClassName)                                  \
  void ClassName::inferMOTypesImpl(                                            \
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,             \
      std::optional<ValueRange> rmoResults,                                    \
      SmallVectorImpl<Type> &moOperandTypes,                                   \
      SmallVectorImpl<Type> &moResultTypes,                                    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) { \
    inferMOTypesPoolImpl(                                                      \
        context, graphOp, cast<TensorValue>(operands.getInput()),              \
        cast<TensorValue>(operands.getFilterShape()),                          \
        cast<TensorValue>(operands.getStrides()),                              \
        cast<TensorValue>(operands.getDilations()),                            \
        cast<TensorValue>(operands.getPaddings()), rmoResults, moOperandTypes, \
        moResultTypes, moInputParamDecls, moOutputParamDecls);                 \
  }

DEFINE_POOL_INFER_MO_TYPES(MOAvgPoolOp)
DEFINE_POOL_INFER_MO_TYPES(MOAvgPoolCeilOp)
DEFINE_POOL_INFER_MO_TYPES(MOMaxPoolOp)
DEFINE_POOL_INFER_MO_TYPES(MOMaxPoolCeilOp)

#undef DEFINE_POOL_INFER_MO_TYPES

void MORangeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  TensorValue start = cast<TensorValue>(operands.getStart());
  TensorValue step = cast<TensorValue>(operands.getStep());
  TensorValue limit = cast<TensorValue>(operands.getLimit());

  auto scalarTensorType =
      TensorType::get(SmallVector<TypedAttr>{},
                      start.getType().getElementType(), getHostDevice(context));

  moOperandTypes.push_back(scalarTensorType);
  moOperandTypes.push_back(scalarTensorType);
  moOperandTypes.push_back(scalarTensorType);

  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  DeviceRefAttr outDevice;
  if (rmoResults.has_value())
    outDevice = getDevice((*rmoResults)[0]);
  else
    outDevice = getHostDevice(context);
  auto result = [&]() -> std::optional<TensorType> {
    auto startInt = tryGetConstantInt(start);
    auto stepInt = tryGetConstantInt(step);
    auto limitInt = tryGetConstantInt(limit);

    if (!startInt || !stepInt || !limitInt)
      return std::nullopt;

    auto maybeNumElements = getRangeNumElements(*startInt, *limitInt, *stepInt);

    if (maybeNumElements.isError())
      return std::nullopt;

    SmallVector<int64_t> arr({maybeNumElements.get()});

    return TensorType::get(arr, start.getType().getElementType(), outDevice);
  }();

  if (result) {
    moResultTypes.push_back(*result);
    moOutputParamDecls.push_back({});
  } else {
    SmallVector<KGEN::ParamDeclAttr> newDecls;

    TensorType outType = getRankedParamTensorType(1, start.getType().getDtype(),
                                                  graphOp, newDecls, outDevice);

    moOutputParamDecls.push_back(std::move(newDecls));
    moResultTypes.push_back(outType);
  }
}

static void inferMOTypesResizeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue size, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` type stays the same.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  DeviceRefAttr dev = input.getType().getDeviceRef();

  // `size` must be 1-D.
  TensorType sizeTy = size.getType();
  if (!sizeTy.hasRank() || sizeTy.getRank() != 1) {
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    moOperandTypes.push_back(getRankedParamTensorType(
        1, sizeTy.getDtype(), graphOp, paramDecls, dev));
    moInputParamDecls.push_back(std::move(paramDecls));
  } else {
    moOperandTypes.push_back(sizeTy);
    moInputParamDecls.push_back({});
  }

  // Infer output type: output rank = input rank.
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType inputTy = input.getType();
  moResultTypes.push_back(getRankedParamTensorType(
      inputTy.getRank(), outputDtype, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOResizeLinearOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesResizeImpl(
      context, graphOp, cast<TensorValue>(operands.getInput()),
      cast<TensorValue>(operands.getSize()), rmoResults, moOperandTypes,
      moResultTypes, moInputParamDecls, moOutputParamDecls);
}

void MOResizeNearestOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesResizeImpl(
      context, graphOp, cast<TensorValue>(operands.getInput()),
      cast<TensorValue>(operands.getSize()), rmoResults, moOperandTypes,
      moResultTypes, moInputParamDecls, moOutputParamDecls);
}

void MOResizeBicubicOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesResizeImpl(
      context, graphOp, cast<TensorValue>(operands.getInput()),
      cast<TensorValue>(operands.getSize()), rmoResults, moOperandTypes,
      moResultTypes, moInputParamDecls, moOutputParamDecls);
}

void MOSelectOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  MO::TensorType condType = cast<MO::TensorType>(operands.getCond().getType());
  MO::TensorType xType = cast<MO::TensorType>(operands.getX().getType());
  MO::TensorType yType = cast<MO::TensorType>(operands.getY().getType());

  // Refine shape with all 3 operands.
  MO::TensorType refinedShape = refineShapeWithDType(
      refineShape(xType, yType), condType, xType.getDtype());
  // Condition is boolean.
  moOperandTypes.push_back(refinedShape.cloneWith(DType::kBool));
  moOperandTypes.push_back(refinedShape);
  moOperandTypes.push_back(refinedShape);
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  moResultTypes.push_back(refinedShape);
  moOutputParamDecls.push_back({});
}

void MOGatherOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue indices = cast<TensorValue>(operands.getIndices());
  TensorValue axis = cast<TensorValue>(operands.getAxis());
  DeviceRefAttr dev = getDevice(operands.getInput());

  // `input` type stays the same.
  TensorType inputTy = input.getType();
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // `indices` type stays the same.
  TensorType indicesTy = indices.getType();
  moOperandTypes.push_back(indicesTy);
  moInputParamDecls.push_back({});

  // `axis` must be a scalar.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Infer output type: output rank = inputRank - 1 + indicesRank.
  DType outputDtype = getDType(rmoResults, 0, inputTy.getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(inputTy.getRank() - 1 + indicesTy.getRank(),
                               outputDtype, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOGatherNdOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue indices = cast<TensorValue>(operands.getIndices());
  DeviceRefAttr dev = getDevice(operands.getInput());

  // `input` type stays the same.
  TensorType inputTy = input.getType();
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // `indices` type stays the same.
  TensorType indicesTy = indices.getType();
  moOperandTypes.push_back(indicesTy);
  moInputParamDecls.push_back({});

  // Infer output type.
  DType outputDtype = getDType(rmoResults, 0, inputTy.getDtype());
  IntegerAttr kAttr;
  if (indicesTy.getRank() > 0)
    kAttr = dyn_cast<IntegerAttr>(indicesTy.getDims().back());

  // output rank = batch_dims + (indices.rank - batch_dims - 1)
  //                          + (input.rank - batch_dims - K).
  // K must be statically known via the last dim of `indices`. If not, fall
  // back to the RMO result's rank.
  std::optional<size_t> outputRank;
  if (kAttr) {
    size_t k = kAttr.getSInt();
    size_t batchDims = operands.getBatchDims().getSExtValue();
    outputRank =
        (indicesTy.getRank() - 1) + (inputTy.getRank() - batchDims - k);
  } else if (rmoResults.has_value()) {
    outputRank = cast<TensorType>((*rmoResults)[0].getType()).getRank();
  }
  ASSERT_STREAM(outputRank.has_value(),
                << "gather_nd output rank cannot be inferred");

  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(getRankedParamTensorType(
      *outputRank, outputDtype, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOTileOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue repeats = cast<TensorValue>(operands.getRepeats());
  DeviceRefAttr dev = getDevice(operands.getInput());

  int64_t rank = input.getType().getRank();

  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      rank, input.getType().getDtype(), graphOp, inputParamDecls, dev));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `repeats` has static shape [rank].
  TypedAttr dimAttr = ShapeAttr::getDim(context, rank);
  moOperandTypes.push_back(
      repeats.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  moInputParamDecls.push_back({});

  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamDecls;
  moResultTypes.push_back(getRankedParamTensorType(rank, outputDtype, graphOp,
                                                   resultParamDecls, dev));
  moOutputParamDecls.push_back(std::move(resultParamDecls));
}

static void inferMOTypesSqueezeShapeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue inputShape,
    TensorValue indices, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Everything is rank1, use rebinds+checks, shape inference do the right thing
  DeviceRefAttr dev = inputShape.getType().getDeviceRef();
  SmallVector<KGEN::ParamDeclAttr> dimXParamDecls, dimYParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      1, inputShape.getType().getDtype(), graphOp, dimXParamDecls, dev));
  moOperandTypes.push_back(getRankedParamTensorType(
      1, indices.getType().getDtype(), graphOp, dimYParamDecls, dev));
  moInputParamDecls.push_back(std::move(dimXParamDecls));
  moInputParamDecls.push_back(std::move(dimYParamDecls));

  DType outDType = getDType(rmoResults, 0, inputShape.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(1, outDType, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOSqueezeShapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesSqueezeShapeImpl(
      context, graphOp, cast<TensorValue>(operands.getInputShape()),
      cast<TensorValue>(operands.getRemoveIndices()), rmoResults,
      moOperandTypes, moResultTypes, moInputParamDecls, moOutputParamDecls);
}

void MOTransposeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue perm = cast<TensorValue>(operands.getPerm());
  DeviceRefAttr dev = getDevice(operands.getInput());

  int64_t rank = input.getType().getRank();

  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      rank, input.getType().getDtype(), graphOp, inputParamDecls, dev));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `perm` has static shape [rank].
  TypedAttr dimAttr = ShapeAttr::getDim(context, rank);
  moOperandTypes.push_back(
      perm.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  moInputParamDecls.push_back({});

  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamDecls;
  moResultTypes.push_back(getRankedParamTensorType(rank, outputDType, graphOp,
                                                   resultParamDecls, dev));
  moOutputParamDecls.push_back(std::move(resultParamDecls));
}

void RMO::inferMOTypesTopKImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input, TensorValue k,
    TensorValue axis, TensorValue sorted, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  DeviceRefAttr dev = input.getType().getDeviceRef();

  int64_t rank = input.getType().getRank();

  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      rank, input.getType().getDtype(), graphOp, inputParamDecls, dev));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // k / axis / sorted are scalars.
  moOperandTypes.push_back(k.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(sorted.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Both outputs have identical shape, with same rank than input.
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  TensorType outputTy = getRankedParamTensorType(rank, outputDtype, graphOp,
                                                 outputParamDecls, dev);
  moResultTypes.push_back(outputTy);
  moOutputParamDecls.push_back(std::move(outputParamDecls));
  moResultTypes.push_back(outputTy.cloneWith(DType::si64));
  moOutputParamDecls.push_back({});
}

void LinAlg::MOBandPartOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue numLower = cast<TensorValue>(operands.getNumLower());
  TensorValue numUpper = cast<TensorValue>(operands.getNumUpper());
  TensorValue exclude = cast<TensorValue>(operands.getExclude());

  // Input type is unchanged.
  TensorType inputTy = input.getType();
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // num_lower / num_upper / exclude are scalars.
  moOperandTypes.push_back(numLower.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(numUpper.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(exclude.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Output type is identical to input type
  moResultTypes.push_back(inputTy);
  moOutputParamDecls.push_back({});
}

static void inferMOTypesPadImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue paddings, TensorValue constant,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  DeviceRefAttr dev = input.getType().getDeviceRef();

  TensorType inputTy = input.getType();
  TensorType paddingsTy = paddings.getType();
  int64_t rank = inputTy.getRank();

  // Set input type.
  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(getRankedParamTensorType(rank, inputTy.getDtype(),
                                                    graphOp, inputDecls, dev));
  moInputParamDecls.push_back(std::move(inputDecls));

  // paddings shape must be [2 * rank].
  moOperandTypes.push_back(paddingsTy.cloneWith(
      SmallVector<TypedAttr>{createStaticDimensionType(graphOp, rank * 2)}));
  moInputParamDecls.push_back({});

  // constant must be scalar.
  if (constant) {
    moOperandTypes.push_back(
        constant.getType().cloneWith(ArrayRef<TypedAttr>{}));
    moInputParamDecls.push_back({});
  }

  // set output type.
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(getRankedParamTensorType(rank, inputTy.getDtype(),
                                                   graphOp, outputDecls, dev));
  moOutputParamDecls.push_back(std::move(outputDecls));
}

void MOPadConstantOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesPadImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                      cast<TensorValue>(operands.getPaddings()),
                      cast<TensorValue>(operands.getConstant()), rmoResults,
                      moOperandTypes, moResultTypes, moInputParamDecls,
                      moOutputParamDecls);
}

void MOPadReflectOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesPadImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                      cast<TensorValue>(operands.getPaddings()), nullptr,
                      rmoResults, moOperandTypes, moResultTypes,
                      moInputParamDecls, moOutputParamDecls);
}

void MOPadRepeatOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesPadImpl(context, graphOp, cast<TensorValue>(operands.getInput()),
                      cast<TensorValue>(operands.getPaddings()), nullptr,
                      rmoResults, moOperandTypes, moResultTypes,
                      moInputParamDecls, moOutputParamDecls);
}

void MOSliceOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue start = cast<TensorValue>(operands.getStart());
  TensorValue stop = cast<TensorValue>(operands.getStop());
  TensorValue step = cast<TensorValue>(operands.getStep());
  DeviceRefAttr dev = getDevice(operands.getInput());

  TensorType inputTy = input.getType();
  int64_t rank = inputTy.getRank();

  // Set input type.
  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(getRankedParamTensorType(rank, inputTy.getDtype(),
                                                    graphOp, inputDecls, dev));
  moInputParamDecls.push_back(std::move(inputDecls));

  // start / stop / step have shape `[rank]`.
  // Only assign param declarations to the `start` operand.
  SmallVector<TypedAttr> sliceDims{createStaticDimensionType(graphOp, rank)};
  moOperandTypes.push_back(start.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(stop.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(step.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});

  // set output type.
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(getRankedParamTensorType(rank, inputTy.getDtype(),
                                                   graphOp, outputDecls, dev));
  moOutputParamDecls.push_back(std::move(outputDecls));
}

void MORoiAlignOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue rois = cast<TensorValue>(operands.getRois());
  TensorValue outputHeight = cast<TensorValue>(operands.getOutputHeight());
  TensorValue outputWidth = cast<TensorValue>(operands.getOutputWidth());
  TensorValue spatialScale = cast<TensorValue>(operands.getSpatialScale());
  TensorValue samplingRatio = cast<TensorValue>(operands.getSamplingRatio());
  DeviceRefAttr dev = getDevice(operands.getInput());

  // `input` must be 4D.
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      4, input.getType().getDtype(), graphOp, inputParamDecls, dev));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `rois` must be 2D.
  SmallVector<KGEN::ParamDeclAttr> roisParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      2, rois.getType().getDtype(), graphOp, roisParamDecls, dev));
  moInputParamDecls.push_back(std::move(roisParamDecls));

  // `output_height` / `output_width` / `spatial_case` / `sampling_ratio` must
  // be scalars.
  moOperandTypes.push_back(
      outputHeight.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      outputWidth.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      spatialScale.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      samplingRatio.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // `result` must be 4D.
  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(4, outputDType, graphOp, resultParamDecls, dev));
  moOutputParamDecls.push_back(std::move(resultParamDecls));
}

void MORandomNormalOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue shape = cast<TensorValue>(operands.getShape());
  TensorValue mean = cast<TensorValue>(operands.getMean());
  TensorValue variance = cast<TensorValue>(operands.getVariance());
  TensorValue seed = cast<TensorValue>(operands.getSeed());
  DeviceRefAttr dev;
  if (rmoResults.has_value())
    dev = getDevice((*rmoResults)[0]);
  else
    dev = getDevice(operands.getMean());

  // Try to infer the output rank
  TensorType shapeTy = shape.getType();
  std::optional<int64_t> rank;
  if (shapeTy.hasStaticShape() && shapeTy.getRank() > 0)
    rank = shapeTy.getIntDim(0);
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // `shape` must have shape [rank]
  SmallVector<KGEN::ParamDeclAttr> shapeParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? shapeTy.cloneWith(SmallVector<TypedAttr>{
                createStaticDimensionType(graphOp, *rank)})
          : getRankedParamTensorType(1, shapeTy.getDtype(), graphOp,
                                     shapeParamDecls, dev));
  moInputParamDecls.push_back(std::move(shapeParamDecls));

  // `mean` / `variance` are scalars.
  moOperandTypes.push_back(mean.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(variance.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // `seed` is rank-1 `[1]`.
  moOperandTypes.push_back(seed.getType().cloneWith(
      SmallVector<TypedAttr>{createStaticDimensionType(graphOp, 1)}));
  moInputParamDecls.push_back({});

  // Output rank.
  ASSERT_STREAM(rank.has_value(),
                << "random_normal output rank cannot be inferred");
  DType outputDType = getDType(rmoResults, 0, mean.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(*rank, outputDType, graphOp, outputDecls, dev));
  moOutputParamDecls.push_back(std::move(outputDecls));
}

void MORandomUniformOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue shape = cast<TensorValue>(operands.getShape());
  TensorValue lowerBound = cast<TensorValue>(operands.getLowerBound());
  TensorValue upperBound = cast<TensorValue>(operands.getUpperBound());
  TensorValue seed = cast<TensorValue>(operands.getSeed());
  DeviceRefAttr dev = getDevice(operands.getLowerBound());

  // Try to infer the output rank
  TensorType shapeTy = shape.getType();
  std::optional<int64_t> rank;
  if (shapeTy.hasStaticShape() && shapeTy.getRank() > 0)
    rank = shapeTy.getIntDim(0);
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // `shape` must have shape [rank]
  SmallVector<KGEN::ParamDeclAttr> shapeParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? shapeTy.cloneWith(SmallVector<TypedAttr>{
                createStaticDimensionType(graphOp, *rank)})
          : getRankedParamTensorType(1, shapeTy.getDtype(), graphOp,
                                     shapeParamDecls, dev));
  moInputParamDecls.push_back(std::move(shapeParamDecls));

  // `lowerBound` / `upperBound` are scalars.
  moOperandTypes.push_back(
      lowerBound.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      upperBound.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // `seed` is rank-1 `[1]`.
  moOperandTypes.push_back(seed.getType().cloneWith(
      SmallVector<TypedAttr>{createStaticDimensionType(graphOp, 1)}));
  moInputParamDecls.push_back({});

  // Output rank.
  ASSERT_STREAM(rank.has_value(),
                << "random_uniform output rank cannot be inferred");
  DType outputDType = getDType(rmoResults, 0, lowerBound.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(*rank, outputDType, graphOp, outputDecls, dev));
  moOutputParamDecls.push_back(std::move(outputDecls));
}

void RMO::inferMOTypesScatterNdImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` / `output` types are identical.
  TensorType inOutTy = input.getType();
  DeviceRefAttr dev = input.getType().getDeviceRef();
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back({});
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});

  // updates.shape = indices.shape[:-1] + tensor.shape[indices.shape[-1]:].
  // updates rank requires indices' last dim to be a static integer; fall back
  // to the updates operand's own rank when it isn't.
  TensorType indicesTy = indices.getType();
  SmallVector<KGEN::ParamDeclAttr> updatesParamDecls;
  std::optional<int64_t> updatesRank;
  if (indicesTy.getRank() > 0 && isa<IntegerAttr>(indicesTy.getDims().back())) {
    int64_t indicesRank = indicesTy.getRank();
    int64_t indicesDepth = indicesTy.getIntDim(indicesRank - 1);
    updatesRank = (indicesRank - 1) + (inOutTy.getRank() - indicesDepth);
  } else {
    updatesRank = updates.getType().getRank();
  }
  moOperandTypes.push_back(
      getRankedParamTensorType(*updatesRank, updates.getType().getDtype(),
                               graphOp, updatesParamDecls, dev));
  moInputParamDecls.push_back(std::move(updatesParamDecls));

  // `indices` type is unchanged.
  moOperandTypes.push_back(indicesTy);
  moInputParamDecls.push_back({});
}

void MONonMaximumSuppressionOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue boxes = cast<TensorValue>(operands.getBoxes());
  TensorValue scores = cast<TensorValue>(operands.getScores());
  TensorValue maxOutputBoxesPerClass =
      cast<TensorValue>(operands.getMaxOutputBoxesPerClass());
  TensorValue iouThreshold = cast<TensorValue>(operands.getIouThreshold());
  TensorValue scoreThreshold = cast<TensorValue>(operands.getScoreThreshold());
  DeviceRefAttr dev = getDevice(operands.getBoxes());

  // `boxes` and `scores` must be rank 3.
  SmallVector<KGEN::ParamDeclAttr> boxesParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      3, boxes.getType().getDtype(), graphOp, boxesParamDecls, dev));
  moInputParamDecls.push_back(std::move(boxesParamDecls));

  SmallVector<KGEN::ParamDeclAttr> scoresParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      3, scores.getType().getDtype(), graphOp, scoresParamDecls, dev));
  moInputParamDecls.push_back(std::move(scoresParamDecls));

  // `maxOutputBoxesPerClass` / `iouThreshold` / `scoreThreshold` must be rank
  // 0.
  moOperandTypes.push_back(
      maxOutputBoxesPerClass.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      iouThreshold.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      scoreThreshold.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // `output` must be rank 2.
  DType outputDType = getDType(rmoResults, 0, DType::si64);
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(2, outputDType, graphOp, outputParamDecls, dev));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOArgNonzeroOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  DeviceRefAttr dev = getDevice(operands.getInput());

  // input[rank] => output [count, rank]. The input rank is known statically;
  // the count of non-zero elements is data-dependent and must be supplied by
  // the caller (Graph API contract: ops.nonzero requires an explicit
  // `out_dim`). Pull the count dim from the existing result type rather than
  // minting a fresh parametric dim here. The result is rank-2 si64 per the
  // op's typed constraint (MO_Rank2I64Tensor).
  int64_t rank = input.getType().getRank();
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  ASSERT_STREAM(rmoResults.has_value(),
                << "rmo.mo.arg_nonzero must be constructed with an explicit "
                   "result type providing the count dimension");
  TensorType outputTy = cast<TensorType>((*rmoResults)[0].getType());
  TypedAttr countAttr = outputTy.getDim(0);
  TypedAttr rankAttr = createStaticDimensionType(graphOp, rank);

  moResultTypes.push_back(TensorType::get(
      graphOp->getContext(), SmallVector<TypedAttr>{countAttr, rankAttr},
      DType::si64, dev));
  moOutputParamDecls.push_back({});
}

void MOCumSumOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue axis = cast<TensorValue>(operands.getAxis());

  // `input` / `output` types are identical.
  TensorType inOutTy = input.getType();
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back({});
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});

  // `axis` type is 0D.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
}

void MOMutableLoadOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  auto inBufferTy = operands.getInBuffer().getType();

  auto chainTy = operands.getInChain().getType();

  moInputParamDecls.push_back({});
  moOperandTypes.push_back(inBufferTy);
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(chainTy);

  moOutputParamDecls.push_back({});
  moResultTypes.push_back(MO::castFromTensorLike(inBufferTy));
  moOutputParamDecls.push_back({});
  moResultTypes.push_back(chainTy);
}

void MOMutableStoreOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  auto inBufferTy = operands.getInBuffer().getType();
  auto chainTy = operands.getInChain().getType();

  moInputParamDecls.push_back({});
  moOperandTypes.push_back(inBufferTy);
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(MO::castFromTensorLike(inBufferTy));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(chainTy);

  moOutputParamDecls.push_back({});
  moResultTypes.push_back(chainTy);
}

void MOMutableStoreSliceOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  auto inBufferTy = operands.getInBuffer().getType();
  auto chainTy = operands.getInChain().getType();

  TensorType inTensorTy = MO::castFromTensorLike(inBufferTy);
  TensorType sliceTy = cast<TensorType>(operands.getSlice().getType());
  TensorValue start = cast<TensorValue>(operands.getStart());
  TensorValue stop = cast<TensorValue>(operands.getStop());
  TensorValue step = cast<TensorValue>(operands.getStep());
  DeviceRefAttr dev = getDevice(operands.getInBuffer());

  // As opposed to the standard mo.slice inferMOTypes which does a bunch of
  // shenanigans around trying to infer a rank to use for all inputs the we can
  // just directly use the rank of the buffer input as BufferTypes are
  // guaranteed to have ranked inputs (at time of writing).
  auto rank = inTensorTy.getRank();

  moOperandTypes.push_back(inBufferTy);
  moInputParamDecls.push_back({});

  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(getRankedParamTensorType(rank, sliceTy.getDtype(),
                                                    graphOp, inputDecls, dev));
  moInputParamDecls.push_back(std::move(inputDecls));

  // start / stop / step have shape `[rank]`.
  SmallVector<KGEN::ParamDeclAttr> startParamDecls;
  // Only assign param declarations to the `start` operand.
  SmallVector<TypedAttr> sliceDims{createStaticDimensionType(graphOp, rank)};

  moOperandTypes.push_back(start.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back(std::move(startParamDecls));
  moOperandTypes.push_back(stop.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(step.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});

  moOperandTypes.push_back(chainTy);
  moInputParamDecls.push_back({});

  moResultTypes.push_back(chainTy);
  moOutputParamDecls.push_back({});
}

void MOMutableLoadOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInBufferMutable());
}

void MOMutableStoreOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

void MOMutableStoreSliceOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

//===----------------------------------------------------------------------===//
// MOTransferOp
//===----------------------------------------------------------------------===//

void MOTransferOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // Input tensor type (first operand for mo.transfer).
  auto inputType = cast<TensorType>(operands.getInput().getType());
  moOperandTypes.push_back(inputType);
  moInputParamDecls.push_back({});

  // Input chain type (second operand for mo.transfer).
  Type chainType = operands.getInChain().getType();
  moOperandTypes.push_back(chainType);
  moInputParamDecls.push_back({});

  // Result tensor type - same as input but potentially different device.
  TensorType resultType = inputType;
  if (rmoResults && !rmoResults->empty())
    resultType = cast<TensorValue>((*rmoResults)[0]).getType();

  moResultTypes.push_back(resultType);
  moOutputParamDecls.push_back({});

  // Output chain type.
  moResultTypes.push_back(chainType);
  moOutputParamDecls.push_back({});
}
