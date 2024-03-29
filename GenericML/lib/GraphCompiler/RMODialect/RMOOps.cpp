//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/Dims.h"
#include "Support/ML/RangeUtils.h"
#include <optional>

using namespace M;
using namespace RMO;

using MO::TensorType;
using MO::TensorValue;

using M::MO::parseParamDecls;
using M::MO::printParamDecls;

//===----------------------------------------------------------------------===//
// RMODialect op support
//===----------------------------------------------------------------------===//

namespace {

/// Create a new attribute representing a static integer dimension.
TypedAttr createStaticDimensionType(MO::GraphOp op, int64_t dimSize) {
  return IntegerAttr::get(IndexType::get(op.getContext()), dimSize);
}

/// Create a new parameter representing a parameterized dimension.
TypedAttr
createDynamicDimensionType(MO::GraphOp graphOp,
                           SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls,
                           const std::string &dimPrefix = "RMO_D") {
  KGEN::ParamDeclAttr paramDecl = graphOp.getNewDimParam(dimPrefix);
  newParamDecls.emplace_back(paramDecl);
  return KGEN::ParamDeclRefAttr::get(paramDecl);
}

/// Create a new parameter representing a dynamically ranked shape.
KGEN::ParamDeclRefAttr createDynamicRankedShapeType(
    MO::GraphOp graphOp, SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls) {
  KGEN::ParamDeclAttr paramDecl = graphOp.getNewShapeParam("RMO_Sh");
  newParamDecls.emplace_back(paramDecl);
  return KGEN::ParamDeclRefAttr::get(paramDecl);
}

TensorType
getUnrankedTensorType(DType dtype, MO::GraphOp graphOp,
                      SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls) {
  TypedAttr dim = createDynamicRankedShapeType(graphOp, newParamDecls);
  TensorType outputTy = TensorType::get(dim, dtype);
  return outputTy;
}

TensorType
getRankedParamTensorType(int64_t rank, DType dtype, MO::GraphOp graphOp,
                         SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls) {
  SmallVector<TypedAttr> newShape;
  for (int64_t i = 0; i < rank; i++) {
    newShape.push_back(createDynamicDimensionType(graphOp, newParamDecls));
  }
  TensorType outputTy = TensorType::get(graphOp->getContext(), newShape, dtype);
  return outputTy;
}

/// Given an optional set of output values, and an index, return the dtype
/// of the underlying tensor at the index.
///
/// Assumes we are dealing with TensorType.
///
/// If `values` is not present, return the fallbackDType.
DType getDType(std::optional<ValueRange> values, size_t index,
               DType fallbackDType) {
  if (!values.has_value())
    return fallbackDType;

  return cast<TensorType>(values.value()[index].getType()).getDtype();
}

} // namespace

namespace M::RMO {

/// Given two tensor types which are meant to represent the same shape
/// return a new type which returns a more "refined type" using information
/// from both existing types
TensorType refineShapeWithDType(TensorType ty1, TensorType ty2, DType dtype) {
  GENERICML_ASSERT(!ty1.hasUnknownShape() && !ty2.hasUnknownShape())
      << " no question marks should be here";
  bool ty1Ranked = ty1.hasRank();
  bool ty2Ranked = ty2.hasRank();

  // Prefer ranked to unranked tensors
  if (!ty1Ranked && !ty2Ranked)
    return ty1.cloneWith(dtype);
  if (!ty1Ranked)
    return ty2.cloneWith(dtype);
  if (!ty2Ranked)
    return ty1.cloneWith(dtype);

  // Refine each dimension individually
  GENERICML_ASSERT(ty1.getRank() == ty2.getRank())
      << " tried to refine types with different ranks";
  SmallVector<TypedAttr> newDims;
  auto refineDim = [&ty1, &ty2](size_t index) {
    if (ty1.isIntegerDim(index) && ty2.isIntegerDim(index))
      GENERICML_ASSERT(ty1.getIntDim(index) == ty2.getIntDim(index))
          << " trying to refine two tensors whose integer dimensions don't "
             "match.";

    // Prefer integers over anything else
    if (ty1.isIntegerDim(index))
      return ty1.getDim(index);
    if (ty2.isIntegerDim(index))
      return ty2.getDim(index);

    // Otherwise both are parameters, just favor the first
    return ty1.getDim(index);
  };
  for (size_t i = 0, e = ty1.getRank(); i < e; i++)
    newDims.push_back(refineDim(i));

  return TensorType::get(ty1.getContext(), newDims, dtype, ty1.getMetadata());
}

TensorType refineShape(TensorType ty1, TensorType ty2) {
  GENERICML_ASSERT(ty1.getDtype() == ty2.getDtype())
      << " tried to refine types with different dtypes";
  return refineShapeWithDType(ty1, ty2, ty1.getDtype());
}

SmallVector<Value> rebindUnknownShapeInValues(ValueRange inputValues,
                                              MO::GraphOp graphOp,
                                              OpBuilder &builder) {
  SmallVector<Value> valuesNoUnknowns = inputValues;

  // Replace values with those rebound expressions
  for (int64_t i = 0, bound_i = inputValues.size(); i < bound_i; i++) {
    auto curTensor = dyn_cast<TensorValue>(inputValues[i]);

    // not a tensor, no need to modify as no `?`
    if (!curTensor)
      continue;

    if (!curTensor.getType().hasUnknownShape())
      continue;

    Location loc = curTensor.getLoc();
    SmallVector<KGEN::ParamDeclAttr> newParamDecls;

    if (!curTensor.getType().hasRank()) {
      TypedAttr newShape = createDynamicRankedShapeType(graphOp, newParamDecls);
      TensorType outTy = curTensor.getType().cloneWith(newShape);
      curTensor = builder.create<RMO::RebindTensorShapeOp>(
          loc, outTy, curTensor, newParamDecls);
    } else {
      SmallVector<TypedAttr> dims(curTensor.getType().getDims());
      for (int64_t j = 0, bound_j = dims.size(); j < bound_j; j++) {
        TypedAttr curDim = dims[j];
        if (TensorType::isUnknownDimParam(curDim)) {
          curDim = createDynamicDimensionType(graphOp, newParamDecls);
          dims[j] = curDim;
        }
      }
      TensorType outTy = curTensor.getType().cloneWith(dims);
      curTensor = builder.create<RMO::RebindTensorShapeOp>(
          loc, outTy, curTensor, newParamDecls);
    }

    valuesNoUnknowns[i] = curTensor;
  }
  return valuesNoUnknowns;
}

OpFoldResult RebindTensorShapeOp::fold(FoldAdaptor adaptor) {
  // Note `?` != `?` in terms of shapes
  if (getInput().getType().hasUnknownShape())
    return {};

  if (getInput().getType() == getResult().getType())
    return getInput();

  return {};
}

void RebindTensorShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TensorValue input, ShapeAttr newShape,
                                ArrayRef<ParamDeclAttr> outputParamDecls) {
  return build(odsBuilder, odsState, input.getType().cloneWith(newShape), input,
               outputParamDecls);
}

void MOShapeOfOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // We should match the dtype of the outputs if we have that information
  DType outDType = getDType(rmoResults, 0, DType::si64);
  TensorValue input = cast<TensorValue>(operands.getInput());

  // Ranked case:
  if (input.getType().hasRank()) {
    int64_t rank = input.getType().getRank();
    TensorType outType = TensorType::get(
        context,
        SmallVector<TypedAttr>{createStaticDimensionType(graphOp, rank)},
        outDType);

    moOperandTypes.push_back(input.getType());
    moInputParamDecls.push_back({});
    moResultTypes.push_back(outType);
    moOutputParamDecls.push_back({});
    return;
  }

  SmallVector<KGEN::ParamDeclAttr> newDecls;

  // Unranked case:
  TensorType outType = getRankedParamTensorType(1, outDType, graphOp, newDecls);
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  moResultTypes.push_back(outType);
  moOutputParamDecls.push_back(std::move(newDecls));
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

  // Everything is rank1, use rebinds+checks, shape inference do the right thing
  SmallVector<KGEN::ParamDeclAttr> dimXParamDecls, dimYParamDecls;
  TensorType xType =
      getRankedParamTensorType(1, outDType, graphOp, dimXParamDecls);
  TensorType yType =
      getRankedParamTensorType(1, outDType, graphOp, dimYParamDecls);
  moOperandTypes.push_back(xType);
  moOperandTypes.push_back(yType);
  moInputParamDecls.push_back(std::move(dimXParamDecls));
  moInputParamDecls.push_back(std::move(dimYParamDecls));

  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType outType =
      getRankedParamTensorType(1, outDType, graphOp, outputParamDecls);
  moResultTypes.push_back(outType);
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOReshapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // First input is not important, always the same
  DType outDType = cast<TensorType>(operands.getInput().getType()).getDtype();
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});

  // Second operand must always be rank 1
  SmallVector<KGEN::ParamDeclAttr> newShapeParamDecls;
  DType newShapeDType =
      cast<TensorType>(operands.getNewShape().getType()).getDtype();
  TensorType newShapeTy =
      getRankedParamTensorType(1, newShapeDType, graphOp, newShapeParamDecls);
  moOperandTypes.push_back(newShapeTy);
  moInputParamDecls.push_back(std::move(newShapeParamDecls));

  // Try to get output rank
  TensorType shapeType = cast<TensorType>(operands.getNewShape().getType());
  auto newRank = shapeType.getNumElements();
  if (newRank.has_value()) {
    SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
    TensorType outputTy = getRankedParamTensorType(newRank.value(), outDType,
                                                   graphOp, outputParamDecls);
    moResultTypes.push_back(outputTy);
    moOutputParamDecls.push_back(std::move(outputParamDecls));
  } else {
    SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
    TensorType outputTy =
        getUnrankedTensorType(outDType, graphOp, outputParamDecls);
    moResultTypes.push_back(outputTy);
    moOutputParamDecls.push_back(std::move(outputParamDecls));
  }
}

void MOBroadcastToOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // First input is not important, always the same
  DType outDType = cast<TensorType>(operands.getInput().getType()).getDtype();
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});

  // Second operand must always be rank-1
  SmallVector<KGEN::ParamDeclAttr> newShapeParamDecls;
  TensorValue bcastShape = cast<TensorValue>(operands.getNewShape());
  if (bcastShape.getType().hasStaticShape()) {
    moOperandTypes.push_back(bcastShape.getType());
    moInputParamDecls.push_back({});
  } else {
    TensorType newShapeTy =
        getRankedParamTensorType(1, outDType, graphOp, newShapeParamDecls);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back(std::move(newShapeParamDecls));
  }

  // Dynamically ranked output passes all verifiers
  if (bcastShape.getType().hasStaticShape()) {
    SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
    TensorType outputTy =
        getRankedParamTensorType(bcastShape.getType().getNumElements().value(),
                                 outDType, graphOp, outputParamDecls);
    moResultTypes.push_back(outputTy);
    moOutputParamDecls.push_back(std::move(outputParamDecls));
  } else {
    SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
    TensorType outputTy =
        getUnrankedTensorType(outDType, graphOp, outputParamDecls);
    moResultTypes.push_back(outputTy);
    moOutputParamDecls.push_back(std::move(outputParamDecls));
  }
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

namespace {
TensorType inferReduceOpReturnType(
    TensorValue input, TensorValue axisValue, DType outputDtype,
    MO::GraphOp graphOp,
    SmallVectorImpl<KGEN::ParamDeclAttr> &outputParamDecls) {
  MLIRContext *ctx = input.getContext();
  TensorType inputTy = input.getType();

  // Unranked input => unranked output.
  if (!inputTy.hasRank())
    return getUnrankedTensorType(outputDtype, graphOp, outputParamDecls);

  std::optional<int64_t> maybeAxis = tryGetConstantInt(axisValue);
  // Unknown axis => ranked dynamic output.
  if (!maybeAxis.has_value())
    return getRankedParamTensorType(inputTy.getRank(), outputDtype, graphOp,
                                    outputParamDecls);

  int64_t axis = normalizeAxis(*maybeAxis, inputTy.getRank());
  SmallVector<TypedAttr> outputShape(inputTy.getTypedShapeAttr().getValues());
  if (axis >= 0 && axis < int64_t(inputTy.getRank()))
    outputShape[axis] = IntegerAttr::get(IndexType::get(ctx), 1);
  return TensorType::get(
      MOSH::ShapeAttr::get(outputShape, MOSH::ShapeType::get(ctx)),
      outputDtype);
}

void inferMOTypesReduceImpl(
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
} // namespace

#define DEFINE_REDUCE_INFER_MO_TYPES(ClassName, ReturnIndices)                 \
  void ClassName::inferMOTypesImpl(                                            \
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,             \
      std::optional<ValueRange> rmoResults,                                    \
      SmallVectorImpl<Type> &moOperandTypes,                                   \
      SmallVectorImpl<Type> &moResultTypes,                                    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) { \
    inferMOTypesReduceImpl(                                                    \
        context, graphOp, operands.getInput().cast<TensorValue>(),             \
        operands.getAxis().cast<TensorValue>(), rmoResults, moOperandTypes,    \
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
void BroadcastLikeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            TensorValue input, TensorValue shapeLike) {
  build(odsBuilder, odsState,
        input.getType().cloneWith(shapeLike.getType().getShapeAttr()), input,
        shapeLike);
}

LogicalResult BroadcastLikeOp::verify() {
  if (getInput().getType().getDtype() != getResult().getType().getDtype())
    return emitError("Input and output tensors do not share dtype!");
  return success();
}

void inferMOTypesScatterImpl(
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

void inferMOTypesElemwiseUnaryImpl(
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

namespace {

void inferConvMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filter, TensorValue strides, TensorValue dilations,
    TensorValue paddings, TensorValue outputPaddings, TensorValue numGroups,
    bool isTranspose, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());

  // input / filter types remain the same.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filter.getType());
  moInputParamDecls.push_back({});

  // strides / dilations / outputPaddings must be 1-D with identical shape.
  // First get or build the common dim value.
  TypedAttr windowParamsDim = nullptr;
  SmallVector<KGEN::ParamDeclAttr> windowParamDecls;
  SmallVector<TensorType, 3> windowParamsTypes = {strides.getType(),
                                                  dilations.getType()};
  if (isTranspose)
    windowParamsTypes.push_back(outputPaddings.getType());
  for (TensorType inputTy : windowParamsTypes)
    if (!windowParamsDim && inputTy.hasRank() && inputTy.getRank() == 1)
      windowParamsDim = inputTy.getTypedShapeAttr().getValues()[0];
  if (!windowParamsDim)
    windowParamsDim = createDynamicDimensionType(graphOp, windowParamDecls);

  // Rebind strides to 1-D.
  moOperandTypes.push_back(
      strides.getType().cloneWith(SmallVector<TypedAttr>{windowParamsDim}));
  moInputParamDecls.push_back(std::move(windowParamDecls));

  // Rebind dilations to 1-D.
  moOperandTypes.push_back(
      dilations.getType().cloneWith(SmallVector<TypedAttr>{windowParamsDim}));
  moInputParamDecls.push_back({});

  // Paddings must be 1-D.
  TensorType paddingsTy = paddings.getType();
  if (!paddingsTy.hasRank() || paddingsTy.getRank() != 1) {
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    moOperandTypes.push_back(getRankedParamTensorType(1, paddingsTy.getDtype(),
                                                      graphOp, paramDecls));
    moInputParamDecls.push_back(std::move(paramDecls));
  } else {
    moOperandTypes.push_back(paddingsTy);
    moInputParamDecls.push_back({});
  }

  if (isTranspose) {
    // Rebind output_paddings to 1D.
    moOperandTypes.push_back(outputPaddings.getType().cloneWith(
        SmallVector<TypedAttr>{windowParamsDim}));
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
  if (!inputTy.hasRank()) {
    moResultTypes.push_back(
        getUnrankedTensorType(outputDtype, graphOp, outputParamDecls));
  } else {
    moResultTypes.push_back(getRankedParamTensorType(
        inputTy.getRank(), outputDtype, graphOp, outputParamDecls));
  }
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

} // namespace

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

namespace {
void inferMOTypesPoolImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue filterShape, TensorValue strides, TensorValue dilations,
    TensorValue paddings, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // input type remains the same.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // filter_shape / strides / dilations must be 1-D with identical shape.
  // First get or build the common dim value.
  TypedAttr windowParamsDim = nullptr;
  SmallVector<KGEN::ParamDeclAttr> windowParamDecls;
  for (TensorType inputTy :
       {filterShape.getType(), strides.getType(), dilations.getType()})
    if (!windowParamsDim && inputTy.hasRank() && inputTy.getRank() == 1)
      windowParamsDim = inputTy.getTypedShapeAttr().getValues()[0];
  if (!windowParamsDim)
    windowParamsDim = createDynamicDimensionType(graphOp, windowParamDecls);

  // Rebind filter_shape to 1D.
  moOperandTypes.push_back(
      filterShape.getType().cloneWith(SmallVector<TypedAttr>{windowParamsDim}));
  // Assign windowParamDecls to the first window operand.
  moInputParamDecls.push_back(std::move(windowParamDecls));

  // Rebind strides to 1D.
  moOperandTypes.push_back(
      strides.getType().cloneWith(SmallVector<TypedAttr>{windowParamsDim}));
  moInputParamDecls.push_back({});

  // Rebind dilations to 1D.
  moOperandTypes.push_back(
      dilations.getType().cloneWith(SmallVector<TypedAttr>{windowParamsDim}));
  moInputParamDecls.push_back({});

  // Paddings must be 1-D.
  TensorType paddingsTy = paddings.getType();
  if (!paddingsTy.hasRank() || paddingsTy.getRank() != 1) {
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    moOperandTypes.push_back(getRankedParamTensorType(1, paddingsTy.getDtype(),
                                                      graphOp, paramDecls));
    moInputParamDecls.push_back(std::move(paramDecls));
  } else {
    moOperandTypes.push_back(paddingsTy);
    moInputParamDecls.push_back({});
  }

  // Infer output type: output rank = input rank.
  // Output is always 4D (for now), but MO verifier requires input-rank ==
  // output-rank.
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType inputTy = input.getType();
  if (!inputTy.hasRank())
    moResultTypes.push_back(
        getUnrankedTensorType(outputDtype, graphOp, outputParamDecls));
  else
    moResultTypes.push_back(getRankedParamTensorType(
        inputTy.getRank(), outputDtype, graphOp, outputParamDecls));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}
} // namespace

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

  auto scalarTensorType = TensorType::get(SmallVector<TypedAttr>{},
                                          start.getType().getElementType());

  moOperandTypes.push_back(scalarTensorType);
  moOperandTypes.push_back(scalarTensorType);
  moOperandTypes.push_back(scalarTensorType);

  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

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

    return TensorType::get(arr, start.getType().getElementType());
  }();

  if (result) {
    moResultTypes.push_back(*result);
    moOutputParamDecls.push_back({});
  } else {
    SmallVector<KGEN::ParamDeclAttr> newDecls;

    TensorType outType = getRankedParamTensorType(1, start.getType().getDtype(),
                                                  graphOp, newDecls);

    moOutputParamDecls.push_back(std::move(newDecls));
    moResultTypes.push_back(outType);
  }
}

namespace {
void inferMOTypesResizeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue size, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` type stays the same.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // `size` must be 1-D.
  TensorType sizeTy = size.getType();
  if (!sizeTy.hasRank() || sizeTy.getRank() != 1) {
    SmallVector<KGEN::ParamDeclAttr> paramDecls;
    moOperandTypes.push_back(
        getRankedParamTensorType(1, sizeTy.getDtype(), graphOp, paramDecls));
    moInputParamDecls.push_back(std::move(paramDecls));
  } else {
    moOperandTypes.push_back(sizeTy);
    moInputParamDecls.push_back({});
  }

  // Infer output type: output rank = input rank.
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TensorType inputTy = input.getType();
  if (!inputTy.hasRank())
    moResultTypes.push_back(
        getUnrankedTensorType(outputDtype, graphOp, outputParamDecls));
  else
    moResultTypes.push_back(getRankedParamTensorType(
        inputTy.getRank(), outputDtype, graphOp, outputParamDecls));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}
} // namespace

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

void RMODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/RMODialect/RMO.cpp.inc"
      >();
}

} // namespace M::RMO

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMO.cpp.inc"
