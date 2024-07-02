//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect operations that are analogue with MO.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOTypes.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GenericML/GraphCompiler/RMODialect/RMOOps.h"
#include "GenericML/Support/Dims.h"
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
  return IntegerAttr::get(IndexType::get(op.getContext()), dimSize);
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

/// Create a new parameter representing a dynamically ranked shape.
static KGEN::ParamDeclRefAttr createDynamicRankedShapeType(
    MO::GraphOp graphOp, SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls) {
  KGEN::ParamDeclAttr paramDecl = graphOp.getNewShapeParam("RMO_Sh");
  newParamDecls.emplace_back(paramDecl);
  return KGEN::ParamDeclRefAttr::get(paramDecl);
}

static TensorType
getUnrankedTensorType(DType dtype, MO::GraphOp graphOp,
                      SmallVectorImpl<KGEN::ParamDeclAttr> &newParamDecls) {
  TypedAttr dim = createDynamicRankedShapeType(graphOp, newParamDecls);
  TensorType outputTy = TensorType::get(dim, dtype);
  return outputTy;
}

static TensorType
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
  ASSERT_STREAM(!ty1.hasUnknownShape() && !ty2.hasUnknownShape(),
                << " no question marks should be here");
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
  for (size_t i = 0, e = ty1.getRank(); i < e; i++)
    newDims.push_back(refineDim(i));

  return TensorType::get(ty1.getContext(), newDims, dtype, ty1.getMetadata());
}

// Return a TensorType with refined shape information given ty1 and ty2 which
// represent the same underlying shape. Returned type will have a DType matching
// ty1.
TensorType RMO::refineShape(TensorType ty1, TensorType ty2) {
  return refineShapeWithDType(ty1, ty2, ty1.getDtype());
}

SmallVector<Value> RMO::rebindUnknownShapeInValues(ValueRange inputValues,
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
        context, SmallVector<int64_t>{outputRank}, newShapeDType);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back({});
  } else {
    TensorType newShapeTy =
        getRankedParamTensorType(1, newShapeDType, graphOp, newShapeParamDecls);
    moOperandTypes.push_back(newShapeTy);
    moInputParamDecls.push_back(std::move(newShapeParamDecls));
  }

  // Try to get output rank
  TensorType shapeType = cast<TensorType>(operands.getNewShape().getType());
  auto newRank = shapeType.getNumElements();

  // If the output type has rank information, consider using that too
  if (!newRank.has_value() && outputValue && outputValue.getType().hasRank())
    newRank = outputValue.getType().getRank();

  // Set output types
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
    TensorType newShapeTy = getRankedParamTensorType(
        1, bcastShape.getType().getDtype(), graphOp, newShapeParamDecls);
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

static TensorType inferReduceOpReturnType(
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

static void inferConvMOTypesImpl(
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
      windowParamsDim = inputTy.getTypedShapeAttr()[0];
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
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // filter_shape / strides / dilations must be 1-D with identical shape.
  // First get or build the common dim value.
  TypedAttr windowParamsDim = nullptr;
  SmallVector<KGEN::ParamDeclAttr> windowParamDecls;
  for (TensorType inputTy :
       {filterShape.getType(), strides.getType(), dilations.getType()})
    if (!windowParamsDim && inputTy.hasRank() && inputTy.getRank() == 1)
      windowParamsDim = inputTy.getTypedShapeAttr()[0];
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

static void inferMOTypesResizeImpl(
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
  if (!inputTy.hasRank() || !indicesTy.hasRank())
    moResultTypes.push_back(
        getUnrankedTensorType(outputDtype, graphOp, outputParamDecls));
  else
    moResultTypes.push_back(
        getRankedParamTensorType(inputTy.getRank() - 1 + indicesTy.getRank(),
                                 outputDtype, graphOp, outputParamDecls));
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
  if (indicesTy.hasRank() && indicesTy.getRank() > 0)
    kAttr = dyn_cast<IntegerAttr>(indicesTy.getDims().back());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  if (!inputTy.hasRank() || !kAttr) {
    moResultTypes.push_back(
        getUnrankedTensorType(outputDtype, graphOp, outputParamDecls));
  } else {
    // output rank = batch_dims + (indices.rank - batch_dims - 1)
    //                          + (input.rank - batch_dims - K).
    size_t k = kAttr.getInt();
    size_t batchDims = operands.getBatchDims().getInt();
    size_t outputRank =
        (indicesTy.getRank() - 1) + (inputTy.getRank() - batchDims - k);
    moResultTypes.push_back(getRankedParamTensorType(
        outputRank, outputDtype, graphOp, outputParamDecls));
  }
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOMatmulOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue inputA = cast<TensorValue>(operands.getInputA());
  TensorValue inputB = cast<TensorValue>(operands.getInputB());

  // mo::MatmulOp cannot have any outputParamDecls (asserts).
  // Define the operand params only and construct the output shape.
  // (M, N) * (N, P) => (M, P).
  SmallVector<KGEN::ParamDeclAttr> inputADecls, inputBDecls;
  TypedAttr dimM = createDynamicDimensionType(graphOp, inputADecls);
  TypedAttr dimN = createDynamicDimensionType(graphOp, inputADecls);
  TypedAttr dimP = createDynamicDimensionType(graphOp, inputBDecls);

  moOperandTypes.push_back(
      inputA.getType().cloneWith(SmallVector<TypedAttr>{dimM, dimN}));
  moInputParamDecls.push_back(std::move(inputADecls));

  moOperandTypes.push_back(
      inputB.getType().cloneWith(SmallVector<TypedAttr>{dimN, dimP}));
  moInputParamDecls.push_back(std::move(inputBDecls));

  DType outputDType = getDType(rmoResults, 0, inputA.getType().getDtype());
  moResultTypes.push_back(TensorType::get(
      context, SmallVector<TypedAttr>{dimM, dimP}, outputDType));
  moOutputParamDecls.push_back({});
}

void MOBatchMatmulOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue inputA = cast<TensorValue>(operands.getInputA());
  TensorValue inputB = cast<TensorValue>(operands.getInputB());

  // Input types are unchanged.
  moOperandTypes.push_back(inputA.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(inputB.getType());
  moInputParamDecls.push_back({});

  // Output type has same rank than input
  DType outputDType = getDType(rmoResults, 0, inputA.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  if (inputA.getType().hasRank()) {
    int64_t rank = inputA.getType().getRank();
    moResultTypes.push_back(
        getRankedParamTensorType(rank, outputDType, graphOp, outputDecls));
  } else {
    moResultTypes.push_back(
        getUnrankedTensorType(outputDType, graphOp, outputDecls));
  }
  moOutputParamDecls.push_back(std::move(outputDecls));
}

void MOBatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            Type type, Value input_a, Value input_b) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({type});
}

void MOBatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                            Type type, Value input_a, Value input_b,
                            ArrayRef<KGEN::ParamDeclAttr> decls) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({type});
  if (!decls.empty()) {
    odsState.addAttribute(
        getOutputParamDeclsAttrName(odsState.name),
        KGEN::ParamDeclArrayAttr::get(odsBuilder.getContext(), decls));
  }
}

void MOTileOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue repeats = cast<TensorValue>(operands.getRepeats());

  // Try to infer the rank.
  std::optional<int64_t> rank;
  if (input.getType().hasRank())
    rank = input.getType().getRank();
  else if (repeats.getType().hasStaticShape() &&
           repeats.getType().getRank() > 0)
    rank = repeats.getType().getIntDim(0);
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // `input` has dynamic or static rank.
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, input.getType().getDtype(), graphOp,
                                     inputParamDecls)
          : getUnrankedTensorType(input.getType().getDtype(), graphOp,
                                  inputParamDecls));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `repeats` is 1D.
  SmallVector<KGEN::ParamDeclAttr> permParamDecls;
  if (rank.has_value()) {
    // repeats has static shape [rank].
    TypedAttr dimAttr = IntegerAttr::get(IndexType::get(context), *rank);
    moOperandTypes.push_back(
        repeats.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  } else {
    moOperandTypes.push_back(getRankedParamTensorType(
        1, repeats.getType().getDtype(), graphOp, permParamDecls));
  }
  moInputParamDecls.push_back(std::move(permParamDecls));

  // `result` has dynamic or static rank.
  DType outputDtype = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDtype, graphOp,
                                     resultParamDecls)
          : getUnrankedTensorType(outputDtype, graphOp, resultParamDecls));
  moOutputParamDecls.push_back(std::move(resultParamDecls));
}

static void inferMOTypesSqueezeShapeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue inputShape,
    TensorValue indices, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Everything is rank1, use rebinds+checks, shape inference do the right thing
  SmallVector<KGEN::ParamDeclAttr> dimXParamDecls, dimYParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      1, inputShape.getType().getDtype(), graphOp, dimXParamDecls));
  moOperandTypes.push_back(getRankedParamTensorType(
      1, indices.getType().getDtype(), graphOp, dimYParamDecls));
  moInputParamDecls.push_back(std::move(dimXParamDecls));
  moInputParamDecls.push_back(std::move(dimYParamDecls));

  DType outDType = getDType(rmoResults, 0, inputShape.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  moResultTypes.push_back(
      getRankedParamTensorType(1, outDType, graphOp, outputParamDecls));
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

void MOUnsqueezeShapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  inferMOTypesSqueezeShapeImpl(
      context, graphOp, cast<TensorValue>(operands.getInputShape()),
      cast<TensorValue>(operands.getPaddingIndices()), rmoResults,
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

  // Try to infer the rank.
  std::optional<int64_t> rank;
  if (input.getType().hasRank())
    rank = input.getType().getRank();
  else if (perm.getType().hasStaticShape())
    rank = perm.getType().getIntDim(0);
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // `input` has static or dynamic rank.
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, input.getType().getDtype(), graphOp,
                                     inputParamDecls)
          : getUnrankedTensorType(input.getType().getDtype(), graphOp,
                                  inputParamDecls));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `perm` is 1D.
  SmallVector<KGEN::ParamDeclAttr> permParamDecls;
  if (rank.has_value()) {
    // perm has static shape [rank].
    TypedAttr dimAttr = IntegerAttr::get(IndexType::get(context), *rank);
    moOperandTypes.push_back(
        perm.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  } else {
    moOperandTypes.push_back(getRankedParamTensorType(
        1, perm.getType().getDtype(), graphOp, permParamDecls));
  }
  moInputParamDecls.push_back(std::move(permParamDecls));

  // `result` has static or dynamic rank.
  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDType, graphOp,
                                     resultParamDecls)
          : getUnrankedTensorType(outputDType, graphOp, resultParamDecls));
  moOutputParamDecls.push_back(std::move(resultParamDecls));
}

void RMO::inferMOTypesTopKImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input, TensorValue k,
    TensorValue axis, TensorValue sorted, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // Infer the input / output rank.
  std::optional<int64_t> rank;
  if (input.getType().hasRank())
    rank = input.getType().getRank();
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[1].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[1].getType()).getRank();

  // `input` has static or dynamic rank.
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, input.getType().getDtype(), graphOp,
                                     inputParamDecls)
          : getUnrankedTensorType(input.getType().getDtype(), graphOp,
                                  inputParamDecls));
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
  TensorType outputTy =
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDtype, graphOp,
                                     outputParamDecls)
          : getUnrankedTensorType(outputDtype, graphOp, outputParamDecls);
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

void LinAlg::MOSolveOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue rhs = cast<TensorValue>(operands.getRhs());

  // All inputs / output have the same rank.
  // Try to infer the rank.
  std::optional<int64_t> rank;
  if (input.getType().hasRank())
    rank = input.getType().getRank();
  else if (rhs.getType().hasRank())
    rank = rhs.getType().getRank();

  // Set all types to unranked or same rank.
  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, input.getType().getDtype(), graphOp,
                                     inputDecls)
          : getUnrankedTensorType(input.getType().getDtype(), graphOp,
                                  inputDecls));
  moInputParamDecls.push_back(inputDecls);

  SmallVector<KGEN::ParamDeclAttr> rhsDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, rhs.getType().getDtype(), graphOp,
                                     rhsDecls)
          : getUnrankedTensorType(rhs.getType().getDtype(), graphOp, rhsDecls));
  moInputParamDecls.push_back(rhsDecls);

  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  DType outputDType = getDType(rmoResults, 0, input.getType().getDtype());
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDType, graphOp, outputDecls)
          : getUnrankedTensorType(outputDType, graphOp, outputDecls));
  moOutputParamDecls.push_back(outputDecls);
}

void MOConcatOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue axis = cast<TensorValue>(operands.getAxis());

  // Try to infer the rank.
  std::optional<int64_t> rank;
  for (Value input : operands.getInputs()) {
    TensorType inputTy = cast<TensorValue>(input).getType();
    if (inputTy.hasRank()) {
      rank = inputTy.getRank();
      break;
    }
  }
  if (rmoResults.has_value() &&
      cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // Axis must be a scalar.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // All inputs must have the same rank (or a dynamic rank).
  for (Value input : operands.getInputs()) {
    TensorType inputTy = cast<TensorValue>(input).getType();
    SmallVector<KGEN::ParamDeclAttr> inputParamsDecls;
    moOperandTypes.push_back(
        rank.has_value() ? getRankedParamTensorType(*rank, inputTy.getDtype(),
                                                    graphOp, inputParamsDecls)
                         : getUnrankedTensorType(inputTy.getDtype(), graphOp,
                                                 inputParamsDecls));
    moInputParamDecls.push_back(std::move(inputParamsDecls));
  }

  // The output must also have the same (or dynamic) rank.
  DType outputDType =
      getDType(rmoResults, 0,
               cast<TensorValue>(operands.getInputs()[0]).getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> resultParamsDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDType, graphOp,
                                     resultParamsDecls)
          : getUnrankedTensorType(outputDType, graphOp, resultParamsDecls));
  moOutputParamDecls.push_back(std::move(resultParamsDecls));
}

void MOSplitOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue splitSizes = cast<TensorValue>(operands.getSplitSizes());
  TensorValue axis = cast<TensorValue>(operands.getAxis());
  TensorType inputTy = input.getType();
  TensorType splitSizesTy = splitSizes.getType();

  // Try to infer the number of outputs for split.
  int64_t outputsCount;
  if (!rmoResults->empty())
    outputsCount = rmoResults->size();
  else if (splitSizesTy.hasStaticShape() && splitSizesTy.getRank() == 1)
    outputsCount = splitSizesTy.getIntDim(0);
  else {
    llvm_unreachable("split outputs count cannot be inferred");
  }

  // `input` type stays the same.
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // `split_sizes` has shape [outputsCount].
  moOperandTypes.push_back(splitSizesTy.cloneWith(SmallVector<TypedAttr>{
      createStaticDimensionType(graphOp, outputsCount)}));
  moInputParamDecls.push_back({});

  // `axis` must be a scalar.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // All outputs must have the same rank than `input` (or a dynamic rank).
  DType outputDType = getDType(rmoResults, 0, inputTy.getDtype());
  for (int64_t i = 0; i < outputsCount; ++i) {
    SmallVector<KGEN::ParamDeclAttr> resultParamsDecls;
    moResultTypes.push_back(
        inputTy.hasRank()
            ? getRankedParamTensorType(inputTy.getRank(), outputDType, graphOp,
                                       resultParamsDecls)
            : getUnrankedTensorType(outputDType, graphOp, resultParamsDecls));
    moOutputParamDecls.push_back(std::move(resultParamsDecls));
  }
}

static void inferMOTypesPadImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue paddings, TensorValue constant,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Try to infer the input / output rank.
  TensorType inputTy = input.getType();
  TensorType paddingsTy = paddings.getType();
  std::optional<int64_t> rank;
  if (inputTy.hasRank())
    rank = inputTy.getRank();
  else if (paddingsTy.hasStaticShape() && paddingsTy.getIntDim(0) % 2 == 0)
    rank = paddingsTy.getIntDim(0) / 2;
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // Set input rank if possible.
  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, inputTy.getDtype(), graphOp,
                                     inputDecls)
          : getUnrankedTensorType(inputTy.getDtype(), graphOp, inputDecls));
  moInputParamDecls.push_back(std::move(inputDecls));

  // paddings shape must be [2 * rank].
  SmallVector<KGEN::ParamDeclAttr> paddingsDecls;
  TypedAttr outDimAttr =
      rank.has_value() ? createStaticDimensionType(graphOp, *rank * 2)
                       : createDynamicDimensionType(graphOp, paddingsDecls);
  moOperandTypes.push_back(
      paddingsTy.cloneWith(SmallVector<TypedAttr>{outDimAttr}));
  moInputParamDecls.push_back(std::move(paddingsDecls));

  // constant must be scalar.
  if (constant) {
    moOperandTypes.push_back(
        constant.getType().cloneWith(ArrayRef<TypedAttr>{}));
    moInputParamDecls.push_back({});
  }

  // set output rank if possible.
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, inputTy.getDtype(), graphOp,
                                     outputDecls)
          : getUnrankedTensorType(inputTy.getDtype(), graphOp, outputDecls));
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

  // Infer the input / output rank.
  TensorType inputTy = input.getType();
  std::optional<int64_t> rank;
  if (inputTy.hasRank())
    rank = inputTy.getRank();
  else if (start.getType().hasStaticShape() && start.getType().getRank() > 0)
    rank = start.getType().getIntDim(0);
  else if (stop.getType().hasStaticShape() && stop.getType().getRank() > 0)
    rank = stop.getType().getIntDim(0);
  else if (step.getType().hasStaticShape() && step.getType().getRank() > 0)
    rank = step.getType().getIntDim(0);
  else if (rmoResults.has_value() &&
           cast<TensorType>((*rmoResults)[0].getType()).hasRank())
    rank = cast<TensorType>((*rmoResults)[0].getType()).getRank();

  // Set input rank if possible.
  SmallVector<KGEN::ParamDeclAttr> inputDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, inputTy.getDtype(), graphOp,
                                     inputDecls)
          : getUnrankedTensorType(inputTy.getDtype(), graphOp, inputDecls));
  moInputParamDecls.push_back(std::move(inputDecls));

  // start / stop / step have shape `[rank]`.
  SmallVector<KGEN::ParamDeclAttr> startParamDecls;
  // Only assign param declarations to the `start` operand.
  SmallVector<TypedAttr> sliceDims{
      rank.has_value() ? createStaticDimensionType(graphOp, *rank)
                       : createDynamicDimensionType(graphOp, startParamDecls)};

  moOperandTypes.push_back(start.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back(std::move(startParamDecls));
  moOperandTypes.push_back(stop.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(step.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});

  // set output rank if possible.
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, inputTy.getDtype(), graphOp,
                                     outputDecls)
          : getUnrankedTensorType(inputTy.getDtype(), graphOp, outputDecls));
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

  // `input` must be 4D.
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      4, input.getType().getDtype(), graphOp, inputParamDecls));
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // `rois` must be 2D.
  SmallVector<KGEN::ParamDeclAttr> roisParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      2, rois.getType().getDtype(), graphOp, roisParamDecls));
  moInputParamDecls.push_back(std::move(roisParamDecls));

  // `ouput_height` / `output_width` / `spatial_case` / `sampling_ratio` must be
  // scalars.
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
      getRankedParamTensorType(4, outputDType, graphOp, resultParamDecls));
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
      rank.has_value() ? shapeTy.cloneWith(SmallVector<TypedAttr>{
                             createStaticDimensionType(graphOp, *rank)})
                       : getRankedParamTensorType(1, shapeTy.getDtype(),
                                                  graphOp, shapeParamDecls));
  moInputParamDecls.push_back(std::move(shapeParamDecls));

  // `mean` / `variance` / `seed` must be scalars.
  moOperandTypes.push_back(mean.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(variance.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(seed.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Output has static or dynamic rank.
  DType outputDType = getDType(rmoResults, 0, mean.getType().getDtype());
  SmallVector<KGEN::ParamDeclAttr> outputDecls;
  moResultTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, outputDType, graphOp, outputDecls)
          : getUnrankedTensorType(outputDType, graphOp, outputDecls));
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
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  if (!inOutTy.hasRank() && rmoResults.has_value()) {
    // TODO: Is it possible to do refineShape(inputTy, outputTy) here ?
    // It doesn't work when outputTy has param declatations.
    TensorType outputTy = cast<TensorType>((*rmoResults)[0].getType());
    if (outputTy.hasRank())
      inOutTy = getRankedParamTensorType(outputTy.getRank(), inOutTy.getDtype(),
                                         graphOp, inputParamDecls);
  }
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back(std::move(inputParamDecls));
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});

  // `updates` rank could be inferred.
  // updates.shape = indices.shape[:-1] + tensor.shape[indices.shape[-1]:].
  TensorType indicesTy = indices.getType();
  SmallVector<KGEN::ParamDeclAttr> updatesParamDecls;
  if (inOutTy.hasRank() && indicesTy.hasRank() && indicesTy.getRank() > 0 &&
      isa<IntegerAttr>(indicesTy.getDims().back())) {
    int64_t indicesRank = indicesTy.getRank();
    int64_t indicesDepth = indicesTy.getIntDim(indicesRank - 1);
    int64_t updatesRank =
        (indicesRank - 1) + (inOutTy.getRank() - indicesDepth);
    moOperandTypes.push_back(getRankedParamTensorType(
        updatesRank, updates.getType().getDtype(), graphOp, updatesParamDecls));
  } else {
    moOperandTypes.push_back(getUnrankedTensorType(updates.getType().getDtype(),
                                                   graphOp, updatesParamDecls));
  }
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

  // `boxes` and `scores` must be rank 3.
  SmallVector<KGEN::ParamDeclAttr> boxesParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      3, boxes.getType().getDtype(), graphOp, boxesParamDecls));
  moInputParamDecls.push_back(std::move(boxesParamDecls));

  SmallVector<KGEN::ParamDeclAttr> scoresParamDecls;
  moOperandTypes.push_back(getRankedParamTensorType(
      3, scores.getType().getDtype(), graphOp, scoresParamDecls));
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
      getRankedParamTensorType(2, outputDType, graphOp, outputParamDecls));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
}

void MOArgNonzeroOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());

  // Try to infer the input rank.
  // input[rank] => output [non-zero-count, rank].
  std::optional<int64_t> rank;
  if (input.getType().hasRank())
    rank = input.getType().getRank();
  else if (rmoResults.has_value()) {
    TensorType outputTy = cast<TensorType>((*rmoResults)[0].getType());
    if (outputTy.hasRank() && outputTy.getRank() >= 2 &&
        outputTy.isIntegerDim(1))
      rank = outputTy.getIntDim(1);
  }
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  moOperandTypes.push_back(
      rank.has_value()
          ? getRankedParamTensorType(*rank, input.getType().getDtype(), graphOp,
                                     inputParamDecls)
          : input.getType());
  moInputParamDecls.push_back(std::move(inputParamDecls));

  // output shape is `[non-zero-count, rank]`.
  SmallVector<KGEN::ParamDeclAttr> outputParamDecls;
  TypedAttr countAttr = createDynamicDimensionType(graphOp, outputParamDecls);
  TypedAttr rankAttr =
      rank.has_value() ? createStaticDimensionType(graphOp, *rank)
                       : createDynamicDimensionType(graphOp, outputParamDecls);
  moResultTypes.push_back(TensorType::get(
      graphOp->getContext(), SmallVector<TypedAttr>{countAttr, rankAttr},
      DType::si64));
  moOutputParamDecls.push_back(std::move(outputParamDecls));
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
  SmallVector<KGEN::ParamDeclAttr> inputParamDecls;
  if (!inOutTy.hasRank() && rmoResults.has_value()) {
    // Refines the input type if output rank if known.
    TensorType outputTy = cast<TensorType>((*rmoResults)[0].getType());
    if (outputTy.hasRank())
      inOutTy = getRankedParamTensorType(outputTy.getRank(), inOutTy.getDtype(),
                                         graphOp, inputParamDecls);
  }
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back(std::move(inputParamDecls));
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});

  // `axis` type is 0D.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
}
