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
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GenericML/Support/Assert.h"

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
TensorType refineShape(TensorType ty1, TensorType ty2) {
  GENERICML_ASSERT(ty1.getDtype() == ty2.getDtype())
      << " tried to refine types with different dtypes";
  bool ty1Ranked = ty1.hasRank();
  bool ty2Ranked = ty2.hasRank();

  // Prefer ranked to unranked tensors
  if (!ty1Ranked && !ty2Ranked) {
    // If both are ranked, prefer a parameterized type over non-parametertized
    if (ty1.hasUnknownShape())
      return ty2;
    return ty1;
  }
  if (!ty1Ranked)
    return ty2;
  if (!ty2Ranked)
    return ty1;

  // Refine each dimension individually
  GENERICML_ASSERT(ty1.getRank() == ty2.getRank())
      << " tried to refine types with different ranks";
  SmallVector<TypedAttr> newDims;
  auto refineDim = [&ty1, &ty2](size_t index) {
    // Prefer anything other than `?`
    if (ty1.isUnknownDim(index))
      return ty2.getDim(index);
    if (ty2.isUnknownDim(index))
      return ty1.getDim(index);

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

  return ty1.cloneWith(newDims);
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

void AddOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // TODO: have adaptor return the correct type
  TensorType lhsType = cast<TensorType>(operands.getInputX().getType());
  TensorType rhsType = cast<TensorType>(operands.getInputY().getType());

  TensorType refinedShape = refineShape(lhsType, rhsType);
  moOperandTypes.push_back(refinedShape);
  moOperandTypes.push_back(refinedShape);
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  moResultTypes.push_back(refinedShape);
  moOutputParamDecls.push_back({});
}

void ShapeOfOp::inferMOTypesImpl(
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

void BroadcastShapeOp::inferMOTypesImpl(
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

void ReshapeOp::inferMOTypesImpl(
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

void BroadcastToOp::inferMOTypesImpl(
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

void CastOp::inferMOTypesImpl(
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

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
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
