//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RMO dialect operations that are analogue with MO.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GraphCompiler/RMODialect/RMOOps.h"
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

/// Verify that the RMO op was constructed with an explicit result type.
///
/// RMO MO-analogue ops have outputs whose symbolic dimensions are user-
/// declared per the Graph API contract — the compiler must not invent fresh
/// `RMO_D*` symbols. This helper asserts that the caller supplied an explicit
/// result type and emits an MLIR diagnostic if not, surfacing as a Python
/// exception rather than an abort. Callers should early-return on `false` so
/// `inferMOTypesImpl` leaves `moResultTypes` empty; the failed op
/// construction surfaces the diagnostic at verification time.
static bool requireRmoResults(MO::GraphOp graphOp,
                              std::optional<ValueRange> rmoResults,
                              llvm::StringRef opMnemonic) {
  if (rmoResults.has_value() && !rmoResults->empty())
    return true;
  graphOp->emitOpError()
      << "rmo.mo." << opMnemonic
      << " requires an explicit result type at construction time";
  return false;
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
  // Inputs are rank-1 shape tensors (MO_ShapeLikeIfRanked); pass through.
  moOperandTypes.push_back(cast<TensorType>(operands.getInputX().getType()));
  moOperandTypes.push_back(cast<TensorType>(operands.getInputY().getType()));
  moInputParamDecls.push_back({});
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "broadcast_shape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOReshapeOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` and `new_shape` pass through (new_shape is rank-1 by typed
  // constraint). The result type comes from the user-declared rmo.reshape
  // output.
  moOperandTypes.push_back(cast<TensorType>(operands.getInput().getType()));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(cast<TensorType>(operands.getNewShape().getType()));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "reshape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOBroadcastToOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` and `new_shape` pass through (new_shape is rank-1 by typed
  // constraint).
  moOperandTypes.push_back(cast<TensorType>(operands.getInput().getType()));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(cast<TensorType>(operands.getNewShape().getType()));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "broadcast_to"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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

static void inferMOTypesReduceImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls,
    bool returnIndices) {
  // input type remains the same. `axis` is a compile-time `index` attribute,
  // not an operand, so it contributes no MO operand.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults,
                         returnIndices ? "arg_reduce" : "reduce"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

#define DEFINE_REDUCE_INFER_MO_TYPES(ClassName, ReturnIndices)                 \
  void ClassName::inferMOTypesImpl(                                            \
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,             \
      std::optional<ValueRange> rmoResults,                                    \
      SmallVectorImpl<Type> &moOperandTypes,                                   \
      SmallVectorImpl<Type> &moResultTypes,                                    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,    \
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) { \
    inferMOTypesReduceImpl(context, graphOp,                                   \
                           cast<TensorValue>(operands.getInput()), rmoResults, \
                           moOperandTypes, moResultTypes, moInputParamDecls,   \
                           moOutputParamDecls, ReturnIndices);                 \
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
    TensorValue updates, TensorValue indices,
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

  // The output of scatter has the same type as the input
  moResultTypes.push_back(input.getType());
  moOutputParamDecls.push_back({});
}

void RMO::inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices, TensorValue axis,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

  // The input/updates/indices operands and result type are identical to the
  // attribute-axis `scatter` op.
  inferMOTypesScatterImpl(context, graphOp, input, updates, indices, rmoResults,
                          moOperandTypes, moResultTypes, moInputParamDecls,
                          moOutputParamDecls);

  // The scatter-reduction ops additionally take the axis as a scalar operand.
  // It is appended after `indices`; the result type lives in a separate vector,
  // so operand ordering is preserved.
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
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
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "unary_explicit_output"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
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
  // input / filter / strides / dilations / paddings / output_paddings pass
  // through (rank-1 window operands are enforced by MO_Rank1IndexTensor).
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filter.getType());
  moInputParamDecls.push_back({});
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

  if (!requireRmoResults(graphOp, rmoResults,
                         isTranspose ? "conv_transpose" : "conv"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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
  // input / filter_shape / strides / dilations / paddings pass through
  // (rank-1 window operands are enforced by MO_Rank1IndexTensor).
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(filterShape.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(strides.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(dilations.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(paddings.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "pool"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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
    return;
  }
  if (!requireRmoResults(graphOp, rmoResults, "range"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

static void inferMOTypesResizeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue size, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` and `size` pass through (size is rank-1 by typed constraint).
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(size.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "resize"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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

  // `input` and `indices` types pass through. `axis` is a compile-time `index`
  // attribute and contributes no MO operand.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(indices.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "gather"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOGatherNdOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` and `indices` pass through.
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(operands.getIndices().getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "gather_nd"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOTileOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());
  TensorValue repeats = cast<TensorValue>(operands.getRepeats());

  // `input` passes through.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // `repeats` has static shape [rank].
  TypedAttr dimAttr = ShapeAttr::getDim(context, input.getType().getRank());
  moOperandTypes.push_back(
      repeats.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "tile"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

static void inferMOTypesSqueezeShapeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue inputShape,
    TensorValue indices, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // Inputs are rank-1 shape tensors by typed constraint; pass through.
  moOperandTypes.push_back(inputShape.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(indices.getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "squeeze_shape"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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

  // `input` passes through.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // `perm` has static shape [rank].
  TypedAttr dimAttr = ShapeAttr::getDim(context, input.getType().getRank());
  moOperandTypes.push_back(
      perm.getType().cloneWith(SmallVector<TypedAttr>{dimAttr}));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "transpose"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void RMO::inferMOTypesTopKImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input, TensorValue k,
    TensorValue axis, TensorValue sorted, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` passes through.
  moOperandTypes.push_back(input.getType());
  moInputParamDecls.push_back({});

  // k / axis / sorted are scalars.
  moOperandTypes.push_back(k.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(axis.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(sorted.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // Both result types come from the rmo.top_k declared outputs.
  if (!requireRmoResults(graphOp, rmoResults, "top_k"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[1].getType()));
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
  TensorType inputTy = input.getType();
  int64_t rank = inputTy.getRank();

  // `input` passes through.
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // paddings shape must be [2 * rank].
  moOperandTypes.push_back(paddings.getType().cloneWith(
      SmallVector<TypedAttr>{createStaticDimensionType(graphOp, rank * 2)}));
  moInputParamDecls.push_back({});

  // constant must be scalar.
  if (constant) {
    moOperandTypes.push_back(
        constant.getType().cloneWith(ArrayRef<TypedAttr>{}));
    moInputParamDecls.push_back({});
  }

  if (!requireRmoResults(graphOp, rmoResults, "pad"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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

  TensorType inputTy = input.getType();
  int64_t rank = inputTy.getRank();

  // `input` passes through.
  moOperandTypes.push_back(inputTy);
  moInputParamDecls.push_back({});

  // start / stop / step have shape `[rank]`.
  SmallVector<TypedAttr> sliceDims{createStaticDimensionType(graphOp, rank)};
  moOperandTypes.push_back(start.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(stop.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(step.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "slice"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MORoiAlignOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue outputHeight = cast<TensorValue>(operands.getOutputHeight());
  TensorValue outputWidth = cast<TensorValue>(operands.getOutputWidth());
  TensorValue spatialScale = cast<TensorValue>(operands.getSpatialScale());
  TensorValue samplingRatio = cast<TensorValue>(operands.getSamplingRatio());

  // `input` and `rois` pass through.
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(operands.getRois().getType());
  moInputParamDecls.push_back({});

  // `output_height` / `output_width` / `spatial_scale` / `sampling_ratio`
  // must be scalars.
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

  if (!requireRmoResults(graphOp, rmoResults, "roi_align"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MORandomNormalOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue mean = cast<TensorValue>(operands.getMean());
  TensorValue variance = cast<TensorValue>(operands.getVariance());
  TensorValue seed = cast<TensorValue>(operands.getSeed());

  // `shape` passes through (rank-1 by typed constraint).
  moOperandTypes.push_back(operands.getShape().getType());
  moInputParamDecls.push_back({});

  // `mean` / `variance` are scalars.
  moOperandTypes.push_back(mean.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(variance.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  // `seed` is rank-1 `[1]`.
  moOperandTypes.push_back(seed.getType().cloneWith(
      SmallVector<TypedAttr>{createStaticDimensionType(graphOp, 1)}));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "random_normal"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MORandomUniformOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue lowerBound = cast<TensorValue>(operands.getLowerBound());
  TensorValue upperBound = cast<TensorValue>(operands.getUpperBound());
  TensorValue seed = cast<TensorValue>(operands.getSeed());

  // `shape` passes through (rank-1 by typed constraint).
  moOperandTypes.push_back(operands.getShape().getType());
  moInputParamDecls.push_back({});

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

  if (!requireRmoResults(graphOp, rmoResults, "random_uniform"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
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
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back({});
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});

  // `updates` and `indices` pass through.
  moOperandTypes.push_back(updates.getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(indices.getType());
  moInputParamDecls.push_back({});
}

void MONonMaximumSuppressionOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue maxOutputBoxesPerClass =
      cast<TensorValue>(operands.getMaxOutputBoxesPerClass());
  TensorValue iouThreshold = cast<TensorValue>(operands.getIouThreshold());
  TensorValue scoreThreshold = cast<TensorValue>(operands.getScoreThreshold());

  // `boxes` and `scores` pass through.
  moOperandTypes.push_back(operands.getBoxes().getType());
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(operands.getScores().getType());
  moInputParamDecls.push_back({});

  // `maxOutputBoxesPerClass` / `iouThreshold` / `scoreThreshold` must be
  // scalars.
  moOperandTypes.push_back(
      maxOutputBoxesPerClass.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      iouThreshold.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(
      scoreThreshold.getType().cloneWith(ArrayRef<TypedAttr>{}));
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "non_maximum_suppression"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOArgNonzeroOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  // `input` passes through. Result is the user-declared rank-2 si64 type
  // (MO_Rank2I64Tensor; the count dim is user-supplied per the Graph API
  // contract).
  moOperandTypes.push_back(operands.getInput().getType());
  moInputParamDecls.push_back({});

  if (!requireRmoResults(graphOp, rmoResults, "arg_nonzero"))
    return;
  moResultTypes.push_back(cast<TensorType>((*rmoResults)[0].getType()));
  moOutputParamDecls.push_back({});
}

void MOCumSumOp::inferMOTypesImpl(
    MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
  TensorValue input = cast<TensorValue>(operands.getInput());

  // `input` / `output` types are identical. `axis` is a compile-time `index`
  // attribute and contributes no MO operand.
  TensorType inOutTy = input.getType();
  moOperandTypes.push_back(inOutTy);
  moInputParamDecls.push_back({});
  moResultTypes.push_back(inOutTy);
  moOutputParamDecls.push_back({});
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
  TensorValue start = cast<TensorValue>(operands.getStart());
  TensorValue stop = cast<TensorValue>(operands.getStop());
  TensorValue step = cast<TensorValue>(operands.getStep());
  auto rank = inTensorTy.getRank();

  // `in_buffer` and `slice` pass through.
  moOperandTypes.push_back(inBufferTy);
  moInputParamDecls.push_back({});
  moOperandTypes.push_back(operands.getSlice().getType());
  moInputParamDecls.push_back({});

  // start / stop / step have shape `[rank]`.
  SmallVector<TypedAttr> sliceDims{createStaticDimensionType(graphOp, rank)};
  moOperandTypes.push_back(start.getType().cloneWith(sliceDims));
  moInputParamDecls.push_back({});
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
