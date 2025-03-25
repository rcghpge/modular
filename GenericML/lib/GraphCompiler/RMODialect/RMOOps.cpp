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
#include "GenericML/GraphCompiler/MODialect/ShapeFunctions.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

using namespace M;
using namespace RMO;

using MO::TensorType;
using MO::TensorValue;
using MO::Var;

using M::MO::parseParamDecls;
using M::MO::printParamDecls;

ErrorOr<M::DeviceRefAttr> RMO::inferDevice(ValueRange values) {
  llvm::SmallDenseSet<M::DeviceRefAttr> valueDevices;
  for (const auto &value : values) {
    // Right now only tensors/buffers have devices.
    auto asTensorType = dyn_cast<M::MO::TensorType>(value.getType());
    if (asTensorType)
      if (M::DeviceRefAttr deviceRef = asTensorType.getDeviceRef())
        valueDevices.insert(deviceRef);
    auto asBufferType = dyn_cast<M::MO::BufferType>(value.getType());
    if (asBufferType)
      if (M::DeviceRefAttr deviceRef = asBufferType.getDeviceRef())
        valueDevices.insert(deviceRef);
  }

  if (valueDevices.size() > 1) {
    // In the case where we have different explicit devices, error.
    std::string devicesString = "";
    for (const auto &device : valueDevices) {
      devicesString.append(device.toString());
      devicesString.append(",");
    }
    return Error(Twine("Differing input devices={")
                     .concat(devicesString)
                     .concat("} is not supported."));
  } else if (valueDevices.size() == 1) {
    // There is only 1 explicit device. Use that as the sole device.
    return *valueDevices.begin();
  } else {
    return M::DeviceRefAttr();
  }
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

void RebindTensorShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TensorValue input, ShapeAttr newShape,
                                StringRef message,
                                ArrayRef<ParamDeclAttr> outputParamDecls) {
  auto messageAttr = odsBuilder.getStringAttr(message);
  return build(odsBuilder, odsState, input.getType().cloneWith(newShape), input,
               outputParamDecls, messageAttr);
}

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

/// ASSUME `values` are all tensor values, map each input value to the shape
/// attribute and return the result.
static SmallVector<TypedAttr> getShapesOf(ValueRange values) {
  SmallVector<TypedAttr> shapes;
  for (Value val : values) {
    auto tensorVal = dyn_cast<MO::TensorValue>(val);
    ASSERT_STREAM(tensorVal, << "expected tensor value");
    shapes.push_back(tensorVal.getType().getShapeAttr());
  }
  return shapes;
}

// Creates an error message for when a tensor contains `?`, but that is not
// allowed.
std::string unknownDimError(const StringRef opName, const StringRef tensorName,
                            ShapeAttr tensorShape) {
  std::string errorMsg;
  llvm::raw_string_ostream os(errorMsg);
  StreamAsmPrinter printer(os);
  printer << opName << " does not allow for " << tensorName << " (shape ";
  MOSH::printShapeTypedAttr(printer, tensorShape);
  printer << ") to contain dynamic dimensions. Please rebind the dimensions to "
             "give them names or integer values.";
  return errorMsg;
}

/// DTypeCategory is the broad classes of dtypes used in calculating promotions.
enum DTypeCategory {
  kBool,
  kUInt,
  kSInt,
  kFloat,
};

/// Returns the DTypeCategory for a DType.
static DTypeCategory getDtypeCategory(DType x) {
  if (x.isBool())
    return kBool;
  if (x.isUInt())
    return kUInt;
  if (x.isSInt())
    return kSInt;
  if (x.isFloat())
    return kFloat;

  ASSERT_STREAM(false, << "Unknown dtype: " << x.getAsString());
  llvm_unreachable("Unknown dtype");
}

/// Returns the dtype represented by a category, bit width, and format.
static DType calcDtype(DTypeCategory category, ssize_t bitWidth,
                       bool inputBF16) {
  switch (category) {

  case kBool:
    return DType::kBool;
    break;
  case kUInt:
    switch (bitWidth) {
    case 1:
      return DType::ui1;
    case 2:
      return DType::ui2;
    case 4:
      return DType::ui4;
    case 8:
      return DType::ui8;
    case 16:
      return DType::ui16;
    case 32:
      return DType::ui32;
    case 64:
      return DType::ui64;
    case 128:
      return DType::ui128;
    }
    break;
  case kSInt:
    switch (bitWidth) {
    case 1:
      return DType::si1;
    case 2:
      return DType::si2;
    case 4:
      return DType::si4;
    case 8:
      return DType::si8;
    case 16:
      return DType::si16;
    case 32:
      return DType::si32;
    case 64:
      return DType::si64;
    case 128:
      return DType::si128;
    }
    break;
  case kFloat:
    switch (bitWidth) {
    case 16:
      if (inputBF16)
        return DType::bf16;
      return DType::f16;
    case 19:
      return DType::tf32;
    case 32:
      return DType::f32;
    case 64:
      return DType::f64;
    }
    break;
  }

  ASSERT_STREAM(false, << "Dtype promotion suggestion is an invalid dtype. "
                          "This should not be possible: "
                       << category << ", " << bitWidth);
  llvm_unreachable("Invalid dtype");
}

/// Given two dtypes, returns the promotion dtype or an error.
/// The rules are:
/// 1. Category is an ordered hierarchy: `bool < unsigned int < signed int <
/// float`
/// 2. Promotion of values `x` and `y` picks:
///    (max(category(x), category(y)), max(bitwidth(x), bitwidth(y))
/// 3. Promotion raises an error if either input dtype might contain a value
///    unrepresentable by the chosen promotion type.
/// 4. Promotion raises an error if two dtypes have the same category and
/// bitwidth
///    but different representations.
ErrorOr<DType> RMO::promoteDtype(DType x, DType y) {

  // The common case where the two types are the same.
  if (x == y)
    return x;

  // Bool is lowest class and always promotes to the other type.
  if (x.isBool())
    return y;
  if (y.isBool())
    return x;

  bool inputBF16 = x == DType::bf16 || y == DType::bf16;
  DTypeCategory category = std::max(getDtypeCategory(x), getDtypeCategory(y));
  ssize_t bitWidth = std::max(x.getWidthInBits(), y.getWidthInBits());
  DType result = calcDtype(category, bitWidth, inputBF16);

  auto isSafeToCast = [&](DType dtype) -> ErrorOrSuccess {
    auto buildError = [&]() {
      return Error(Twine("Unsafe cast from ")
                       .concat(dtype.getAsString())
                       .concat(" to ")
                       .concat(result.getAsString())
                       .concat(". Insert an explicit cast op if this "
                               "conversion is wanted."));
    };

    // Unsigned int to signed int is only safe if the bitwidth increases
    // such that all values are exactly representable.
    if (dtype.isUInt() && result.isSInt() &&
        dtype.getWidthInBits() == result.getWidthInBits())
      return buildError();

    // Any int to float is only safe if the significand precision can hold the
    // full integer ignoring the sign bit. This means the conversion is exactly
    // representable.
    if (result.isFloat()) {
      const llvm::fltSemantics *semantics = result.getFloatSemantics();
      if (!semantics)
        return Error(Twine("Calculated result type ")
                         .concat(result.getAsString())
                         .concat(" is not supported by the modular stack."));

      ssize_t resultPrecision = APFloat::semanticsPrecision(*semantics);
      if (dtype.isSInt() && dtype.getWidthInBits() - 1 > resultPrecision)
        return buildError();

      if (dtype.isUInt() && dtype.getWidthInBits() > resultPrecision)
        return buildError();
    }

    // Float is unsafe if formats differ for same virtual bitwidth (tf32 with
    // f32 or bf16 with f16). I say "virtual bitwidth" cause tf32 is techincally
    // only 19bits.
    if ((dtype == DType::f16 && result == DType::bf16) ||
        (dtype == DType::bf16 && result == DType::f16) ||
        (dtype == DType::tf32 && result == DType::f32))
      return buildError();

    return success();
  };

  ErrorOrSuccess xCastSafe = isSafeToCast(x);
  if (xCastSafe.isError())
    return xCastSafe.takeError();

  ErrorOrSuccess yCastSafe = isSafeToCast(y);
  if (yCastSafe.isError())
    return yCastSafe.takeError();

  return result;
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryElemwiseImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  auto xShape = xType.getTypedShapeAttr();
  if (xShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError(OpT::getOperationName(), "lhs", xShape));
  auto yShape = yType.getTypedShapeAttr();
  if (yShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError(OpT::getOperationName(), "rhs", yShape));

  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoBinaryElementwiseShape(graphOp, loc, xShape, yShape);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << OpT::getOperationName()
                << " shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  inferredReturnTypes.push_back(TensorType::get(
      resultShapesOr->front(), *promotedDtype, *outputDeviceOr));
  return success();
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryComparisonImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());

  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  auto xShape = xType.getTypedShapeAttr();
  if (xShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError(OpT::getOperationName(), "lhs", xShape));
  auto yShape = yType.getTypedShapeAttr();
  if (yShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError(OpT::getOperationName(), "rhs", yShape));

  // Even though it is not used in the result type, still calculate the the
  // promotion is valid here.
  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoBinaryElementwiseShape(graphOp, loc, xShape, yShape);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << OpT::getOperationName()
                << " shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  inferredReturnTypes.push_back(
      TensorType::get(resultShapesOr->front(), DType::kBool, *outputDeviceOr));
  return success();
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryLogicalImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());

  if (!xType.getDtype().isBool() || !yType.getDtype().isBool())
    return mlir::emitOptionalError(
        location, llvm::formatv("{0}: Inputs must be bools, got {1}, {2}",
                                OpT::getOperationName(), xType.getDtype(),
                                yType.getDtype()));

  return inferReturnTypesBinaryComparisonImpl<OpT>(
      context, location, operands, attributes, properties, regions,
      inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto axisAttr = attributes.getAs<IntegerAttr>("axis");
  ASSERT_STREAM(axisAttr, << "rmo.concat must have a constant integer `axis`");

  // Retrieve input shapes
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  // TODO(GEX-551) remove once we eliminate `?` in Graph API.
  for (auto [i, inputShapeAttr] : llvm::enumerate(inputShapes)) {
    auto inputShape = mlir::cast<ShapeAttr>(inputShapeAttr);
    if (inputShape.hasUnknownDims())
      return mlir::emitOptionalError(
          location,
          unknownDimError("Concat", "input" + std::to_string(i), inputShape));
  }

  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeConcatShape(graphOp, loc, axisAttr, inputShapes);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "concat shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr.get().front();
  // If the user set the output dimensions, This is an explicit rebind request
  // from the user. We trust their input here for shape inference. A rebind is
  // inserted in lowering to handle this shape specification at runtime.
  if (auto outDimAttr = attributes.getAs<TypedAttr>("out_dim")) {
    size_t rank = resultShape.getRank();
    SmallVector<TypedAttr> resultDims{};
    resultDims.reserve(rank);
    int64_t axis = axisAttr.getInt();
    axis = axis < 0 ? axis + rank : axis;

    for (size_t i = 0; i < rank; ++i) {
      if (static_cast<size_t>(axis) == i) {
        resultDims.push_back(outDimAttr);
        continue;
      }
      resultDims.push_back(resultShape[i]);
    }
    resultShape = ShapeAttr::get(resultDims, resultShape.getType());
  }

  DType inputDtype = cast<TensorType>(operands[0].getType()).getDtype();
  for (auto [i, operand] : llvm::enumerate(operands)) {
    DType opDtype = cast<TensorType>(operand.getType()).getDtype();
    if (opDtype != inputDtype) {
      std::string errorMsg;
      llvm::raw_string_ostream os(errorMsg);
      StreamAsmPrinter printer(os);
      printer << "Concat does not allow for input" << i << " (dtype " << opDtype
              << ") to have a differing dtype from input0 (dtype " << inputDtype
              << ").";
      return mlir::emitOptionalError(location, errorMsg);
    }
  }
  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  inferredReturnTypes.push_back(
      TensorType::get(resultShape, inputDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto newShapeAttr = attributes.getAs<ShapeAttr>("newShape");
  ASSERT_STREAM(newShapeAttr,
                << "rmo.reshape must have a ShapeAttr `newShape`");
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (newShapeAttr.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Reshape", "target", newShapeAttr));

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 1,
                << "rmo.reshape must have exactly one input");
  auto inputShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (inputShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Reshape", "input", inputShape));

  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeReshapeShape(graphOp, loc, inputShape, newShapeAttr,
                                  /*allowParametricDenominator=*/true);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "reshape shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr.get().front();
  DType inputDtype = cast<TensorType>(operands[0].getType()).getDtype();
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, inputDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastToOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto newShapeAttr = attributes.getAs<ShapeAttr>("newShape");
  ASSERT_STREAM(newShapeAttr,
                << "rmo.broadcast_to must have a ShapeAttr `newShape`");
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (newShapeAttr.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("BroadcastTo", "target", newShapeAttr));

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 1,
                << "rmo.broadcast_to must have exactly one input");
  auto inputShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (inputShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("BroadcastTo", "input", inputShape));

  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeBroadcastToShape(graphOp, loc, inputShape, newShapeAttr);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "broadcast_to shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr->front();
  DType inputDtype = cast<TensorType>(operands[0].getType()).getDtype();
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, inputDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 2,
                << "rmo.matmul must have exactly two inputs");

  auto lhsShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  auto rhsShape = mlir::cast<ShapeAttr>(inputShapes[1]);
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (lhsShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location,
        unknownDimError("Matrix multiplication", "input lhs", lhsShape));
  if (rhsShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location,
        unknownDimError("Matrix multiplication", "input rhs", rhsShape));

  ErrorOr<DType> promotedDtype =
      promoteDtype(mlir::cast<TensorType>(operands[0].getType()).getDtype(),
                   mlir::cast<TensorType>(operands[1].getType()).getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }
  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoMatmulShape(graphOp, loc, lhsShape, rhsShape);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "matmul shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr.get().front();
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, *promotedDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 3,
                << "rmo.select must have exactly 3 inputs");

  auto condShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  auto xShape = mlir::cast<ShapeAttr>(inputShapes[1]);
  auto yShape = mlir::cast<ShapeAttr>(inputShapes[2]);
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (condShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Select", "input cond", condShape));
  if (xShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Select", "input x", xShape));
  if (yShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Select", "input y", yShape));

  ErrorOr<DType> promotedDtype =
      promoteDtype(mlir::cast<TensorType>(operands[1].getType()).getDtype(),
                   mlir::cast<TensorType>(operands[2].getType()).getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }
  // Materialize shape function.
  auto graphOp =
      operands.front().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoSelectShape(graphOp, loc, condShape, xShape, yShape);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "matmul shape logic must be materializable");

  // Shape propagation over materialized ops and target shape.
  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(successRes.bindings, successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr.get().front();
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, *promotedDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

LogicalResult
TopKOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         mlir::OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(operands[0]);
  auto inputShape = cast<ShapeAttr>(input.getType().getShapeAttr());
  auto kAttr = attributes.getAs<IntegerAttr>("k");
  const int64_t k = kAttr.getSInt();
  int64_t axis = attributes.getAs<IntegerAttr>("axis").getSInt();

  // Handle negative axis.
  const int64_t rank = inputShape.getRank();
  axis = axis < 0 ? axis + rank : axis;
  if (axis < 0 || axis >= rank)
    return emitOptionalError(location,
                             llvm::formatv("axis ({0}) out of bounds", axis));

  TypedAttr axisDim = inputShape[axis];
  if (auto intDim = dyn_cast<IntegerAttr>(axisDim)) {
    // Validate k when axis is statically known.
    int64_t axisDimInt = intDim.getValue().getSExtValue();
    if (k > axisDimInt) {
      return emitOptionalError(
          location,
          llvm::formatv(
              "k ({0}) cannot be larger than dimension size along axis {1}", k,
              axisDimInt));
    }
  }

  SmallVector<TypedAttr> outDims;
  for (int i = 0; i < rank; ++i) {
    if (i == axis)
      outDims.push_back(kAttr);
    else
      outDims.push_back(inputShape[i]);
  }

  auto resultShape = ShapeAttr::get(outDims, inputShape.getType());

  DeviceRefAttr deviceRef = input.getType().getDeviceRef();
  if (!deviceRef)
    return emitOptionalError(
        location, "rmo.top_k requires input tensor to have an assigned device");

  inferredReturnTypes.push_back(
      TensorType::get(resultShape, input.getType().getDtype(), deviceRef));
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, DType::si64, deviceRef));

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Retrieve input and start, stop, step shapes.
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 1,
                << "rmo.slice must have exactly 1 input");

  auto inputShape = cast<ShapeAttr>(inputShapes[0]);
  auto starts = attributes.getAs<ShapeAttr>("starts");
  auto stops = attributes.getAs<ShapeAttr>("stops");

  auto stepsAttr = attributes.getAs<ShapeAttr>("steps");
  auto maybeSteps = MO::tryGetConstantIntVector(stepsAttr);
  if (!maybeSteps ||
      llvm::any_of(*maybeSteps, [](int64_t step) { return step == 0; }))
    return mlir::emitOptionalError(
        location,
        llvm::formatv("rmo.slice expects static steps != 0 but got {0}",
                      stepsAttr));
  const SmallVector<int64_t> &steps = *maybeSteps;

  // Check that the input rank matches the provided starts/stop/steps sizes.
  if (starts.getSize() != inputShape.getRank() ||
      !llvm::all_equal({starts.getSize(), stops.getSize(), steps.size()})) {
    return mlir::emitOptionalError(
        location,
        "Input rank must match the number of elements in starts/stops/steps "
        "bounds");
  }

  // Materialize shape function.
  auto input = cast<MO::TensorValue>(operands.front());
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result = MO::materializeRmoSliceShape(
      graphOp, loc, inputShape, starts, stops, steps);

  // Preliminary error checks.
  if (result.isError())
    return mlir::emitOptionalError(location, result.getError().msg);

  // This depends on GraphAPI having static rank.
  ASSERT_STREAM(!result.isNonMaterializable(),
                << "matmul shape logic must be materializable");

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(input);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  auto &successRes = result.getSuccess();
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr = MO::propagateShape(
      successRes.bindings, /*targetShapes=*/successRes.resultShapes);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());
  inferredReturnTypes.push_back(TensorType::get(
      resultShapesOr->front(), input.getType().getDtype(), *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// ShapeToTensor
//===----------------------------------------------------------------------===//

LogicalResult ShapeToTensorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto shapeAttr = attributes.getAs<ShapeAttr>("shape");
  ASSERT_STREAM(
      shapeAttr, << "rmo.shape_to_tensor must have a ShapeAttr `shape`");
  // TODO(GEX-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (shapeAttr.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("ShapeToTensor", "shape", shapeAttr));

  auto dimType = ShapeAttr::getDimType(context);

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr = inferDevice(operands);
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  // The output is simply an i64 tensor with rank 1.
  inferredReturnTypes.push_back(TensorType::get(
      ShapeAttr::get({IntegerAttr::get(dimType, shapeAttr.getRank())},
                     shapeAttr.getType()),
      DType::si64, *outputDeviceOr));

  return success();
}

bool ShapeToTensorOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  for (auto [a, b] : llvm::zip(lhs, rhs)) {
    if (a == b)
      continue;
    auto aAsTensor = dyn_cast<MO::TensorType>(a);
    auto bAsTensor = dyn_cast<MO::TensorType>(b);

    // Device may be provided directly by original creator -- no way to infer
    if (aAsTensor && bAsTensor &&
        aAsTensor.cloneWith(DeviceRefAttr{}) ==
            bAsTensor.cloneWith(DeviceRefAttr{}))
      continue;

    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Op registration
//===----------------------------------------------------------------------===//

void RMODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/RMODialect/RMO.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/RMODialect/RMO.cpp.inc"
