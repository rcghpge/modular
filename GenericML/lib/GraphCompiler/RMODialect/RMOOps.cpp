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
#include "Support/MDialect/ParserUtils.h"
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

#define DEBUG_TYPE "rmo-ops"

std::optional<M::DeviceRefAttr> M::RMO::getDeviceIfAvailable(Type t) {
  // Right now only tensors/buffers have devices.
  auto asTensorType = dyn_cast<M::MO::TensorType>(t);
  if (asTensorType)
    return asTensorType.getDeviceRef();
  auto asBufferType = dyn_cast<M::MO::BufferType>(t);
  if (asBufferType)
    return asBufferType.getDeviceRef();
  return getHostDevice(t.getContext());
}

std::optional<M::DeviceRefAttr> M::RMO::getDeviceIfAvailable(Value v) {
  return getDeviceIfAvailable(v.getType());
}

M::DeviceRefAttr M::RMO::getDevice(Type t) {
  return getDeviceIfAvailable(t).value();
}

M::DeviceRefAttr M::RMO::getDevice(Value v) {
  return getDeviceIfAvailable(v).value();
}

/// Infer that all types have the same device and return such device
static ErrorOr<M::DeviceRefAttr> inferDeviceAllSameDevices(TypeRange tys) {
  std::optional<M::DeviceRefAttr> device;
  for (auto ty : tys) {
    std::optional<M::DeviceRefAttr> newDevice = getDeviceIfAvailable(ty);
    if (!newDevice.has_value())
      continue;

    if (!device.has_value()) {
      device = std::move(newDevice);
    } else if (newDevice != device) {
      return Error("Not all devices are the same!");
    }
  }
  if (!device.has_value())
    return Error("No device found for inference!");

  return *device;
}

/// Return the first operand's device and check all remaining operands are on
/// host.
static ErrorOr<M::DeviceRefAttr>
inferDeviceAsFirstOperandRestHost(TypeRange tys) {
  std::optional<M::DeviceRefAttr> device = getDeviceIfAvailable(*tys.begin());

  for (auto it = tys.begin() + 1; it != tys.end(); ++it) {
    std::optional<M::DeviceRefAttr> newDevice = getDeviceIfAvailable(*it);
    if (!newDevice.has_value())
      continue;

    auto ctx = (*tys.begin()).getContext();
    if (*newDevice != getHostDevice(ctx))
      return Error("Remaining operands are not on host!");
  }
  if (!device.has_value())
    return Error("No device found on first argument! Inference failed.");

  return *device;
}

static LogicalResult runShapePropagation(
    ValueRange operands, std::optional<Location> location, DType promotedDType,
    MO::ShapeMaterializeResult &materialization,
    SmallVectorImpl<Type> &inferredReturnTypes, M::DeviceRefAttr outputDevice,
    bool validateAsserts = true) {
  // Preliminary error checks.
  if (materialization.isError())
    return mlir::emitOptionalError(location, materialization.getError().msg);

  if (materialization.isNonMaterializable())
    return mlir::emitOptionalError(location,
                                   "shape computation must be materializable");

  auto &result = materialization.getSuccess();

  // Shape propagation over materialized ops and target shape.
  ErrorOr<SmallVector<ShapeAttr>> resultShapesOr =
      MO::propagateShape(result.bindings, result.resultShapes,
                         /*validateAsserts=*/validateAsserts);
  if (resultShapesOr.isError())
    return mlir::emitOptionalError(location, resultShapesOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapesOr.get().front();
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, promotedDType, outputDevice));

  return success();
}

OpFoldResult RebindTensorShapeOp::fold(FoldAdaptor adaptor) {
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

static TypedAttr getShapeOf(Value value) {
  return cast<MO::TensorValue>(value).getType().getShapeAttr();
}

/// ASSUME `values` are all tensor values, map each input value to the shape
/// attribute and return the result.
static SmallVector<TypedAttr> getShapesOf(ValueRange values) {
  return llvm::map_to_vector(values, getShapeOf);
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

/// DTypeCategory is the broad classes of dtypes used in calculating
/// promotions.
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
    // full integer ignoring the sign bit. This means the conversion is
    // exactly representable.
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

    // Float is unsafe if formats differ for same virtual bitwidth (bf16 with
    // f16).
    if ((dtype == DType::f16 && result == DType::bf16) ||
        (dtype == DType::bf16 && result == DType::f16))
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

static LogicalResult inferReturnTypesBinaryElemwiseImpl(
    MLIRContext *context, llvm::StringLiteral opName,
    std::optional<Location> location, TensorValue inputX, TensorValue inputY,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = inputX.getType();
  auto yType = inputY.getType();

  auto xShape = xType.getTypedShapeAttr();
  auto yShape = yType.getTypedShapeAttr();

  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  // Materialize shape function.
  auto graphOp =
      inputX.getParentRegion()->template getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoBinaryElementwiseShape(graphOp, loc, xShape, yShape);

  return runShapePropagation(ValueRange{inputX, inputY}, location,
                             *promotedDtype, result, inferredReturnTypes,
                             getDevice(xType));
}

static LogicalResult inferReturnTypesBinaryComparisonImpl(
    MLIRContext *context, llvm::StringLiteral opName,
    std::optional<Location> location, TensorValue inputX, TensorValue inputY,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = inputX.getType();
  auto yType = inputY.getType();

  auto xShape = xType.getTypedShapeAttr();
  auto yShape = yType.getTypedShapeAttr();

  // Even though it is not used in the result type, still calculate the the
  // promotion is valid here.
  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  // Materialize shape function.
  auto graphOp = inputX.getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoBinaryElementwiseShape(graphOp, loc, xShape, yShape);

  return runShapePropagation(ValueRange{inputX, inputY}, location, DType::kBool,
                             result, inferredReturnTypes, getDevice(xType));
}

static LogicalResult inferReturnTypesBinaryLogicalImpl(
    MLIRContext *context, llvm::StringLiteral opName,
    std::optional<Location> location, TensorValue inputX, TensorValue inputY,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = inputX.getType();
  auto yType = inputY.getType();

  if (!xType.getDtype().isBool() || !yType.getDtype().isBool())
    return mlir::emitOptionalError(
        location, llvm::formatv("{0}: Inputs must be bools, got {1}, {2}",
                                opName, xType.getDtype(), yType.getDtype()));

  return inferReturnTypesBinaryComparisonImpl(context, opName, location, inputX,
                                              inputY, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult
ConcatOp::inferReturnTypes(MLIRContext *context,
                           std::optional<Location> location, Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {
  auto axisAttr = adaptor.getAxisAttr();

  // Retrieve input shapes
  SmallVector<TypedAttr> inputShapes = getShapesOf(adaptor.getInputs());

  // Materialize shape function.
  auto graphOp = adaptor.getOperands()
                     .front()
                     .getParentRegion()
                     ->getParentOfType<MO::GraphOp>();
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
  if (auto outDimAttr =
          dyn_cast_if_present<TypedAttr>(adaptor.getOutDimAttr())) {
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

  DType inputDtype =
      cast<TensorType>(adaptor.getInputs()[0].getType()).getDtype();
  for (auto [i, operand] : llvm::enumerate(adaptor.getInputs())) {
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
  ErrorOr<M::DeviceRefAttr> outputDeviceOr =
      inferDeviceAllSameDevices(adaptor.getInputs());
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  inferredReturnTypes.push_back(
      TensorType::get(resultShape, inputDtype, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult
ReshapeOp::inferReturnTypes(MLIRContext *context,
                            std::optional<Location> location, Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto newShapeAttr = adaptor.getNewShapeAttr();

  // Retrieve input shape
  auto input = cast<TensorValue>(adaptor.getInput());
  auto inputShape = cast<ShapeAttr>(getShapeOf(input));

  // Materialize shape function.
  auto graphOp =
      adaptor.getInput().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeReshapeShape(graphOp, loc, inputShape, newShapeAttr);

  DType inputDtype = input.getType().getDtype();
  return runShapePropagation(adaptor.getOperands(), location, inputDtype,
                             result, inferredReturnTypes, getDevice(input));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastToOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  ShapeAttr newShapeAttr = adaptor.getNewShape();

  auto input = cast<TensorValue>(adaptor.getInput());

  // Retrieve input shape
  auto inputShape = cast<ShapeAttr>(getShapeOf(input));

  // Materialize shape function.
  auto graphOp =
      adaptor.getInput().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeBroadcastToShape(graphOp, loc, inputShape, newShapeAttr);

  DType inputDtype = input.getType().getDtype();
  return runShapePropagation(adaptor.getOperands(), location, inputDtype,
                             result, inferredReturnTypes, getDevice(input));
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult
MatmulOp::inferReturnTypes(MLIRContext *context,
                           std::optional<Location> location, Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {

  // Retrieve input shape
  auto lhs = cast<TensorValue>(adaptor.getInputX());
  auto rhs = cast<TensorValue>(adaptor.getInputY());

  auto lhsShape = cast<ShapeAttr>(getShapeOf(lhs));
  auto rhsShape = cast<ShapeAttr>(getShapeOf(rhs));

  ErrorOr<DType> promotedDtype =
      promoteDtype(lhs.getType().getDtype(), rhs.getType().getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }
  // Materialize shape function.
  auto graphOp = lhs.getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoMatmulShape(graphOp, loc, lhsShape, rhsShape);

  return runShapePropagation(adaptor.getOperands(), location, *promotedDtype,
                             result, inferredReturnTypes, getDevice(lhs));
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOp::inferReturnTypes(MLIRContext *context,
                           std::optional<Location> location, Adaptor adaptor,
                           SmallVectorImpl<Type> &inferredReturnTypes) {

  // Retrieve input shape
  auto condShape = cast<ShapeAttr>(getShapeOf(adaptor.getCond()));
  auto xShape = cast<ShapeAttr>(getShapeOf(adaptor.getInputX()));
  auto yShape = cast<ShapeAttr>(getShapeOf(adaptor.getInputY()));

  ErrorOr<DType> promotedDtype = promoteDtype(
      mlir::cast<TensorType>(adaptor.getInputX().getType()).getDtype(),
      mlir::cast<TensorType>(adaptor.getInputY().getType()).getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }
  // Materialize shape function.
  auto graphOp =
      adaptor.getCond().getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result =
      MO::materializeRmoSelectShape(graphOp, loc, condShape, xShape, yShape);

  return runShapePropagation(adaptor.getOperands(), location, *promotedDtype,
                             result, inferredReturnTypes,
                             getDevice(adaptor.getInputX()));
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

LogicalResult
TopKOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(adaptor.getInput());
  auto inputShape = cast<ShapeAttr>(input.getType().getShapeAttr());

  auto kAttr = adaptor.getKAttr();
  int64_t k = adaptor.getK();
  int64_t axis = adaptor.getAxis();

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

  // Propagate devices by inferring it from operands
  ErrorOr<M::DeviceRefAttr> outputDeviceOr =
      inferDeviceAsFirstOperandRestHost(adaptor.getOperands());
  if (outputDeviceOr.isError())
    return mlir::emitOptionalError(location, outputDeviceOr.getError());

  inferredReturnTypes.push_back(TensorType::get(
      resultShape, input.getType().getDtype(), *outputDeviceOr));
  inferredReturnTypes.push_back(
      TensorType::get(resultShape, DType::si64, *outputDeviceOr));

  return success();
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult
SliceOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  // Retrieve input and start, stop, step shapes.
  auto inputShape = cast<ShapeAttr>(getShapeOf(adaptor.getInput()));

  auto starts = adaptor.getStarts();
  auto stops = adaptor.getStops();

  auto stepsAttr = adaptor.getSteps();
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
  auto input = cast<MO::TensorValue>(adaptor.getInput());
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  Location loc = location.value_or(graphOp.getLoc());
  MO::ShapeMaterializeResult result = MO::materializeRmoSliceShape(
      graphOp, loc, inputShape, starts, stops, steps);

  return runShapePropagation(
      adaptor.getOperands(), location, input.getType().getDtype(), result,
      inferredReturnTypes, getDevice(adaptor.getInput()));
}

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult
ConvOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         Adaptor adaptor,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(adaptor.getInput());
  auto filter = cast<TensorValue>(adaptor.getFilter());

  ErrorOr<DType> promotedDtype =
      promoteDtype(input.getType().getDtype(), filter.getType().getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  TypedAttr inputShape = cast<ShapeAttr>(getShapeOf(input));
  TypedAttr filterShape = cast<ShapeAttr>(getShapeOf(filter));

  Location loc = location.value_or(input.getLoc());

  // Materialize shape function.
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  MO::ShapeMaterializeResult result = MO::materializeConvShape(
      graphOp, loc, inputShape, filterShape, adaptor.getStrides(),
      adaptor.getDilations(), adaptor.getPaddings(), adaptor.getNumGroupsAttr(),
      adaptor.getInputLayoutAttr(),
      cast<MO::TensorType>(adaptor.getFilter().getType()).getLayoutAttr());

  return runShapePropagation(adaptor.getOperands(), location, *promotedDtype,
                             result, inferredReturnTypes, getDevice(input),
                             /*validateAsserts=*/false);
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult ConvTransposeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(adaptor.getInput());
  auto filter = cast<TensorValue>(adaptor.getFilter());

  ErrorOr<DType> promotedDtype =
      promoteDtype(input.getType().getDtype(), filter.getType().getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  TypedAttr inputShape = cast<ShapeAttr>(getShapeOf(input));
  TypedAttr filterShape = cast<ShapeAttr>(getShapeOf(filter));

  Location loc = location.value_or(input.getLoc());

  // Materialize shape function.
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  MO::ShapeMaterializeResult result = MO::materializeConvTransposeShape(
      graphOp, loc, inputShape, filterShape, adaptor.getStrides(),
      adaptor.getDilations(), adaptor.getPaddings(),
      adaptor.getOutputPaddings(), adaptor.getInputLayoutAttr(),
      cast<MO::TensorType>(adaptor.getFilter().getType()).getLayoutAttr());

  return runShapePropagation(adaptor.getOperands(), location, *promotedDtype,
                             result, inferredReturnTypes, getDevice(input),
                             /*validateAsserts=*/false);
}

//===----------------------------------------------------------------------===//
// Pool Ops
//===----------------------------------------------------------------------===//

LogicalResult
MaxPoolOp::inferReturnTypes(MLIRContext *context,
                            std::optional<Location> location, Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(adaptor.getInput());

  TypedAttr inputShape = cast<ShapeAttr>(getShapeOf(input));

  Location loc = location.value_or(input.getLoc());

  // Materialize shape function.
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  MO::ShapeMaterializeResult result = MO::materializePoolShape(
      graphOp, loc, inputShape, adaptor.getFilterShape(), adaptor.getStrides(),
      adaptor.getDilations(), adaptor.getPaddings(), adaptor.getCeilMode());

  return runShapePropagation(adaptor.getOperands(), location,
                             input.getType().getDtype(), result,
                             inferredReturnTypes, getDevice(input),
                             /*validateAsserts=*/false);
}

LogicalResult
AvgPoolOp::inferReturnTypes(MLIRContext *context,
                            std::optional<Location> location, Adaptor adaptor,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = cast<TensorValue>(adaptor.getInput());

  TypedAttr inputShape = cast<ShapeAttr>(getShapeOf(input));

  Location loc = location.value_or(input.getLoc());

  // Materialize shape function.
  auto graphOp = input.getParentRegion()->getParentOfType<MO::GraphOp>();
  MO::ShapeMaterializeResult result = MO::materializePoolShape(
      graphOp, loc, inputShape, adaptor.getFilterShape(), adaptor.getStrides(),
      adaptor.getDilations(), adaptor.getPaddings(), adaptor.getCeilMode());

  return runShapePropagation(adaptor.getOperands(), location,
                             input.getType().getDtype(), result,
                             inferredReturnTypes, getDevice(input),
                             /*validateAsserts=*/false);
}

//===----------------------------------------------------------------------===//
// ShapeToTensor
//===----------------------------------------------------------------------===//

LogicalResult ShapeToTensorOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto shapeAttr = adaptor.getShapeAttr();

  auto dimType = ShapeAttr::getDimType(context);
  M::DeviceRefAttr host = M::DeviceRefAttr::getDefault(context);

  // The output is simply an i64 tensor with rank 1.
  inferredReturnTypes.push_back(TensorType::get(
      ShapeAttr::get({IntegerAttr::get(dimType, shapeAttr.getRank())},
                     shapeAttr.getType()),
      DType::si64, host));

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
