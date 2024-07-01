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
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/ML/RangeUtils.h"
#include <llvm/Support/ErrorHandling.h>
#include <optional>

using namespace M;
using namespace RMO;

using MO::TensorType;
using MO::TensorValue;

using M::MO::parseParamDecls;
using M::MO::printParamDecls;

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
  auto message_attr = odsBuilder.getStringAttr(message);
  return build(odsBuilder, odsState, input.getType().cloneWith(newShape), input,
               outputParamDecls, message_attr);
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
    case 8:
      return DType::f8;
    case 16:
      if (inputBF16)
        return DType::bf16;
      return DType::f16;
    case 19:
      return DType::tf32;
    case 24:
      return DType::f24;
    case 32:
      return DType::f32;
    case 64:
      return DType::f64;
    case 80:
      return DType::f80;
    case 128:
      return DType::f128;
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
    auto build_error = [&]() {
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
      return build_error();

    // Any int to float is only safe if the significand precision can hold the
    // full integer ignoring the sign bit. This means the conversion is exactly
    // representable.
    if (result.isFloat()) {
      std::optional<ssize_t> resultPrecision =
          result.getSignificandPrecisionInBits();
      if (!resultPrecision)
        return Error(Twine("Calculated result type ")
                         .concat(result.getAsString())
                         .concat(" is not supported by the modular stack."));

      if (dtype.isSInt() && dtype.getWidthInBits() - 1 > *resultPrecision)
        return build_error();

      if (dtype.isUInt() && dtype.getWidthInBits() > *resultPrecision)
        return build_error();
    }

    // Float is unsafe if formats differ for same virtual bitwidth (tf32 with
    // f32 or bf16 with f16). I say "virtual bitwidth" cause tf32 is techincally
    // only 19bits.
    if ((dtype == DType::f16 && result == DType::bf16) ||
        (dtype == DType::bf16 && result == DType::f16) ||
        (dtype == DType::tf32 && result == DType::f32))
      return build_error();

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
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
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
  ErrorOr<ShapeAttr> resultShapeOr =
      MO::propagateShape(successRes.bindings, successRes.resultShape);
  if (resultShapeOr.isError())
    return mlir::emitOptionalError(location, resultShapeOr.getError());

  inferredReturnTypes.push_back(
      TensorType::get(*resultShapeOr, *promotedDtype));
  return success();
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryComparisonImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());

  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
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
  ErrorOr<ShapeAttr> resultShapeOr =
      MO::propagateShape(successRes.bindings, successRes.resultShape);
  if (resultShapeOr.isError())
    return mlir::emitOptionalError(location, resultShapeOr.getError());

  inferredReturnTypes.push_back(TensorType::get(*resultShapeOr, DType::kBool));
  return success();
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
  // TODO(GRA-551) remove once we eliminate `?` in Graph API.
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
  ErrorOr<ShapeAttr> resultShapeOr =
      MO::propagateShape(successRes.bindings, successRes.resultShape);
  if (resultShapeOr.isError())
    return mlir::emitOptionalError(location, resultShapeOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapeOr.get();
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
  inferredReturnTypes.push_back(TensorType::get(resultShape, inputDtype));

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
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (newShapeAttr.hasUnknownDims())
    return mlir::emitOptionalError(
        location, unknownDimError("Reshape", "target", newShapeAttr));

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 1,
                << "rmo.reshape must have exactly one input");
  auto inputShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
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
  ErrorOr<ShapeAttr> resultShapeOr =
      MO::propagateShape(successRes.bindings, successRes.resultShape);
  if (resultShapeOr.isError())
    return mlir::emitOptionalError(location, resultShapeOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapeOr.get();
  DType inputDtype = cast<TensorType>(operands[0].getType()).getDtype();
  inferredReturnTypes.push_back(TensorType::get(resultShape, inputDtype));

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
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
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
  ErrorOr<ShapeAttr> resultShapeOr =
      MO::propagateShape(successRes.bindings, successRes.resultShape);
  if (resultShapeOr.isError())
    return mlir::emitOptionalError(location, resultShapeOr.getError());

  // Create output type.
  ShapeAttr resultShape = resultShapeOr.get();
  inferredReturnTypes.push_back(TensorType::get(resultShape, *promotedDtype));

  return success();
}

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
