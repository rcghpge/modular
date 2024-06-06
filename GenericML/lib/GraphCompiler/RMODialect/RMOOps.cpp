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
    ASSERT_STREAM(tensorVal) << "expected tensor value";
    shapes.push_back(tensorVal.getType().getShapeAttr());
  }
  return shapes;
}

// TODO(GRA-551) remove once we eliminate `?` in Graph API.
/// Return whether any Attribute in `attrs` is a `ShapeAttr` that contains
/// the unknown dimension `?`.
static bool hasUnknownDim(ArrayRef<TypedAttr> attrs) {
  for (TypedAttr attr : attrs) {
    if (auto shapeAttr = dyn_cast<ShapeAttr>(attr)) {
      if (shapeAttr.hasUnknownDims())
        return true;
    }
  }
  return false;
}

// TODO(GRA-551) remove once we eliminate `?` in Graph API.
/// Return a shape attribute of given rank, where each dimension is `?`.
static ShapeAttr createUnknownShapeOfRank(MLIRContext *ctx, size_t rank) {
  auto unknownDimAttr = KGEN::UnknownAttr::get(IndexType::get(ctx));
  SmallVector<TypedAttr> dimAttrs(rank, unknownDimAttr);
  return ShapeAttr::get(dimAttrs, MOSH::ShapeType::get(ctx));
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

  ASSERT_STREAM(false) << "Unknown dtype: " << x.getAsString();
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

  ASSERT_STREAM(false) << "Dtype promotion suggestion is an invalid dtype. "
                          "This should not be possible: "
                       << category << ", " << bitWidth;
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

/// Given two tensor types which are meant to broadcast to the same shape
/// return a new type which returns a more "refined type" using information
/// from both existing types
///
/// Only static one dimensions are valid for broadcasting.
/// Any other dimension mismatches will return an error.
static ErrorOr<TensorType> refineBroadcastedShape(TensorType ty1,
                                                  TensorType ty2) {
  ASSERT_STREAM(!ty1.isDynamicallyRanked() && !ty2.isDynamicallyRanked())
      << " only static rank is supported";

  auto idxTy = IndexType::get(ty1.getContext());
  auto intDim1 = IntegerAttr::get(idxTy, 1ll);
  auto refineDim = [&ty1, ty2, intDim1](
                       std::optional<TypedAttr> dim1,
                       std::optional<TypedAttr> dim2) -> ErrorOr<TypedAttr> {
    if (!dim1.has_value())
      return dim2.value();
    if (!dim2.has_value())
      return dim1.value();

    TypedAttr dim1Val = dim1.value();
    TypedAttr dim2Val = dim2.value();

    // Equivalent is always good.
    if (dim1Val == dim2Val)
      return dim1Val;

    // Const 1 broadcasts to anything else.
    if (dim1Val == intDim1)
      return dim2Val;
    if (dim2Val == intDim1)
      return dim1Val;

    // TODO: restrict this further.
    // For now, 2 different parametric dimensions are considered equivalent.
    // The current graph api use `[?] + [?]` which will become `[D1] + [D2]`, so
    // we need to allow it for now.
    if (TensorType::isParameterizedDimParam(dim1Val) &&
        TensorType::isParameterizedDimParam(dim2Val))
      return dim1Val;

    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    StreamAsmPrinter printer(os);
    printer << "Inputs shapes ";
    MOSH::printShapeTypedAttr(printer, ty1.getTypedShapeAttr());
    printer << " and ";
    MOSH::printShapeTypedAttr(printer, ty2.getTypedShapeAttr());
    printer << " are not compatible. Dims \"";
    MOSH::printDimensionValue(printer, dim1Val);
    printer << "\" and \"";
    MOSH::printDimensionValue(printer, dim2Val);
    printer << "\" are neither equivalent nor broadcastable. Insert an "
               "explicit rebind if these dims will be equivalent at runtime.";
    return Error(Twine(errorMsg));
  };

  SmallVector<TypedAttr> newDims;
  newDims.reserve(std::max(ty1.getRank(), ty2.getRank()));
  for (auto [dim1, dim2] : llvm::zip_longest(llvm::reverse(ty1.getDims()),
                                             llvm::reverse(ty2.getDims()))) {
    ErrorOr<TypedAttr> refined = refineDim(dim1, dim2);
    if (refined.isError()) {
      return refined.takeError();
    }
    newDims.push_back(refined.takeValue());
  }

  return ty1.cloneWith(SmallVector<TypedAttr>(llvm::reverse(newDims)));
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryElemwiseImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  bool dynamicRankInputs = llvm::any_of(operands, [](Value v) {
    return cast<TensorType>(v.getType()).isDynamicallyRanked();
  });
  if (dynamicRankInputs) {
    return mlir::emitOptionalError(
        location, OpT::getOperationName(),
        " ops do not support dynamic rank. Use the MO analogue ops instead.");
  }

  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());
  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  ErrorOr<TensorType> resultType = refineBroadcastedShape(xType, yType);
  if (resultType.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid shape: ",
                                   StringRef{resultType.getError()});
  }

  inferredReturnTypes.push_back(
      resultType.takeValue().cloneWith(promotedDtype.takeValue()));
  return success();
}

// Due to resolution of `?`, we may hit cases where we have 2 differently named
// dynamic dimensions. Those results are compatible. A later rebind will resolve
// the issue.
static bool isCompatibleReturnTypesBinaryElemwiseImpl(TypeRange lhs,
                                                      TypeRange rhs) {
  auto lhsType = cast<TensorType>(lhs[0]);
  auto rhsType = cast<TensorType>(rhs[0]);

  if (lhsType.getDtype() != rhsType.getDtype())
    return false;
  if (lhsType.getRank() != rhsType.getRank())
    return false;

  for (size_t i = 0, e = lhsType.getRank(); i < e; i++) {
    TypedAttr lhsDim = lhsType.getDim(i);
    TypedAttr rhsDim = rhsType.getDim(i);

    // If both are paramterized or unknown dims, skip the comparison.
    if ((TensorType::isParameterizedDimParam(lhsDim) ||
         TensorType::isUnknownDimParam(lhsDim)) &&
        (TensorType::isParameterizedDimParam(rhsDim) ||
         TensorType::isUnknownDimParam(rhsDim)))
      continue;

    if (lhsDim != rhsDim)
      return false;
  }
  return true;
}

template <typename OpT>
static LogicalResult inferReturnTypesBinaryComparisonImpl(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  bool dynamicRankInputs = llvm::any_of(operands, [](Value v) {
    return cast<TensorType>(v.getType()).isDynamicallyRanked();
  });
  if (dynamicRankInputs) {
    return mlir::emitOptionalError(
        location, OpT::getOperationName(),
        " ops do not support dynamic rank. Use the MO analogue ops instead.");
  }

  auto xType = cast<TensorType>(operands[0].getType());
  auto yType = cast<TensorType>(operands[1].getType());

  // Even though it is not used in the result type, still calculate the the
  // promotion is valid here.
  ErrorOr<DType> promotedDtype =
      promoteDtype(xType.getDtype(), yType.getDtype());
  if (promotedDtype.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid dtype: ",
                                   StringRef{promotedDtype.getError()});
  }

  ErrorOr<TensorType> resultType = refineBroadcastedShape(xType, yType);
  if (resultType.isError()) {
    return mlir::emitOptionalError(location, "Failed to resolve valid shape: ",
                                   StringRef{resultType.getError()});
  }

  inferredReturnTypes.push_back(resultType.takeValue().cloneWith(DType::kBool));
  return success();
}

static bool isCompatibleReturnTypesBinaryComparisonImpl(TypeRange lhs,
                                                        TypeRange rhs) {
  // Comparisions have all of the same return type compatibility rules as other
  // binary elementwise ops.
  return isCompatibleReturnTypesBinaryElemwiseImpl(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

// TODO(GRA-551) remove once we eliminate `?` in Graph API.
LogicalResult ConcatOp::refineReturnTypes(
    MLIRContext *, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &returnTypes) {
  return success();
}

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto axisAttr = attributes.getAs<IntegerAttr>("axis");
  ASSERT_STREAM(axisAttr) << "rmo.concat must have a constant integer `axis`";

  // Retrieve input shapes
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  // TODO(GRA-551) remove once we eliminate `?` in Graph API.
  if (hasUnknownDim(inputShapes)) {
    auto inputTensorTy = cast<TensorType>(operands[0].getType());
    DType inputDtype = inputTensorTy.getDtype();
    ShapeAttr resultShape =
        createUnknownShapeOfRank(context, inputTensorTy.getRank());
    inferredReturnTypes.push_back(TensorType::get(resultShape, inputDtype));
    return success();
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
  ASSERT_STREAM(!result.isNonMaterializable())
      << "concat shape logic must be materializable";

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
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto newShapeAttr = attributes.getAs<ShapeAttr>("newShape");
  ASSERT_STREAM(newShapeAttr) << "rmo.reshape must have a ShapeAttr `newShape`";
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (newShapeAttr.hasUnknownDims())
    return mlir::emitOptionalError(
        location, "rmo.reshape does not allow for `newShape` to contain "
                  "dynamic dimensions");

  // Retrieve input shape
  SmallVector<TypedAttr> inputShapes = getShapesOf(operands);
  ASSERT_STREAM(inputShapes.size() == 1)
      << "rmo.reshape must have exactly one input";
  auto inputShape = mlir::cast<ShapeAttr>(inputShapes[0]);
  // TODO(GRA-578): Once the graph api has banned dynamic dims, remove this
  // check.
  if (inputShape.hasUnknownDims())
    return mlir::emitOptionalError(
        location, "rmo.reshape does not allow for `inputShape` to contain "
                  "dynamic dimensions");

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
  ASSERT_STREAM(!result.isNonMaterializable())
      << "reshape shape logic must be materializable";

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
