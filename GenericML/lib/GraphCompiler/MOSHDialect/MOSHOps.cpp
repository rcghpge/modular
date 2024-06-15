//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOSH dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/ShapeFuncHelpers.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// ParamToValueOp support
//===----------------------------------------------------------------------===//

static ParseResult parseParamToValueOpValue(OpAsmParser &p, TypedAttr &value,
                                            Type &resultType) {
  if (KGEN::parseColonTypeOrIndex(p, resultType) || p.parseEqual() ||
      p.parseLess() || KGEN::parseParamValue(p, value, resultType) ||
      p.parseGreater())
    return failure();
  return success();
}

static void printParamToValueOpValue(OpAsmPrinter &p, Operation *,
                                     TypedAttr value, Type resultType) {
  KGEN::printColonTypeOrIndex(p, value.getType());
  p << " = <";
  KGEN::printParamValue(p, value);
  p << ">";
}

LogicalResult ParamToValueOp::verify() {
  Type paramType = getValue().getType();
  Type resultType = getType();
  if (paramType == resultType)
    return success();

  return emitOpError() << "parameter type " << paramType
                       << " does not match result type " << resultType;
}

void ParamToValueOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// Shape ops
//===----------------------------------------------------------------------===//

// NOTE Why do we need these folders?
// Shape inference does symbolic simplifications like `add(X, 0) --> X`, and
// these simplification logic are rooted in the constructors of
// `KGEN::ParamOperatorAttr` (the attribute we use to represent symbolic
// expressions). However KGEN has a pre-defined set of "ops" they support in
// `KGEN::ParamOperatorAttr`, how do we define the shape ops (like concat) that
// need simplification? The answer is `KGEN::MLIROpAttr`, which requires us to
// define the simplification logic of the shape ops as MLIR folders.

// NOTE Why do we need these canonicalizers? See comments in
// `StaticizeShapesPass`.

// Below are some abstractions to help us get the same outcome from running
// `--canonicalize` on some MOSH ops, as if `MOSH::ParamToValueOp` is a
// ConstantLike, without having to duplicate the folding across `fold` and
// `canonicalize`. (Recall that folders are necessary, but we can't materialize
// their output attribute into `MOSH::ParamToValueOp` anymore so
// `--canonicalize` won't work without these canonicalizers).

// Try to apply `op`'s folder to its input attributes (if any), and return the
// result attribute if successful, else return an error.
static ErrorOr<TypedAttr> tryFoldOp(Operation *op) {
  ASSERT_STREAM(isa<MOSHDialect>(op->getDialect()), << "expected MOSH op");

  // Turn inputs into attributes, this works because we know as far as mosh ops
  // are concerned, the inputs are either parametric shape/dimension or index
  // constant, and we know what ops they'd come from.
  SmallVector<Attribute> attrs;
  for (auto input : op->getOperands()) {
    if (auto paramToValueOp = input.getDefiningOp<ParamToValueOp>()) {
      attrs.push_back(paramToValueOp.getValue());

    } else {
      return Error("An input isn't from ParamToValueOp");
    }
  }

  // Run this op's folder.
  SmallVector<OpFoldResult> foldResults;
  if (failed(op->fold(attrs, foldResults)))
    return Error("Failed to fold the op");

  // Collect the singular `TypedAttr` result from folder, or fail.
  if (foldResults.size() != 1)
    return Error("Multiple results from fold");

  if (auto foldResultAttr = dyn_cast<Attribute>(foldResults.front()))
    if (auto foldResultTypedAttr = dyn_cast<TypedAttr>(foldResultAttr))
      return foldResultTypedAttr;

  return Error("Fold result isn't a typed attribute");
}

// Canonicalize `op` based on its folder; that is, apply `tryFoldOp` to `op` and
// convert the result to a `ParamToValueOp` if possible.
static LogicalResult genericCanonicalize(Operation *op,
                                         PatternRewriter &rewriter) {
  ASSERT_STREAM(isa<MOSHDialect>(op->getDialect()), << "expected MOSH op");
  Location loc = op->getLoc();

  // Apply op's folder
  ErrorOr<TypedAttr> resultAttrOr = tryFoldOp(op);
  if (resultAttrOr.isError())
    return rewriter.notifyMatchFailure(loc, resultAttrOr.getError());

  TypedAttr resultAttr = resultAttrOr.get();
  // For the folder's result, we support converting these types of attributes
  // into a `ParamToValueOp`.
  if (isa<KGEN::ParamDeclRefAttr, KGEN::ParamOperatorAttr, IntegerAttr,
          ShapeAttr>(resultAttr)) {
    rewriter.replaceOpWithNewOp<ParamToValueOp>(op, resultAttr);
    return success();
  }

  return rewriter.notifyMatchFailure(
      loc, "fold result attribute type isn't supported");
}

LogicalResult BroadcastOp::canonicalize(BroadcastOp op,
                                        PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult EqualOp::canonicalize(EqualOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult GetDimOp::canonicalize(GetDimOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult GetRankOp::canonicalize(GetRankOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult NewOp::canonicalize(NewOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult NumElementsOp::canonicalize(NumElementsOp op,
                                          PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

LogicalResult SliceOp::canonicalize(SliceOp op, PatternRewriter &rewriter) {
  return genericCanonicalize(op, rewriter);
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  auto xShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputX());
  auto yShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputY());
  if (xShape && yShape) {
    ErrorOr<ShapeAttr> shapeOr = ShapeAttr::broadcastShapes(xShape, yShape);
    if (!shapeOr.isError())
      return *shapeOr;
  }
  return {};
}

void ConcatOp::build(OpBuilder &builder, OperationState &state,
                     TypedValue<ShapeType> input1,
                     TypedValue<ShapeType> input2) {
  return build(builder, state, {input1, input2});
}

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  // Single shape parameter or dimension parameters.
  if (adaptor.getInputs().size() == 1)
    return adaptor.getInputs().front();

  // Collect dimension parameters from all inputs
  SmallVector<TypedAttr> finalDims;
  for (Attribute input : adaptor.getInputs()) {
    if (auto inputDims = dyn_cast_or_null<ShapeAttr>(input))
      llvm::append_range(finalDims, inputDims.getValues());
    else
      return {};
  }
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(finalDims, shapeTy);
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  if (genericCanonicalize(op, rewriter).succeeded())
    return success();

  auto inputs = op.getInputs();
  if (inputs.size() == 1) {
    rewriter.replaceAllUsesWith(op, inputs[0]);
    return success();
  }

  SmallVector<Value> newInputs;
  for (Value arg : inputs) {
    if (auto precedingOp = arg.getDefiningOp<ConcatOp>()) {
      auto preInputs = precedingOp.getInputs();
      newInputs.append(preInputs.begin(), preInputs.end());
    } else {
      newInputs.emplace_back(arg);
    }
  }

  auto isEqual = [](auto ins) { return std::get<0>(ins) == std::get<1>(ins); };
  if (llvm::all_of(llvm::zip(inputs, newInputs), isEqual))
    return failure();

  op.getInputsMutable().assign(newInputs);
  return success();
}

TypedAttr EqualOp::asExpression(ShapeAttr shape1, ShapeAttr shape2) {
  ASSERT_STREAM(shape1.getRank() == shape2.getRank(),
                << "Shapes must have same rank but they are "
                << shape1.getRank() << " and " << shape2.getRank());
  SmallVector<TypedAttr> andOperands;
  for (auto [xDim, yDim] : llvm::zip(shape1.getValues(), shape2.getValues())) {
    andOperands.emplace_back(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {xDim, yDim}));
  }
  return KGEN::ParamOperatorAttr::get(KGEN::POC::And, andOperands);
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  auto input1 = adaptor.getInput1();
  auto input2 = adaptor.getInput2();
  if (!input1 || !input2)
    return {};

  if (input1 == input2) // fast path
    return BoolAttr::get(getContext(), true);

  auto shape1 = dyn_cast_or_null<ShapeAttr>(input1);
  auto shape2 = dyn_cast_or_null<ShapeAttr>(input2);
  if (!shape1 || !shape2)
    return {};

  // If the inputs are different rank, we can fold.
  if (shape1.getRank() != shape2.getRank())
    return BoolAttr::get(getContext(), false);

  // If both inputs have the same static rank, we fold into an expression.
  return asExpression(shape1, shape2);
}

TypedAttr GetDimOp::emitStaticCall(TypedAttr inputShape, int64_t axis) {
  auto indexTy = IndexType::get(inputShape.getContext());
  return KGEN::emitMLIROperationCall(
      GetDimOp::getOperationName(), {},
      {inputShape, IntegerAttr::get(indexTy, axis)}, indexTy);
}

OpFoldResult GetDimOp::fold(FoldAdaptor adaptor) {
  auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput());
  auto dim = dyn_cast_or_null<IntegerAttr>(adaptor.getDim());
  if (input && dim) {
    auto normalizedAxis =
        normalizeAxis(dim.getValue().getSExtValue(), input.getRank());
    ArrayRef<TypedAttr> values = input.getValues();
    ASSERT_STREAM((normalizedAxis >= 0) &&
                      (normalizedAxis < static_cast<int64_t>(values.size())),
                  "Normalized axis must be in range [0, "
                      << values.size() << "] but it is " << normalizedAxis);
    return values[normalizedAxis];
  }
  return {};
}

OpFoldResult GetRankOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    auto indexTy = IndexType::get(getContext());
    return IntegerAttr::get(indexTy, input.getRank());
  }
  return {};
}

OpFoldResult NewOp::fold(FoldAdaptor adaptor) {
  SmallVector<TypedAttr> newDims;
  for (Attribute dimAttr : adaptor.getDims()) {
    if (auto dim = dyn_cast_or_null<TypedAttr>(dimAttr))
      newDims.emplace_back(dim);
    else
      return {};
  }
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(newDims, shapeTy);
}

OpFoldResult NumElementsOp::fold(FoldAdaptor adaptor) {
  if (auto inputDimAttrs = dyn_cast_or_null<ShapeAttr>(adaptor.getInput()))
    return inputDimAttrs.getSymbolicNumElements();
  return {};
}

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = dyn_cast_or_null<ShapeAttr>(adaptor.getInput());
  auto startAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getStart());
  auto endAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getEnd());
  if (!inputAttr || !startAttr || !endAttr)
    return {};

  // Clamp `start` and `end` to the range [0, rank], since based on slice
  // semantics these arguments may fall outside the rank of the `ShapeAttr`.
  // TODO: is this really needed? Maybe we should just enforce this in IR.
  int64_t rank = inputAttr.getRank();
  int64_t end = clampAxis(endAttr.getValue().getSExtValue(), rank);
  int64_t start = clampAxis(startAttr.getValue().getSExtValue(), rank);

  // Ensure that the indices we use to create slicedDims are valid.
  if (start > end) {
    ASSERT_STREAM(false, << "start index (" << start
                         << ") must be less than end index (" << end << ")");
    return {};
  }

  auto it = inputAttr.getValues().begin();
  SmallVector<TypedAttr> slicedDims(it + start, it + end);
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(slicedDims, shapeTy);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

using KGEN::parseIndexParamValue;
using KGEN::parseStringParam;
using KGEN::printIndexParamValue;
using KGEN::printStringParam;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

void MOSHDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MOSHDialect/MOSH.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MOSHDialect/MOSH.cpp.inc"
