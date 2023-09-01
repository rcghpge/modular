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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexAttrs.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace MOSH;

namespace Index = mlir::index;
namespace Arith = mlir::arith;

//===----------------------------------------------------------------------===//
// Shape ops
//===----------------------------------------------------------------------===//

/// Normalize axis by incrementing negative axis with given rank. For
/// instance, for a rank-3 tensor, the `-1` axis is the last axis, i.e., axis
/// `2`, which is `-1 + 3`.
static int64_t normalizeAxis(const APInt &axisApInt, int64_t rank) {
  int64_t axis = axisApInt.getSExtValue();
  return (axis < 0) ? rank + axis : axis;
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
  assert(shape1.getRank() == shape2.getRank());
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
  if (input && dim)
    return input.getValues()[normalizeAxis(dim.getValue(), input.getRank())];
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

/// Clamps `axisApInt` to the range [0, rank].
static int64_t clampAxis(const APInt &axisApInt, int64_t rank) {
  int64_t axis = normalizeAxis(axisApInt, rank);
  return std::max(std::min(axis, rank), (int64_t)0);
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
  int64_t end = clampAxis(endAttr.getValue(), rank);
  int64_t start = clampAxis(startAttr.getValue(), rank);

  auto it = inputAttr.getValues().begin();
  SmallVector<TypedAttr> slicedDims(it + start, it + end);
  auto shapeTy = ShapeType::get(getContext());
  return ShapeAttr::get(slicedDims, shapeTy);
}

//===----------------------------------------------------------------------===//
// Shape::BroadcastOp
//===----------------------------------------------------------------------===//

namespace ShapeBcastImpl {

static KGEN::FuncOp buildForwardShapeFunc(Location loc, StringRef fwFuncName,
                                          size_t smallerRank,
                                          mlir::RewriterBase &rewriter) {
  KGEN::FuncOp funcOp = createNewShapeFunc(fwFuncName, 2, loc, rewriter);

  Block *entryBlock = funcOp.addEntryBlock();
  rewriter.setInsertionPointToEnd(entryBlock);
  ShapeValue xShape = getBlockArg(entryBlock, 0);
  ShapeValue yShape = getBlockArg(entryBlock, 1);

  SmallVector<Value> newDims;
  auto constOne = rewriter.create<Index::ConstantOp>(loc, 1);
  for (int64_t dim = -smallerRank; dim < 0; ++dim) {
    auto dimValue = rewriter.create<Index::ConstantOp>(loc, dim);
    auto xDim = rewriter.create<GetDimOp>(loc, xShape, dimValue);
    auto yDim = rewriter.create<GetDimOp>(loc, yShape, dimValue);

    auto xDimIsOne = rewriter.create<Index::CmpOp>(
        loc, Index::IndexCmpPredicate::EQ, xDim, constOne);
    auto yDimIsOne = rewriter.create<Index::CmpOp>(
        loc, Index::IndexCmpPredicate::EQ, yDim, constOne);
    auto xOrYIsOne = rewriter.create<Arith::OrIOp>(loc, xDimIsOne, yDimIsOne);
    auto xDimIsYDim = rewriter.create<Index::CmpOp>(
        loc, Index::IndexCmpPredicate::EQ, xDim, yDim);
    auto bcastCond = rewriter.create<Arith::OrIOp>(loc, xOrYIsOne, xDimIsYDim);
    rewriter.create<AssertOp>(loc, bcastCond,
                              "dimensions are not broadcastable");

    auto resDim = rewriter.create<Index::MaxSOp>(loc, xDim, yDim);
    newDims.emplace_back(resDim);
  }

  // One of these shapes is empty because we know the smaller rank.
  auto constZero = rewriter.create<Index::ConstantOp>(loc, 0);
  auto constSmallerRank = rewriter.create<Index::ConstantOp>(loc, smallerRank);
  auto constNegOne = rewriter.create<Index::ConstantOp>(loc, -1);
  auto negSmallerRank =
      rewriter.create<Index::MulOp>(loc, constSmallerRank, constNegOne);

  auto xBatch =
      rewriter.create<SliceOp>(loc, xShape, constZero, negSmallerRank);
  auto yBatch =
      rewriter.create<SliceOp>(loc, yShape, constZero, negSmallerRank);
  auto batchDims = rewriter.create<ConcatOp>(loc, xBatch, yBatch);
  auto bcastedDims = rewriter.create<NewOp>(loc, newDims);
  auto res = rewriter.create<ConcatOp>(loc, batchDims, bcastedDims);

  rewriter.create<KGEN::ReturnOp>(loc, SmallVector<Value>{res});

  return funcOp;
}

} // namespace ShapeBcastImpl

KGEN::FuncOp BroadcastOp::getForwardShapeFunc(Location loc, ModuleOp module,
                                              size_t smallerRank,
                                              mlir::RewriterBase &rewriter) {
  using namespace ShapeBcastImpl;

  std::string fwShapeFuncName =
      ("broadcast_" + Twine(smallerRank) + "_fw").str();
  if (auto funcOp = module.lookupSymbol<KGEN::FuncOp>(fwShapeFuncName))
    return funcOp;

  auto insPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  KGEN::FuncOp funcOp =
      buildForwardShapeFunc(loc, fwShapeFuncName, smallerRank, rewriter);
  rewriter.setInsertionPoint(insPoint.getBlock(), insPoint.getPoint());
  return funcOp;
}

//===----------------------------------------------------------------------===//
// MO_AssertOp
//===----------------------------------------------------------------------===//

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond,
                     Twine message) {
  auto attr = StringAttr::get(message.str(),
                              KGEN::StringType::get(builder.getContext()));
  build(builder, state, cond, cast<TypedAttr>(attr));
}

TypedAttr AssertOp::emitStaticCall(TypedAttr cond, StringRef message) {
  MLIRContext *ctx = cond.getContext();
  SmallVector<std::pair<StringAttr (*)(mlir::OperationName), Attribute>>
      msgAttrNameFuncAndArg = {
          {AssertOp::getMessageAttrName,
           StringAttr::get(message, KGEN::StringType::get(ctx))}};
  auto chainTy = ChainType::get(ctx);
  return KGEN::emitMLIROperationCall(AssertOp::getOperationName(),
                                     msgAttrNameFuncAndArg, {cond}, chainTy);
}

OpFoldResult AssertOp::fold(FoldAdaptor adaptor) {
  // If the condition is true, we return a chain attribute, otherwise we leave
  // the op as is and a pass can either check for failing asserts or it becomes
  // a guaranteed runtime failure.
  if (auto boolAttr = dyn_cast_or_null<BoolAttr>(adaptor.getCond()))
    if (boolAttr.getValue())
      return ChainAttr::get(ChainType::get(getContext()));
  return {};
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
