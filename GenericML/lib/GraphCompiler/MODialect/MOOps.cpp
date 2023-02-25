//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MO dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MODialect/MOInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// MO_GraphOp support
//===----------------------------------------------------------------------===//

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    TypeRange inputTypes, TypeRange resultTypes) {
  // Below we basically inline and specialize `buildWithEntryBlock`, which
  // doesn't work because we use `signature` and not `function_type`.
  state.addAttribute(M::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  // Add signature attribute
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::SignatureType::get(funcType);
  state.addAttribute(GraphOp::getSignatureAttrName(state.name),
                     mlir::TypeAttr::get(sig));

  // Add the function body.
  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

/// Parses an MO Graph.
///
/// operation ::= `graph` function-signature function-attributes? function-body
///
ParseResult GraphOp::parse(OpAsmParser &parser, OperationState &result) {
  return KGEN::parseGeneratorOrFunc(parser, result,
                                    KGEN::GeneratorOrFuncKind::func);
}

/// Print the GraphOp.
void GraphOp::print(OpAsmPrinter &p) { KGEN::printGeneratorOrFunc(p, *this); }

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

//===----------------------------------------------------------------------===//
// OutputOp support
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  // ODS checks the parent is a GraphOp.
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  if (graph.getResultParams().size() != this->getParameters().size())
    return emitError(
        "output bindings and result parameters must have the same size");

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> ResParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] :
       llvm::zip(graph.getResultParams(), this->getParameters())) {
    ResParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      TypedAttr newShapeAttr = tensorTy.getShapeAttr().replace(
          [&ResParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = ResParamRefToBindingAttr.lookup(paramRef);
            return resolvedAttr ? resolvedAttr : paramRef;
          });
      expectedOutputOperandTypes.emplace_back(tensorTy.cloneWith(newShapeAttr));
    } else {
      expectedOutputOperandTypes.emplace_back(ty);
    }
  }

  return checkResultTypes(getOperation(), expectedOutputOperandTypes);
}

//===----------------------------------------------------------------------===//
// MO_ConstantOp support
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int64_t> values, Type elementType) {
  auto type = MO::TensorType::get({(int64_t)values.size()}, elementType);
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      values, mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, valueAttr);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APFloat> values, Type elementType) {
  auto type = MO::TensorType::get({(int64_t)values.size()}, elementType);
  auto valueAttr = FloatArrayElementsAttr::get(values, type.getElementType());
  build(odsBuilder, odsState, type, valueAttr);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, Type elementType) {
  auto type = MO::TensorType::get(SmallVector<int64_t>{}, elementType);
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(SmallVector<int64_t>{value}),
      mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, valueAttr);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APFloat value, Type elementType) {
  auto type = MO::TensorType::get(SmallVector<int64_t>{}, elementType);
  auto valueAttr =
      FloatArrayElementsAttr::get({std::move(value)}, type.getElementType());
  build(odsBuilder, odsState, type, valueAttr);
}

//===----------------------------------------------------------------------===//
// MO_ConcatOp support
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs < 2)
    return emitError("concat must have at least 2 input tensors");

  SmallVector<TypedValue<TensorType>> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TypedValue<TensorType>>(input));

  tensors.emplace_back(getResult());
  if (failed(impl::verifyMatchingRanks(tensors)))
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

LogicalResult MaxPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks that the input and result have
  // matching ranks, so we can be more specific with the error message here
  if (failed(impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes({getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  TensorType aType = getTensorA().getType();

  // Check we have 2D * 2D tensors as input.
  if (aType.getRank() != 2) {
    return emitError("matmul input tensors must be rank 2, got: ")
           << aType.getRank();
  }

  return LogicalResult::success();
}

LogicalResult BatchMatmulOp::verify() {
  // TODO: improve this verifier
  TensorType aType = getTensorA().getType();

  // Check we have at least 3d tensors
  if (aType.hasRank() && aType.getRank() < 3) {
    return emitError(
               "batch_matmul input tensors must be at least rank 3, got: ")
           << aType.getRank();
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// MO_RebindOp support
//===----------------------------------------------------------------------===//

static ParseResult parseParamDecls(OpAsmParser &p,
                                   KGEN::ParamDeclArrayAttr &paramDecls) {
  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (p.parseLParen() || p.parseRParen() || p.parseArrow() ||
        KGEN::parseParamDecls(p, paramDecls) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

static void printParamDecls(OpAsmPrinter &p, Operation *,
                            KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<() -> ";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}

OpFoldResult RebindOp::fold(FoldAdaptor adaptor) {
  // We fold if the result and operand types are the same.
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

LogicalResult RebindOp::verify() {
  TensorType inTy = getInput().getType();
  TensorType outTy = getType();
  // Nothing to check if either type is dynamically ranked
  if (!inTy.hasRank() || !outTy.hasRank())
    return success();

  if (inTy.getRank() != outTy.getRank())
    return emitError("input and output static ranks must match");

  SmallDenseMap<KGEN::ParamDeclRefAttr, IntegerAttr> dimParamToInt;
  auto tryUnifyParamToIntDim = [&](TypedAttr srcDimAttr,
                                   TypedAttr dstDimAttr) -> LogicalResult {
    auto srcDimParamRef = dyn_cast<KGEN::ParamDeclRefAttr>(srcDimAttr);
    auto dstDimIntAttr = dyn_cast<IntegerAttr>(dstDimAttr);
    if (srcDimParamRef && dstDimIntAttr) {
      auto [cachedEntry, _] =
          dimParamToInt.try_emplace(srcDimParamRef, dstDimIntAttr);
      IntegerAttr cachedDimIntAttr = cachedEntry->second;
      if (cachedDimIntAttr != dstDimIntAttr)
        return failure();
    }
    return success();
  };
  auto inDimAttrs = dyn_cast<ShapeAttr>(inTy.getShapeAttr());
  auto outDimAttrs = dyn_cast<ShapeAttr>(outTy.getShapeAttr());
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimIntAttr = dyn_cast<IntegerAttr>(inDimAttr);
      auto outDimIntAttr = dyn_cast<IntegerAttr>(outDimAttr);
      if (inDimIntAttr && outDimIntAttr && inDimIntAttr != outDimIntAttr)
        return emitError("statically known dimensions must match");
      else if (failed(tryUnifyParamToIntDim(inDimAttr, outDimAttr)) ||
               failed(tryUnifyParamToIntDim(outDimAttr, inDimAttr)))
        return emitError("one dimension parameter cannot bind to different "
                         "static dimensions");
    }
  }
  return success();
}

void RebindOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  if (!getParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

//===----------------------------------------------------------------------===//
// MO_ShapeOfOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeOfOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();
  if (resTy.hasStaticShape()) {
    if (!inputTy.hasRank())
      return emitError("input must have static rank if result has known size");
    if ((int64_t)inputTy.getRank() != resTy.getIntDim(0))
      return emitError("input rank and output size must match");
    return success();
  }

  if (inputTy.hasRank())
    return emitError("input must have dynamic rank if result has unknown size");
  return success();
}

//===----------------------------------------------------------------------===//
// MO_TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  // The permutation has known shape iff the argument and result are statically
  // ranked.
  TensorType permTy = getPerm().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!permTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if perm "
                       "has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if perm "
                     "has known shape");
  }

  if (inputTy.getRank() != resTy.getRank())
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)resTy.getRank() != permTy.getIntDim(0))
    return emitError("input and result ranks do not match perm size");
  return success();
}

//===----------------------------------------------------------------------===//
// Shape ops
//===----------------------------------------------------------------------===//

/// Normalize axis by incrementing negative axis with given rank. For instance,
/// for a rank-3 tensor, the `-1` axis is the last axis, i.e., axis `2`, which
/// is `-1 + 3`.
static int64_t normalizeAxis(const APInt &axisApInt, int64_t rank) {
  int64_t axis = axisApInt.getSExtValue();
  return (axis < 0) ? rank + axis : axis;
}

void Shape::AssertOp::build(OpBuilder &builder, OperationState &state,
                            Value cond) {
  build(builder, state, cond, StringAttr());
}

void Shape::AssertOp::build(OpBuilder &builder, OperationState &state,
                            Value cond, StringRef message) {
  build(builder, state, cond, builder.getStringAttr(message));
}

OpFoldResult Shape::ConcatOp::fold(FoldAdaptor adaptor) {
  auto input1 = dyn_cast_or_null<ShapeAttr>(adaptor.getInput1());
  auto input2 = dyn_cast_or_null<ShapeAttr>(adaptor.getInput2());
  if (input1 && input2) {
    SmallVector<TypedAttr> finalDims;
    llvm::append_range(finalDims, input1.getValues());
    llvm::append_range(finalDims, input2.getValues());
    auto shapeTy = ShapeType::get(getContext());
    return ShapeAttr::get(finalDims, shapeTy);
  }
  return {};
}

OpFoldResult Shape::EqualOp::fold(FoldAdaptor adaptor) {
  auto input1 = adaptor.getInput1();
  auto input2 = adaptor.getInput2();
  if (!input1 || !input2)
    return {};

  if (input1 == input2) // fast path
    return BoolAttr::get(getContext(), true);

  auto shape1 = dyn_cast_or_null<ShapeAttr>(input1);
  auto shape2 = dyn_cast_or_null<ShapeAttr>(input2);
  if (shape1 && shape2) {
    if (shape1.getRank() != shape2.getRank())
      return BoolAttr::get(getContext(), false);

    // we can only claim false here if inputs are different integers
    for (auto [dim1, dim2] : llvm::zip(shape1.getValues(), shape2.getValues()))
      if (dim1.isa<IntegerAttr>() && dim2.isa<IntegerAttr>() && dim1 != dim2)
        return BoolAttr::get(getContext(), false);
  }
  return {};
}

OpFoldResult Shape::FromTensorOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ElementsAttr>(adaptor.getInput())) {
    auto newDims = llvm::to_vector(
        llvm::map_range(input.getValues<IntegerAttr>(),
                        [](IntegerAttr dim) { return dim.getSInt(); }));
    auto shapeTy = ShapeType::get(getContext());
    return ShapeAttr::get(newDims, shapeTy);
  }
  return {};
}

void Shape::GetDimOp::build(OpBuilder &builder, OperationState &state,
                            Value input, int64_t dim) {
  build(builder, state, input, builder.getIndexAttr(dim));
}

OpFoldResult Shape::GetDimOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    int64_t dim = normalizeAxis(adaptor.getDim(), input.getRank());
    return input.getValues()[dim];
  }
  return {};
}

OpFoldResult Shape::GetRankOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    auto indexTy = IndexType::get(getContext());
    return IntegerAttr::get(indexTy, input.getRank());
  }
  return {};
}

OpFoldResult Shape::NewOp::fold(FoldAdaptor adaptor) {
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

void Shape::SliceOp::build(OpBuilder &builder, OperationState &state,
                           Value input, int64_t start, int64_t end) {
  build(builder, state, input, builder.getIndexAttr(start),
        builder.getIndexAttr(end));
}

void Shape::SliceOp::build(OpBuilder &builder, OperationState &state,
                           Value input, int64_t start) {
  build(builder, state, input, builder.getIndexAttr(start), {});
}

OpFoldResult Shape::SliceOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    int64_t start = normalizeAxis(adaptor.getStart(), input.getRank());
    int64_t end = input.getRank();
    if (adaptor.getEnd().has_value())
      end = normalizeAxis(adaptor.getEnd().value(), input.getRank());
    SmallVector<TypedAttr> slicedDims(input.getValues().begin() + start,
                                      input.getValues().begin() + end);
    auto shapeTy = ShapeType::get(getContext());
    return ShapeAttr::get(slicedDims, shapeTy);
  }
  return {};
}

LogicalResult Shape::SliceOp::verify() {
  // NOTE we can't check for input rank here, so verification is limited
  IntegerAttr startAttr = getStartAttr();
  IntegerAttr endAttr = getEndAttr();
  if (!endAttr)
    return success();

  int64_t start = startAttr.getInt();
  int64_t end = endAttr.getInt();
  auto bothOrNeitherIndicesNegative =
      (start < 0 && end < 0) || (start >= 0 && end >= 0);
  if (bothOrNeitherIndicesNegative && end < start) {
    return emitError("end axis cannot be smaller than start axis if they are "
                     "both or neither negative");
  }
  return success();
}

OpFoldResult Shape::ToTensorOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    if (input.isConstant()) {
      return M::IntArrayElementsAttr::get(
          getContext(), ArrayRef<int64_t>{input.getIntDims()},
          IntegerType::SignednessSemantics::Signed);
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Needed because the type and attribute constraints in the generated file are
// not placed in the correct namespace by tablegen.
using namespace Shape;

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
