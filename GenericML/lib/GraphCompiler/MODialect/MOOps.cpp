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
#include "GenericML/GraphCompiler/MODialect/Broadcast.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/RangeUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <numeric>

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// If `val` is defined by a ConstantOp or a RebindOp that takes in a
/// ConstantOp with given rank, return the ConstantOp, else return null.
static ConstantOp tryGetConstantOpOfRank(TensorValue val, size_t rank) {
  Operation *inputOp = val.getDefiningOp();
  if (auto rebindOp = dyn_cast_or_null<RebindOp>(inputOp))
    inputOp = rebindOp.getInput().getDefiningOp();

  auto constantOp = dyn_cast_or_null<ConstantOp>(inputOp);
  if (!constantOp)
    return {};

  MO::TensorType type = constantOp.getType();
  if (!type.hasRank() || type.getRank() != rank)
    return {};

  return constantOp;
}

/// Convert the data held by `attr` into a vector of int64_t.
template <typename T>
static SmallVector<int64_t>
denseResourceAttrToIntVec(DenseResourceElementsAttr attr) {
  ArrayRef<T> data = attr.getRawHandle().getBlob()->getDataAs<T>();
  return SmallVector<int64_t>(data.begin(), data.end());
}

/// Return the contents of the IntArrayElementsAttr as a vector in int64_t.
static SmallVector<int64_t> getBlob(IntArrayElementsAttr intElemsAttr) {
  return map_to_vector(intElemsAttr.getValues(), [](const llvm::APInt &dim) {
    return dim.getSExtValue();
  });
}

/// If `attr` represents a blob of constant integers, return them in a vector,
/// else return None.
static std::optional<SmallVector<int64_t>>
tryGetConstantIntBlob(mlir::ElementsAttr attr, DType dType) {
  if (auto intElemsAttr = dyn_cast<IntArrayElementsAttr>(attr))
    return getBlob(intElemsAttr);

  if (auto axisValueAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
    return dType.dispatch<std::optional<SmallVector<int64_t>>>()
        .when<M::DType::si32>(
            [&]() { return denseResourceAttrToIntVec<int32_t>(axisValueAttr); })
        .when<M::DType::si64>(
            [&]() { return denseResourceAttrToIntVec<int64_t>(axisValueAttr); })
        .otherwise([&]() { return std::nullopt; });
  }

  return std::nullopt;
}

/// If `val` is defined by a ConstantOp or a RebindOp that takes in a
/// ConstantOp with given rank, convert it to an integer vector, else return
/// None.
static std::optional<SmallVector<int64_t>>
tryGetConstantIntOfRank(TensorValue val, size_t rank) {
  if (ConstantOp constantOp = tryGetConstantOpOfRank(val, rank)) {
    return tryGetConstantIntBlob(constantOp.getValueAttr(),
                                 constantOp.getType().getDtype());
  }
  return std::nullopt;
}

std::optional<int64_t> M::MO::tryGetConstantInt(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantIntOfRank(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
M::MO::tryGetConstantIntVector(TensorValue val) {
  return tryGetConstantIntOfRank(val, 1);
}

//===----------------------------------------------------------------------===//
// MO_AssertOp
//===----------------------------------------------------------------------===//

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond) {
  build(builder, state, cond, StringAttr());
}

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond,
                     StringRef message) {
  build(builder, state, cond, builder.getStringAttr(message));
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
// MO_IndexToTensorOp support
//===----------------------------------------------------------------------===//

OpFoldResult IndexToTensorOp::fold(FoldAdaptor adaptor) {
  auto inputAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getInput());
  if (!inputAttr)
    return {};

  SmallVector<APInt> res{inputAttr.getValue()};
  auto rankedResTy = RankedTensorType::get({}, getType().getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

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
// MO_OutputOp support
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
// IfOp support
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  // Verify that the return types match the block arguments.
  // The ODS verifier already checked the cond to be present and bool.
  auto *thenBlock = &getThenRegion().front();
  auto *elseBlock = &getElseRegion().front();

  auto thenOutput = cast<YieldOp>(thenBlock->getTerminator());
  if (failed(checkResultTypes(thenOutput, getResultTypes())))
    return failure();

  auto elseOutput = cast<YieldOp>(elseBlock->getTerminator());
  return checkResultTypes(elseOutput, getResultTypes());
}

//===----------------------------------------------------------------------===//
// MO_BroadcastShapeOp support
//===----------------------------------------------------------------------===//

LogicalResult BroadcastShapeOp::verify() {
  TensorType inputXType = getInputX().getType();
  TensorType inputYType = getInputY().getType();
  TensorType resultType = getResult().getType();
  // For dynamically ranked/shaped result, we defer to shape inference
  if (!resultType.hasStaticShape())
    return success();

  bool inputXIsStatic = inputXType.hasStaticShape();
  bool inputYIsStatic = inputYType.hasStaticShape();
  // Note that inputs and output are rank-1 Shape tensors
  int64_t inputXRank = inputXIsStatic ? inputXType.getIntDim(0) : 0;
  int64_t inputYRank = inputYIsStatic ? inputYType.getIntDim(0) : 0;
  int64_t resultRank = resultType.getIntDim(0);

  // Check first input
  if (inputXIsStatic && inputXRank > resultRank) {
    return emitError("result rank must be no less than first input rank if "
                     "both are statically ranked");
  }

  // Check second input
  if (inputYIsStatic && inputYRank > resultRank) {
    return emitError("result rank must be no less than second input rank if "
                     "both are statically ranked");
  }

  // Check both inputs together
  if (inputXIsStatic && inputYIsStatic &&
      resultRank != std::max(inputXRank, inputYRank)) {
    return emitError("output rank must be the max between 2 input ranks, if "
                     "all are statically ranked");
  }

  return success();
}

OpFoldResult BroadcastShapeOp::fold(FoldAdaptor adaptor) {
  auto xIntDims = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputX());
  auto yIntDims = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputY());
  if (xIntDims && yIntDims) {
    // This is an inefficient way of doing this, but since this op would be
    // usually lowered to `mo.shape.broadcast`, it should be folded rarely.
    auto shapeTy = ShapeType::get(getContext());
    auto xShape = ShapeAttr::get(xIntDims, shapeTy);
    auto yShape = ShapeAttr::get(yIntDims, shapeTy);
    ErrorOr<MO::ShapeAttr> shapeOr = broadcastShape(xShape, yShape);
    if (!shapeOr.isError()) {
      // Ensure we return with the appropriate element type.
      auto getResAttrs = [&](auto values) {
        return IntArrayElementsAttr::get(
            getContext(), values, IntegerType::SignednessSemantics::Signed);
      };
      if (getType().getElementType().getIntOrFloatBitWidth() == 32) {
        SmallVector<int32_t> dims = map_to_vector(
            shapeOr->getIntDims(), [](int64_t dim) -> int32_t { return dim; });
        return getResAttrs(ArrayRef<int32_t>{dims});
      }
      return getResAttrs(ArrayRef<int64_t>{shapeOr->getIntDims()});
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// MO_AddOp support
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputY());
  if (!xAttr || !yAttr)
    return {};

  TensorType resTy = getType();
  DType resDType = resTy.getDtype();
  auto maybeX = tryGetConstantIntBlob(xAttr, resDType);
  auto maybeY = tryGetConstantIntBlob(yAttr, resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(xArr.size());
  for (size_t idx = 0, e = xArr.size(); idx < e; idx++)
    res[idx] = APInt(64, xArr[idx] + yArr[idx], /*isSigned=*/true);

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

//===----------------------------------------------------------------------===//
// MO_BroadcastToOp
//===----------------------------------------------------------------------===//

/// Check if the input type can be broadcasted to the result type, assumin both
/// have static ranks.
static ErrorOrSuccess verifyBroadcastStaticRanks(TensorType inputTy,
                                                 TensorType resTy) {
  if (inputTy.hasRank()) {
    if (inputTy.getRank() > resTy.getRank())
      return Error("result rank must be at least the input rank");

    ArrayRef<TypedAttr> fromDims = inputTy.getTypedShapeAttr().getValues();
    ArrayRef<TypedAttr> toDims = resTy.getTypedShapeAttr().getValues();
    for (auto [fromDim, toDim] :
         llvm::zip(llvm::reverse(fromDims), llvm::reverse(toDims))) {
      auto fromIntDim = dyn_cast<IntegerAttr>(fromDim);
      auto toIntDim = dyn_cast<IntegerAttr>(toDim);
      if (fromIntDim && toIntDim) {
        // We can only assert anything if both dimensions are known constants.
        int64_t inDim = fromIntDim.getInt();
        if (inDim > 1 && (inDim != toIntDim.getInt()))
          return Error("input shape not broadcastable to result shape");
      }
    }
  }
  return success();
}

LogicalResult BroadcastToOp::verify() {
  TensorType resTy = getType();
  TensorType newShapeTy = getNewShape().getType();

  if (!newShapeTy.hasStaticShape()) {
    if (!resTy.hasRank())
      return success();
    return emitError(
        "result must be dynamically ranked if new shape has unknown size");
  }

  if (!resTy.hasRank()) {
    return emitError(
        "result must be statically ranked if new shape has static size");
  }

  if ((int64_t)resTy.getRank() != newShapeTy.getIntDim(0))
    return emitError("new shape size and output rank must match");

  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), resTy))
    return emitError(err.getError());

  return success();
}

//===----------------------------------------------------------------------===//
// MOStatic_BroadcastToOp
//===----------------------------------------------------------------------===//

OpFoldResult Static::BroadcastToOp::fold(FoldAdaptor adaptor) {
  if (getInput().getType() == getType())
    return getInput();
  return {};
}

LogicalResult Static::BroadcastToOp::verify() {
  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), getType()))
    return emitError(err.getError());
  return success();
}

//===----------------------------------------------------------------------===//
// MO_ConstantOp support
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::canonicalize(ConstantOp op,
                                       PatternRewriter &rewriter) {
  if (op.getType().hasUnknownShape()) {
    Location loc = op.getLoc();
    ElementsAttr elements = op.getValue();
    auto staticTensorType = TensorType::get(elements.getType().getShape(),
                                            elements.getElementType());
    auto constantOp =
        rewriter.create<ConstantOp>(loc, staticTensorType, elements);
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), constantOp);
  }
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");
  return getValueAttr();
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int64_t> values) {
  auto type =
      MO::TensorType::get({(int64_t)values.size()},
                          odsBuilder.getIntegerType(64, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      values, mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, valueAttr);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int32_t> values) {
  auto type =
      MO::TensorType::get({(int64_t)values.size()},
                          odsBuilder.getIntegerType(32, /*isSigned=*/true));
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
                       int64_t value) {
  auto type = MO::TensorType::get(
      SmallVector<int64_t>{}, odsBuilder.getIntegerType(64, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(SmallVector<int64_t>{value}),
      mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, valueAttr);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int32_t value) {
  auto type = MO::TensorType::get(
      SmallVector<int64_t>{}, odsBuilder.getIntegerType(32, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int32_t>(SmallVector<int32_t>{value}),
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

  SmallVector<TensorValue> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TensorValue>(input));

  tensors.emplace_back(getResult());
  if (failed(impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// MO_GatherOp support
//===----------------------------------------------------------------------===//

/// Return an empty array with the given element type.
static ArrayElementsAttr getEmptyArray(Type elTy) {
  return ArrayElementsAttr::get({}, RankedTensorType::get({0}, elTy));
}

OpFoldResult GatherOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  if (!getOutputParamDecls().empty())
    return {};

  // The verifier should ensure that indices is empty, but we don't check it.
  if (auto inputElems = dyn_cast_or_null<ElementsAttr>(adaptor.getInput()))
    if (inputElems.empty())
      return getEmptyArray(inputElems.getElementType());

  if (auto idxElems = dyn_cast_or_null<ElementsAttr>(adaptor.getIndices()))
    if (idxElems.empty())
      return getEmptyArray(idxElems.getElementType());

  return {};
}

LogicalResult GatherOp::verify() {
  TensorType paramsType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  // Nothing we can say if either input or indices is dynamically ranked
  if (!paramsType.hasRank() || !indicesType.hasRank())
    return LogicalResult::success();

  if (!resultType.hasRank()) {
    return emitError(
        "result must be ranked if both param and indices are ranked");
  }

  if (paramsType.getRank() - 1 + indicesType.getRank() !=
      resultType.getRank()) {
    return emitError("result rank must be 1 less than the sum of param and "
                     "indices ranks if they are all static");
  }

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// MO_GreaterEqualOp support
//===----------------------------------------------------------------------===//

OpFoldResult GreaterEqualOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  // TODO refactor with other binary op folders

  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputY());
  if (!xAttr || !yAttr)
    return {};

  TensorType resTy = getType();
  DType resDType = resTy.getDtype();
  auto maybeX = tryGetConstantIntBlob(xAttr, resDType);
  auto maybeY = tryGetConstantIntBlob(yAttr, resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(xArr.size());
  for (size_t idx = 0, e = xArr.size(); idx < e; idx++)
    res[idx] = APInt(1, xArr[idx] >= yArr[idx]);

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

//===----------------------------------------------------------------------===//
// MO_CastOp support
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  Attribute inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};

  TensorType resTy = getResult().getType();
  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  if (auto intElemsAttr = dyn_cast_or_null<IntArrayElementsAttr>(inputAttr)) {
    // This is inefficient, but simple.
    SmallVector<APInt> res =
        map_to_vector(intElemsAttr.getValues(), [](const llvm::APInt &dim) {
          return APInt(64, dim.getSExtValue(), /*isSigned=*/true);
        });

    return IntArrayElementsAttr::get(rankedResTy, res);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// MO_SelectOp support
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  // TODO refactor parts of this with other folders

  auto condAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getCond());
  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getY());
  if (!condAttr || !xAttr || !yAttr)
    return {};

  TensorType resTy = getType();
  DType condDType = getCond().getType().getDtype();
  DType resDType = resTy.getDtype();
  auto maybeCond = tryGetConstantIntBlob(condAttr, condDType);
  auto maybeX = tryGetConstantIntBlob(xAttr, resDType);
  auto maybeY = tryGetConstantIntBlob(yAttr, resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> condArr = *maybeCond;
  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(condAttr.size());
  for (size_t idx = 0, e = condAttr.size(); idx < e; idx++) {
    int64_t resVal = condArr[idx] ? xArr[idx] : yArr[idx];
    res[idx] = APInt(64, resVal, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
}

// TODO: Place this in Support somewhere. Maybe already there?
static bool shapesMatch(TensorType x, TensorType y) {
  if (!x.hasRank() || !y.hasRank())
    return true;
  if (x.getRank() != y.getRank())
    return false;
  for (int i = 0, e = x.getRank(); i < e; i++) {
    auto xDim = x.getIntDim(i);
    auto yDim = y.getIntDim(i);
    if (mlir::ShapedType::isDynamic(xDim) || mlir::ShapedType::isDynamic(yDim))
      continue;
    if (xDim != yDim)
      return false;
  }
  return true;
}

LogicalResult SelectOp::verify() {
  TensorType condType = getCond().getType();
  TensorType xType = getX().getType();
  TensorType yType = getY().getType();
  TensorType resultType = getResult().getType();

  if (!shapesMatch(xType, yType))
    emitError("operands must have matching shapes");
  if (!shapesMatch(xType, condType))
    emitError("operands and condition must have matching shapes");
  if (!shapesMatch(xType, resultType))
    emitError("operands and result must have matching shapes");

  DType xDType = xType.getDtype();
  DType yDType = yType.getDtype();
  DType resultDType = getResult().getType().getDtype();

  if (xDType != yDType)
    return emitError("operands must have the same element type");
  if (xDType != resultDType)
    return emitError(
        "operands and result type must have the same element type");

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

LogicalResult WhereOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();

  if (!resTy.hasRank() || resTy.getRank() != 2)
    return emitError("mo.where: invalid output shape: rank must be 2");

  if ((inputTy.hasRank() && resTy.getIntDim(1) != (int64_t)inputTy.getRank()) ||
      (!inputTy.hasRank() && !ShapedType::isDynamic(resTy.getIntDim(1))))
    return emitError("mo.where: invalid output shape: inner_dim != input_rank");

  return LogicalResult::success();
}

//===----------------------------------------------------------------------===//
// Windowed ops
//===----------------------------------------------------------------------===//

LogicalResult AvgPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult MaxPoolOp::verify() {
  auto op = cast<SlidingWindow>(this->getOperation());
  if (failed(op.verifyWindowParamShape(getFilterShape(), "filter_shape")))
    return failure();

  return op.verifyWindowParamSizes(
      {getFilterShape(), getDilations(), getStrides()});
}

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks that the input and result have
  // matching ranks, so we can be more specific with the error message here.
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
// MO_RangeOp support
//===----------------------------------------------------------------------===//

OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  TensorType outTy = getType();
  // Don't fold if there are `?` in the output shape; in such cases,
  // parameterization should run first.
  if (outTy.hasUnknownShape())
    return {};

  auto startAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getStart());
  auto limitAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getLimit());
  auto stepAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getStep());
  if (!startAttr || !limitAttr || !stepAttr)
    return {};

  int64_t start = getBlob(startAttr).front();
  int64_t limit = getBlob(limitAttr).front();
  int64_t step = getBlob(stepAttr).front();

  auto numResElementsOr = getRangeNumElements(start, limit, step);
  if (numResElementsOr.isError())
    return {};
  int64_t numResElements = *numResElementsOr;
  SmallVector<APInt> res((size_t)numResElements);
  for (size_t idx = 0; idx < (size_t)numResElements; ++idx) {
    res[idx] = APInt(64, start, /*isSigned=*/true);
    start += step;
  }

  auto attrTy =
      RankedTensorType::get({numResElements}, startAttr.getElementType());
  auto foldedAttr = IntArrayElementsAttr::get(attrTy, res);
  if (outTy.hasStaticShape())
    return foldedAttr;

  Location loc = getLoc();
  OpBuilder builder(*this); // insert before this op
  auto resTy = TensorType::get({numResElements}, outTy.getElementType());
  auto constantOp = builder.create<ConstantOp>(loc, resTy, foldedAttr);
  auto rebindOp =
      builder.create<RebindOp>(loc, outTy, constantOp, getOutputParamDecls());
  return rebindOp.getResult();
}

LogicalResult RangeOp::verify() {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = getOutputParamDecls();
  if (!paramDecls.empty()) {
    if (paramDecls.size() != 1)
      return emitError("cannot declare more than one parameter");
    if (!paramDecls[0].getType().isIndex())
      return emitError("can only declare index parameters");
  }
  return success();
}

LogicalResult RangeOp::deparameterize(mlir::RewriterBase &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> decls = getOutputParamDecls();
  if (decls.empty())
    return failure();
  setOutputParamDecls({});

  TensorType resTy = getType();
  auto shapeTy = rewriter.getType<ShapeType>();
  TensorType unknownResTy;
  if (resTy.hasRank())
    unknownResTy = resTy.cloneWith(resTy.getIntDims(), resTy.getElementType());
  else
    unknownResTy = resTy.cloneWith(KGEN::UnknownAttr::get(shapeTy));

  auto newOp = rewriter.create<MO::RangeOp>(
      getLoc(), unknownResTy, getOperands(),
      getOperation()->getAttrDictionary().getValue());
  rewriter.replaceOpWithNewOp<RebindOp>(*this, resTy, newOp, decls);

  return success();
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

static LogicalResult materializeShapeInference(RebindOp op,
                                               PatternRewriter &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = op.getOutputParamDecls();
  if (paramDecls.empty())
    return failure();

  // TODO lift into a method on MO::TensorType?
  auto doesShapeAttrContainUnknown = [](TypedAttr shapeAttr) {
    if (auto dimAttrs = dyn_cast<ShapeAttr>(shapeAttr))
      return dimAttrs.hasUnknownDims();
    return isa<KGEN::UnknownAttr>(shapeAttr);
  };

  Location loc = op.getLoc();
  TypedAttr inputShapeAttr = op.getInput().getType().getShapeAttr();
  if (paramDecls.size() == 1) {
    KGEN::ParamDeclAttr newDecl = paramDecls.front();
    if (isa<MO::ShapeType>(newDecl.getType()) &&
        !doesShapeAttrContainUnknown(inputShapeAttr)) {
      // If there is a single declaration in rebind that is a shape parameter,
      // we materialize into a kgen.param.declare.
      rewriter.create<KGEN::ParamDeclareOp>(loc, newDecl, inputShapeAttr);
      op.setOutputParamDecls({});
      return success();
    }
  }

  TypedAttr outputShapeAttr = op.getResult().getType().getShapeAttr();
  auto outputDimAttrs = dyn_cast<ShapeAttr>(outputShapeAttr);
  bool allDeclsAreDimAttrs = llvm::all_of(
      paramDecls, [](auto decl) { return isa<IndexType>(decl.getType()); });
  if (allDeclsAreDimAttrs && outputDimAttrs) {
    llvm::SmallDenseMap<StringAttr, KGEN::ParamDeclAttr> nameToParamDecl;
    for (KGEN::ParamDeclAttr paramDecl : paramDecls)
      nameToParamDecl.try_emplace(paramDecl.getName(), paramDecl);
    if (auto inputShapeParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D1> : Sh0 -> [D0, D1, 44]
      // --->
      //   kgen.param.declare D1 = <apply("mo.shape.get_dim", Sh0, 1)>
      //   rebind : Sh0 -> [D0, D1, 44]
      for (auto [axis, outputDimAttr] :
           llvm::enumerate(outputDimAttrs.getValues())) {
        KGEN::ParamDeclAttr paramDecl;
        auto outputDimParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr);
        if (outputDimParamRef &&
            (paramDecl = nameToParamDecl.lookup(outputDimParamRef.getName()))) {
          TypedAttr mlirOpCallAttr =
              Shape::GetDimOp::emitStaticCall(inputShapeParamRef, axis);
          rewriter.create<KGEN::ParamDeclareOp>(loc, paramDecl, mlirOpCallAttr);
        }
      }
      op.setOutputParamDecls({});
      return success();
    }

    if (auto inputDimAttrs = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D4> : [D0, D1, D2] -> [D3, D4, 42]
      // --->
      //   kgen.param.declare D4 = <D1>
      //   rebind : [D0, D1, D2] -> [D0, D4, 42]
      llvm::DenseSet<KGEN::ParamDeclAttr> unknownParamDecls;
      for (auto [inputDimAttr, outputDimAttr] :
           llvm::zip(inputDimAttrs.getValues(), outputDimAttrs.getValues())) {
        if (auto outputDimParamRef =
                dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr)) {
          if (KGEN::ParamDeclAttr paramDecl =
                  nameToParamDecl.lookup(outputDimParamRef.getName())) {
            if (isa<KGEN::UnknownAttr>(inputDimAttr))
              unknownParamDecls.insert(paramDecl);
            else
              rewriter.create<KGEN::ParamDeclareOp>(loc, paramDecl,
                                                    inputDimAttr);
          }
        }
      }
      if (unknownParamDecls.size() == paramDecls.size())
        return failure();
      op.setOutputParamDecls(SmallVector<KGEN::ParamDeclAttr>(
          unknownParamDecls.begin(), unknownParamDecls.end()));
      return success();
    }
  }

  // TODO: support materialization for other rebinds
  return failure();
}

LogicalResult RebindOp::canonicalize(RebindOp op, PatternRewriter &rewriter) {
  // We fold if the result and operand types are the same.
  if (op.getInput().getType() == op.getType()) {
    assert(op.getOutputParamDecls().empty());
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

  // We materialize the parameter declarations explictly.
  LogicalResult paramsResolved = materializeShapeInference(op, rewriter);

  // Likely already processed, or doesn't help shape inference.
  if (isa_and_nonnull<RebindOp>(op.getInput().getDefiningOp()))
    return success(succeeded(paramsResolved));
  //  %0 = mo.rebind<...>(%arg) // dominator
  //  %1 = mo.rebind<...>(%arg)
  //  %2 = mo.rebind<...>(%arg)
  // ==>
  //  %0 = mo.rebind<...>(%arg) // dominator
  //  %1 = mo.rebind<...>(%0)
  //  %2 = mo.rebind<...>(%0)
  //
  // NOTE This helps shape inference propagate parameter constraints across
  // these rebinds, in case `%arg`'s type is never parameterized and contains
  // unknown dimension `?`.
  SmallVector<RebindOp> rebindsWithSameInput;
  RebindOp dominatorRebindOp = op;
  for (Operation *rebindInputUser : op.getInput().getUsers()) {
    auto rebindOp = dyn_cast<RebindOp>(rebindInputUser);
    // TODO To support this across blocks, we need `DominanceInfo`, which
    // might require us to move this into a pass
    if (rebindOp && rebindOp->getBlock() == dominatorRebindOp->getBlock()) {
      rebindsWithSameInput.push_back(rebindOp);
      if (rebindOp->isBeforeInBlock(dominatorRebindOp))
        dominatorRebindOp = rebindOp;
    }
  }
  if (rebindsWithSameInput.size() > 1) {
    for (RebindOp rebindOp : rebindsWithSameInput)
      if (rebindOp != dominatorRebindOp)
        rebindOp.getInputMutable().assign(dominatorRebindOp);
    return success();
  }

  return success(succeeded(paramsResolved));
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

//===----------------------------------------------------------------------===//
// MO_ReduceMulOp support
//===----------------------------------------------------------------------===//

OpFoldResult ReduceMulOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto inputArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInput());
  auto axisArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getAxis());
  if (!inputArr || !axisArr)
    return {};

  // We only support rank-1 integers (i.e. shape-like).
  if (inputArr.getType().getRank() != 1)
    return {};

  int64_t axis = getBlob(axisArr).front();
  if (axis != 0)
    return {};

  TensorType resTy = getType();
  auto maybeInput = tryGetConstantIntBlob(inputArr, resTy.getDtype());
  if (!maybeInput.has_value())
    return {};

  ArrayRef<int64_t> inputs = *maybeInput;
  int64_t res =
      std::accumulate(inputs.begin(), inputs.end(), 1, std::multiplies<>());

  SmallVector<APInt> resArr{APInt(64, res, /*isSigned=*/true)};
  auto rankedResTy = RankedTensorType::get({1}, resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, resArr);
}

//===----------------------------------------------------------------------===//
// MO_ReshapeOp support
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::canonicalize(ReshapeOp op, PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  if (inputTy.hasUnknownShape())
    return failure();

  TensorType resTy = op.getType();
  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // TODO(#11269) Emit runtime checks.
  // [] --> [?]
  if (inputTy.getRank() == 0 && resTy.getRank() == 1) {
    auto inferredResTy =
        TensorType::get(SmallVector<int64_t>({1}), resTy.getElementType());
    auto newReshapeOp = rewriter.create<Static::ReshapeOp>(
        op.getLoc(), inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp);
    return success();
  }

  // [?] --> []
  if (inputTy.getRank() == 1 && resTy.getRank() == 0) {
    auto inferredResTy =
        TensorType::get(SmallVector<int64_t>({}), resTy.getElementType());
    auto newReshapeOp = rewriter.create<Static::ReshapeOp>(
        op.getLoc(), inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp);
    return success();
  }

  return failure();
}

// TODO refactor with `RangeOp::deparameterize`
LogicalResult ReshapeOp::deparameterize(mlir::RewriterBase &rewriter) {
  ArrayRef<KGEN::ParamDeclAttr> decls = getOutputParamDecls();
  if (decls.empty())
    return failure();
  setOutputParamDecls({});

  TensorType resTy = getType();
  auto shapeTy = rewriter.getType<ShapeType>();
  TensorType unknownResTy;
  if (resTy.hasRank())
    unknownResTy = resTy.cloneWith(resTy.getIntDims(), resTy.getElementType());
  else
    unknownResTy = resTy.cloneWith(KGEN::UnknownAttr::get(shapeTy));

  auto newOp = rewriter.create<MO::ReshapeOp>(
      getLoc(), unknownResTy, getOperands(),
      getOperation()->getAttrDictionary().getValue());
  rewriter.replaceOpWithNewOp<RebindOp>(*this, resTy, newOp, decls);

  return success();
}

/// Helper to fold reshape ops.
static ArrayElementsAttr foldReshape(ArrayRef<int64_t> outIntDims,
                                     ArrayElementsAttr inputArr) {
  if (llvm::is_contained(outIntDims, -1))
    return {};
  auto outTy = RankedTensorType::get(outIntDims, inputArr.getElementType());
  return ArrayElementsAttr::get(inputArr.getRawData(), outTy);
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto inputArr = dyn_cast_or_null<ArrayElementsAttr>(adaptor.getInput());
  auto shapeArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getNewShape());
  auto outputShapeAttr = dyn_cast<ShapeAttr>(getType().getShapeAttr());
  if (!inputArr || !shapeArr || !outputShapeAttr)
    return {};

  // Compute number of elements specified by target shape
  int64_t shapeNumElems = 1;
  for (llvm::APInt dim : shapeArr.getValues()) {
    if (dim == -1)
      return {};
    shapeNumElems *= dim.getSExtValue();
  }

  // Compute number of elements specified by output type
  int64_t outputNumElems = 1;
  SmallVector<int64_t> outIntDims = outputShapeAttr.getIntDims();
  for (int64_t intDim : outIntDims) {
    if (mlir::ShapedType::isDynamic(intDim) || intDim == -1)
      return {};
    outputNumElems *= intDim;
  }

  // Reshape is valid if number of elements match.
  if (shapeNumElems == outputNumElems)
    return foldReshape(outIntDims, inputArr);

  return {};
}

//===----------------------------------------------------------------------===//
// MOStatic_ReshapeOp
//===----------------------------------------------------------------------===//

/// Returns a parameter constant attribute through a variant of ToTensor ops if
/// possible. Otherwise returns a null attribute.
template <typename OpTy>
static TypedAttr getParamConstThrough(TensorValue inputVal) {
  if (auto toTensorOp = inputVal.template getDefiningOp<OpTy>()) {
    if (auto constOp = toTensorOp.getInput()
                           .template getDefiningOp<KGEN::ParamConstantOp>()) {
      return constOp.getValue();
    }
  }
  return {};
}

LogicalResult Static::ReshapeOp::canonicalize(Static::ReshapeOp op,
                                              PatternRewriter &rewriter) {
  // We fold consecutive static reshapes into a single reshape.
  if (auto precedingOp = op.getInput().getDefiningOp<Static::ReshapeOp>()) {
    op.setOperand(precedingOp.getInput());
    return success();
  }

  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  TensorType resTy = op.getType();
  if (!resTy.hasRank() || !inputTy.hasRank())
    return failure();

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank == 0 && resRank == 1) {
    if (TypedAttr param = getParamConstThrough<IndexToTensorOp>(inputVal)) {
      // TODO: test
      auto shape = ShapeAttr::get(SmallVector<TypedAttr>{param},
                                  rewriter.getType<ShapeType>());
      auto newConstOp =
          rewriter.create<KGEN::ParamConstantOp>(op->getLoc(), shape);
      rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, newConstOp);
    }
  } else if (inputRank == 1 && resRank == 0) {
    if (TypedAttr param = getParamConstThrough<Shape::ToTensorOp>(inputVal)) {
      if (auto shape = dyn_cast<ShapeAttr>(param);
          shape && shape.getRank() == 1) {
        // TODO: test
        auto newConstOp = rewriter.create<KGEN::ParamConstantOp>(
            op->getLoc(), shape.getValues()[0]);
        rewriter.replaceOpWithNewOp<IndexToTensorOp>(op, resTy, newConstOp);
      }
    }
  }

  if (inputTy.getRank() != resTy.getRank())
    return failure();

  // A reshape that changes only a single dimension is just a rebind.
  bool foundDifferentDim = false;
  for (auto [inDim, resDim] : llvm::zip(inputTy.getDims(), resTy.getDims())) {
    if (inDim != resDim) {
      if (foundDifferentDim)
        return failure();
      foundDifferentDim = true;
    }
  }

  rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, op.getInput());
  return success();
}

OpFoldResult Static::ReshapeOp::fold(FoldAdaptor adaptor) {
  if (auto inputArr = dyn_cast_or_null<ArrayElementsAttr>(adaptor.getInput())) {
    auto outputShape = dyn_cast<ShapeAttr>(getType().getShapeAttr());
    if (outputShape && outputShape.isStatic())
      return foldReshape(outputShape.getIntDims(), inputArr);
  }

  if (getInput().getType() == getType())
    return getInput();

  return {};
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
// MO_SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  if (inputTy.hasRank()) {
    int64_t inputRank = inputTy.getRank();
    if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
      return emitError("start shape must be [inputRank] if it's static");

    if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
      return emitError("stop shape must be [inputRank] if it's static");

    if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
      return emitError("step shape must be [inputRank] if it's static");

  } else {
    if (startTy.hasStaticShape()) {
      return emitError(
          "start shape must be dynamic if input is has dynamic rank");
    }

    if (stopTy.hasStaticShape()) {
      return emitError(
          "stop shape must be dynamic if input is has dynamic rank");
    }

    if (stepTy.hasStaticShape()) {
      return emitError(
          "step shape must be dynamic if input is has dynamic rank");
    }
  }
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
// Linear Algebra Ops
//===----------------------------------------------------------------------===//

LogicalResult MatrixBandPartOp::verify() {
  // TODO: Refactor this into a trait when we see another op that does this
  if (getInput().getType() != getResult().getType())
    return emitError("first operand and the result must have the same type");
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

OpFoldResult Shape::BroadcastOp::fold(FoldAdaptor adaptor) {
  auto xShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputX());
  auto yShape = dyn_cast_or_null<ShapeAttr>(adaptor.getInputY());
  if (xShape && yShape) {
    ErrorOr<MO::ShapeAttr> shapeOr = broadcastShape(xShape, yShape);
    if (!shapeOr.isError())
      return *shapeOr;
  }
  return {};
}

void Shape::ConcatOp::build(OpBuilder &builder, OperationState &state,
                            TypedValue<ShapeType> input1,
                            TypedValue<ShapeType> input2) {
  return build(builder, state, {input1, input2});
}

OpFoldResult Shape::ConcatOp::fold(FoldAdaptor adaptor) {
  // Single shape parameter or dimension parameters
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

TypedAttr Shape::EqualOp::asExpression(ShapeAttr shape1, ShapeAttr shape2) {
  assert(shape1.getRank() == shape2.getRank());
  SmallVector<TypedAttr> andOperands;
  for (auto [xDim, yDim] : llvm::zip(shape1.getValues(), shape2.getValues())) {
    andOperands.emplace_back(
        KGEN::ParamOperatorAttr::get(KGEN::POC::EQ, {xDim, yDim}));
  }
  return KGEN::ParamOperatorAttr::get(KGEN::POC::And, andOperands);
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
  if (!shape1 || !shape2)
    return {};

  // If the inputs are different rank, we can fold.
  if (shape1.getRank() != shape2.getRank())
    return BoolAttr::get(getContext(), false);

  // If both inputs have the same static rank, we fold into an expression.
  return asExpression(shape1, shape2);
}

OpFoldResult Shape::FromTensorOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInput()))
    return ShapeAttr::get(input, ShapeType::get(getContext()));
  return {};
}

void Shape::GetDimOp::build(OpBuilder &builder, OperationState &state,
                            Value input, int64_t dim) {
  build(builder, state, input, builder.getIndexAttr(dim));
}

TypedAttr Shape::GetDimOp::emitStaticCall(TypedAttr inputShape, int64_t axis) {
  auto indexTy = IndexType::get(inputShape.getContext());
  SmallVector<std::pair<StringAttr (*)(mlir::OperationName), Attribute>>
      attrNameFuncAndArg = {
          {Shape::GetDimOp::getDimAttrName, IntegerAttr::get(indexTy, axis)}};
  return KGEN::emitMLIROperationCall(Shape::GetDimOp::getOperationName(),
                                     attrNameFuncAndArg, {inputShape}, indexTy);
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

OpFoldResult Shape::NumElementsOp::fold(FoldAdaptor adaptor) {
  if (auto inputDimAttrs = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    return KGEN::ParamOperatorAttr::get(KGEN::POC::MulNuw,
                                        inputDimAttrs.getValues());
  }

  return {};
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
    if (getType().hasUnknownShape()) {
      OpBuilder builder(getOperation());
      builder.setInsertionPointAfter(getOperation());
      TensorType newTy = getType().cloneWith(ShapeAttr::get(
          {(int64_t)input.getRank()}, builder.getType<MO::ShapeType>()));
      Location loc = getLoc();
      auto newOp = builder.create<Shape::ToTensorOp>(loc, newTy, getInput());
      auto rebindOp = builder.create<RebindOp>(loc, getType(), newOp);
      getResult().replaceAllUsesExcept(rebindOp, rebindOp.getOperation());
      return rebindOp.getResult();
    }

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
