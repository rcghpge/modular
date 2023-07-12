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
#include "GenericML/GraphCompiler/MODialect/MOAttrs.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/Permutation.h"
#include "Support/ML/RangeUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <numeric>

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

SmallVector<ConstantOp> M::MO::tryGetConstantOpsOfRank(TensorValue val,
                                                       size_t rank) {
  SmallVector<ConstantOp> constantOps;

  Operation *inputOp = val.getDefiningOp();

  // Handle identity chains.
  auto isIdentityLike = [&](Operation *op) {
    if (dyn_cast_or_null<RebindOp>(op)) {
      return true;
    } else if (dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(op)) {
      return true;
    } else if (dyn_cast_or_null<ReshapeOp>(op)) {
      // After a ReshapeOp, accept constants of any rank.
      rank = std::numeric_limits<size_t>::max();
      return true;
    }
    return false;
  };

  while (isIdentityLike(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();

  // Handle concatenations of constant tensors.
  auto concatOp = dyn_cast_or_null<ConcatOp>(inputOp);
  if (concatOp) {
    for (unsigned i = 1, e = concatOp->getNumOperands(); i < e; ++i) {
      TensorValue operand = cast<TensorValue>(concatOp->getOperand(i));
      SmallVector<ConstantOp> operandConstantOps =
          tryGetConstantOpsOfRank(operand, rank);
      if (operandConstantOps.empty())
        return {};
      constantOps.insert(constantOps.end(), operandConstantOps.begin(),
                         operandConstantOps.end());
    }
    return constantOps;
  }

  // Handle ConstantOp last.
  auto constantOp = dyn_cast_or_null<ConstantOp>(inputOp);
  if (!constantOp)
    return {};

  MO::TensorType type = constantOp.getType();
  if (rank != std::numeric_limits<size_t>::max() &&
      (!type.hasRank() || type.getRank() != rank))
    return {};

  constantOps.push_back(constantOp);

  return constantOps;
}

std::optional<float> M::MO::tryGetConstantFloat(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<float>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<int64_t> M::MO::tryGetConstantInt(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<bool> M::MO::tryGetConstantBool(TensorValue val) {
  if (auto maybeIntBlob = tryGetConstantOfRank<int64_t>(val, 0))
    return maybeIntBlob.value().front();
  return std::nullopt;
}

std::optional<SmallVector<int64_t>>
M::MO::tryGetConstantIntVector(TensorValue val) {
  if (val.getType().getElementType().isIntOrIndex())
    return tryGetConstantOfRank<int64_t>(val, 1);
  return std::nullopt;
}

std::optional<SmallVector<bool>>
M::MO::tryGetConstantBoolVector(TensorValue val) {
  if (val.getType().getElementType().isInteger(1))
    return tryGetConstantOfRank<bool>(val, 1);
  return std::nullopt;
}

template <typename Adaptor>
OpFoldResult foldElemwiseOp(MO::TensorType resTy, Adaptor adaptor,
                            std::function<int64_t(int64_t, int64_t)> lambda,
                            size_t bitwidth = 64, bool isSigned = true) {
  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputY());
  if (!xAttr || !yAttr)
    return {};

  DType resDType = resTy.getDtype();
  auto maybeX =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(xAttr), resDType);
  auto maybeY =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(yAttr), resDType);
  if (!maybeX.has_value() || !maybeY.has_value())
    return {};

  ArrayRef<int64_t> xArr = *maybeX;
  ArrayRef<int64_t> yArr = *maybeY;
  SmallVector<APInt> res(xArr.size());
  for (size_t idx = 0, e = xArr.size(); idx < e; idx++)
    res[idx] = APInt(bitwidth, lambda(xArr[idx], yArr[idx]), isSigned);

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, res);
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
  state.addAttribute(getSignatureAttrName(state.name), TypeAttr::get(sig));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));
  auto emptyParams = KGEN::ParamDeclArrayAttr::get(builder.getContext(), {});
  state.addAttribute(getInputParamsAttrName(state.name), emptyParams);
  state.addAttribute(getResultParamsAttrName(state.name), emptyParams);

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
                                    KGEN::GeneratorOrFuncKind::generator);
}

/// Print the GraphOp.
void GraphOp::print(OpAsmPrinter &p) { KGEN::printGeneratorOrFunc(p, *this); }

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

LogicalResult GraphOp::verify() {
  mlir::SymbolTableAnalysis symTabAnalysis(getOperation());
  mlir::LockedSymbolTableCollection symtabs(symTabAnalysis.getSymbolTables());
  KGEN::ParameterCollector::Analysis cache;
  KGEN::ParameterUseDefGraph useDefGraph(getRegion());

  // check that no expressions in the input parameters
  mlir::AttrTypeWalker walker;
  bool found = false;
  walker.addWalk([&](KGEN::ParamOperatorAttr attr) {
    found = true;
    return WalkResult::interrupt();
  });
  for (auto ty : getArgumentTypes())
    walker.walk(ty);
  if (found) {
    return emitError(
        "Expressions are not (currently) allowed in function signatures");
  }
  return useDefGraph.verify(symtabs, cache);
}

KGEN::ParamDeclAttr GraphOp::getNewDimParam(StringRef prefix) {
  return getNewParamOfType(prefix, IndexType::get(getContext()));
}

KGEN::ParamDeclAttr GraphOp::getNewShapeParam(StringRef prefix) {
  return getNewParamOfType(prefix, ShapeType::get(getContext()));
}

KGEN::ParamDeclAttr GraphOp::getNewParamOfType(StringRef prefix, Type type) {
  return KGEN::ParamDeclAttr::get(getNewParamName(prefix), type);
}

std::string GraphOp::getNewParamName(StringRef prefix) {
  // TODO(#8829) avoid duplication with existing parameters in the input graph
  int64_t newCounter = getProperties().getCounter() + 1;
  getProperties().setCounter(newCounter);
  return prefix.str() + std::to_string(newCounter);
}

//===----------------------------------------------------------------------===//
// MO_OutputOp support
//===----------------------------------------------------------------------===//

void OutputOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParams();
  ArrayRef<TypedAttr> defs = this->getParameters();
  assert(decls.size() == defs.size() &&
         "output bindings and result parameters must have the same size");

  for (auto [decl, def] : llvm::zip(decls, defs))
    walkDef(decl, def);
}

LogicalResult OutputOp::verify() {
  // ODS checks the parent is a GraphOp.
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParams();
  ArrayRef<TypedAttr> defs = this->getParameters();
  if (decls.size() != defs.size()) {
    return emitError(
        "output bindings and result parameters must have the same size");
  }

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> ResParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] : llvm::zip(decls, defs)) {
    ResParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      TypedAttr newShapeAttr = cast<TypedAttr>(tensorTy.getShapeAttr().replace(
          [&ResParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = ResParamRefToBindingAttr.lookup(paramRef);
            return resolvedAttr ? resolvedAttr : paramRef;
          }));
      expectedOutputOperandTypes.emplace_back(tensorTy.cloneWith(newShapeAttr));
    } else {
      expectedOutputOperandTypes.emplace_back(ty);
    }
  }

  return checkOperandTypes(getOperation(), expectedOutputOperandTypes);
}

//===----------------------------------------------------------------------===//
// IfOp support
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  // The 'then' and 'else' blocks take no arguments.
  auto *thenBlock = &getThenRegion().front();
  if (thenBlock->getNumArguments() != 0)
    return emitOpError("expected no arguments to 'then' block");

  auto *elseBlock = &getElseRegion().front();
  if (elseBlock->getNumArguments() != 0)
    return emitOpError("expected no arguments to 'else' block");

  // The 'then' block results must match the mo.if results.
  auto thenOutput = cast<YieldOp>(thenBlock->getTerminator());
  if (failed(checkOperandTypes(thenOutput, getResultTypes())))
    return failure();

  // The 'else' block results must match the mo.if results.
  auto elseOutput = cast<YieldOp>(elseBlock->getTerminator());
  return checkOperandTypes(elseOutput, getResultTypes());
}

//===----------------------------------------------------------------------===//
// WhileOp support
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();

  if (parseParenOperandListWithShadowing(parser, result, argumentInfo) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegionWithShadowing(parser, argumentInfo, *condRegion) ||
      parser.parseKeyword("do") ||
      parseRegionWithShadowing(parser, argumentInfo, *bodyRegion))
    return failure();

  // The op's return types match those of the signature arguments.
  result.types.reserve(result.operands.size());
  for (auto operand : result.operands)
    result.types.push_back(operand.getType());

  return success();
}

void WhileOp::print(OpAsmPrinter &p) {
  p << ' ';
  printParenOperandListWithShadowing(p, getInputs(),
                                     getCondRegion().getArguments());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegionWithShadowing(p, getInputs(), getCondRegion());
  p << " do ";
  printRegionWithShadowing(p, getInputs(), getBodyRegion());
}

LogicalResult WhileOp::verify() {
  // The grt.while results must match the signature types.
  if (failed(checkMatchingTypes(*this, "result", getResultTypes(),
                                getInputs().getTypes())))
    return failure();

  // The 'cond' block should accept arguments matching the grt.while signature.
  auto *condBlock = &getCondRegion().front();
  if (failed(
          checkArgumentTypes(*this, "cond", condBlock, getInputs().getTypes())))
    return failure();

  // The 'body' block should accept arguments matching the grt.while signature.
  auto *bodyBlock = &getBodyRegion().front();
  if (failed(
          checkArgumentTypes(*this, "body", bodyBlock, getInputs().getTypes())))
    return failure();

  // The 'cond' block returns a single 'scalar' boolean.
  auto condOutput = cast<MO::YieldOp>(condBlock->getTerminator());
  SmallVector<Type, 1> expectedTypes;
  expectedTypes.emplace_back(MO::TensorType::get(
      SmallVector<int64_t, 0>(), IntegerType::get(getContext(), 1)));
  if (failed(checkOperandTypes(condOutput, expectedTypes)))
    return failure();

  // The 'body' block returns a result for each mo.while result.
  auto bodyOutput = cast<MO::YieldOp>(bodyBlock->getTerminator());
  if (failed(checkOperandTypes(bodyOutput, getResultTypes())))
    return failure();

  return success();
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
    // usually lowered to `mosh.broadcast`, it should be folded rarely.
    auto shapeTy = ShapeType::get(getContext());
    auto xShape = ShapeAttr::get(xIntDims, shapeTy);
    auto yShape = ShapeAttr::get(yIntDims, shapeTy);
    ErrorOr<ShapeAttr> shapeOr = ShapeAttr::broadcastShapes(xShape, yShape);
    if (!shapeOr.isError()) {
      // Ensure we return with the appropriate element type.
      auto getResAttrs = [&](auto values) {
        return IntArrayElementsAttr::get(
            getContext(), values, IntegerType::SignednessSemantics::Signed);
      };
      if (getType().getElementType().getIntOrFloatBitWidth() == 32) {
        SmallVector<int32_t> dims = llvm::map_to_vector(
            shapeOr->getIntDims(), [](int64_t dim) -> int32_t { return dim; });
        return getResAttrs(ArrayRef<int32_t>{dims});
      }
      return getResAttrs(ArrayRef<int64_t>{shapeOr->getIntDims()});
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Elementwise folder support
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInputY());

  if (xAttr && yAttr) {
    return foldElemwiseOp(getType(), adaptor,
                          [](int64_t x, int64_t y) { return x + y; });
  } else {
    // We normally would reject non-constant inputs, but if one of the inputs is
    // a broadcast of 0 which we can still reason about them.
    TensorValue lhs = getLhsInput(), rhs = getRhsInput();

    // Check if a given value is the product of a constant of 0 being
    // broadcasted.
    auto isValueBroadcasted0Const = [](Value operand) {
      if (auto bcast = dyn_cast_or_null<Static::BroadcastToOp>(
              operand.getDefiningOp())) {
        if (auto constOp = dyn_cast_or_null<MO::ConstantOp>(
                bcast.getOperand().getDefiningOp())) {
          TensorType type = cast<TensorType>(constOp.getType());
          // Interpret as double. Should be ok as integers will be casted to
          // double.
          auto data =
              tryGetConstantBlob<double>(constOp.getValue(), type.getDtype());
          if (!data.has_value())
            return false;

          // Check the constant is zero.
          for (double i : *data) {
            if (i != 0.0)
              return false;
          }

          // We have a match.
          return true;
        }
      }
      return false;
    };

    // Check combination of 0 + x
    if (isValueBroadcasted0Const(lhs))
      return rhs;

    // Otherwise check x + 0
    if (isValueBroadcasted0Const(rhs))
      return lhs;
  }

  return {};
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  return foldElemwiseOp(getType(), adaptor,
                        [](int64_t x, int64_t y) { return x / y; });
}

OpFoldResult GreaterEqualOp::fold(FoldAdaptor adaptor) {
  return foldElemwiseOp(
      getType(), adaptor, [](int64_t x, int64_t y) { return x >= y; },
      /*bitwidth=*/1, /*isSigned=*/false);
}

OpFoldResult GreaterOp::fold(FoldAdaptor adaptor) {
  return foldElemwiseOp(
      getType(), adaptor, [](int64_t x, int64_t y) { return x > y; },
      /*bitwidth=*/1, /*isSigned=*/false);
}

OpFoldResult EqualOp::fold(FoldAdaptor adaptor) {
  return foldElemwiseOp(
      getType(), adaptor, [](int64_t x, int64_t y) { return x == y; },
      /*bitwidth=*/1, /*isSigned=*/false);
}

OpFoldResult NotEqualOp::fold(FoldAdaptor adaptor) {
  return foldElemwiseOp(
      getType(), adaptor, [](int64_t x, int64_t y) { return x != y; },
      /*bitwidth=*/1, /*isSigned=*/false);
}

//===----------------------------------------------------------------------===//
// MO_BroadcastToOp
//===----------------------------------------------------------------------===//

/// Check if the input type can be broadcasted to the result type, assuming both
/// have static ranks.
static ErrorOrSuccess verifyBroadcastStaticRanks(TensorType inputTy,
                                                 TensorType resTy) {
  if (inputTy.hasRank() && resTy.hasRank()) {
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

LogicalResult Static::BroadcastToOp::canonicalize(Static::BroadcastToOp op,
                                                  PatternRewriter &rewriter) {
  TensorType resTy = op.getType();
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();

  if (auto pastBroadcastOp =
          op.getInput().getDefiningOp<Static::BroadcastToOp>()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(
        op, op.getType(), pastBroadcastOp.getInput());
    return success();
  }

  if (resTy.getShapeAttr() == inputTy.getShapeAttr()) {
    rewriter.replaceAllUsesExcept(op, inputVal, op);
    return success();
  }

  OpBuilder::InsertPoint insertPoint = rewriter.saveInsertionPoint();
  for (Operation *user :
       op->getUsers()) { // intentional sabatoge -- remove if it catches things
    if (auto sinkInterface = dyn_cast<SinkStaticBroadcastToInterface>(user)) {
      rewriter.setInsertionPoint(user);
      if (succeeded(sinkInterface.sinkStaticBroadcastTo(rewriter)))
        return success();
    }
  }
  rewriter.restoreInsertionPoint(insertPoint);

  if (!inputTy.hasRank() || !resTy.hasRank())
    return failure();

  // If a broadcast has an empty input, it is not equivalent to a reshape.
  auto resTyAttr = dyn_cast<IntegerAttr>(resTy.getSymbolicNumElements());
  auto inputTyAttr = dyn_cast<IntegerAttr>(inputTy.getSymbolicNumElements());
  if ((resTyAttr && resTyAttr.getInt() == 0) ||
      (inputTyAttr && inputTyAttr.getInt() == 0))
    return failure();

  // If a broadcast does not change the number of elements, it is equivalent to
  // a reshape. This is something we can check symbolically.
  if (resTy.getSymbolicNumElements() == inputTy.getSymbolicNumElements()) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
    return success();
  }

  return failure();
}

OpFoldResult Static::BroadcastToOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  TensorType outTy = getType();
  auto floatInputAttr =
      dyn_cast_or_null<FloatArrayElementsAttr>(adaptor.getInput());
  if (floatInputAttr && floatInputAttr.getNumElements() == 1 &&
      outTy.hasStaticShape() && outTy.getElementType().isF32()) {
    float inputScalar = getFloatBlob(floatInputAttr).front();

    // TODO consider lifting this into a TensorType util.
    SmallVector<int64_t> outIntDims = outTy.getIntDims();
    int64_t outNumElems = std::accumulate(outIntDims.begin(), outIntDims.end(),
                                          1, std::multiplies<int64_t>());
    if (outNumElems > 10)
      return {};

    SmallVector<APFloat> res(outNumElems, APFloat(inputScalar));

    auto rankedResTy =
        RankedTensorType::get(outIntDims, outTy.getElementType());
    return FloatArrayElementsAttr::get(rankedResTy, res);
  }
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
    auto staticTensorType =
        TensorType::get(cast<ShapedType>(elements.getType()).getShape(),
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
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int32_t> values) {
  auto type =
      MO::TensorType::get({(int64_t)values.size()},
                          odsBuilder.getIntegerType(32, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      values, mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APFloat> values, Type elementType) {
  auto type = MO::TensorType::get({(int64_t)values.size()}, elementType);
  auto valueAttr = FloatArrayElementsAttr::get(values, type.getElementType());
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<bool> values) {
  auto type =
      MO::TensorType::get({(int64_t)values.size()}, odsBuilder.getI1Type());
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      values, mlir::IntegerType::Signless);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value) {
  auto type = MO::TensorType::get(
      SmallVector<int64_t>{}, odsBuilder.getIntegerType(64, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int64_t>(SmallVector<int64_t>{value}),
      mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int32_t value) {
  auto type = MO::TensorType::get(
      SmallVector<int64_t>{}, odsBuilder.getIntegerType(32, /*isSigned=*/true));
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<int32_t>(SmallVector<int32_t>{value}),
      mlir::IntegerType::Signed);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APFloat value, Type elementType) {
  auto type = MO::TensorType::get(SmallVector<int64_t>{}, elementType);
  auto valueAttr =
      FloatArrayElementsAttr::get({std::move(value)}, type.getElementType());
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       float value) {
  Type dtype = odsBuilder.getF32Type();
  auto type = MO::TensorType::get(SmallVector<int64_t>{}, dtype);
  SmallVector<APFloat> data;
  data.emplace_back(value);
  auto valueAttr = FloatArrayElementsAttr::get(data, dtype);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       double value) {
  Type dtype = odsBuilder.getF64Type();
  auto type = MO::TensorType::get(SmallVector<int64_t>{}, dtype);
  SmallVector<APFloat> data;
  data.emplace_back(value);
  auto valueAttr = FloatArrayElementsAttr::get(data, dtype);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       bool value) {
  auto type =
      MO::TensorType::get(SmallVector<int64_t>{}, odsBuilder.getI1Type());
  auto valueAttr = odsBuilder.getAttr<IntArrayElementsAttr>(
      ArrayRef<bool>(SmallVector<bool>{value}), mlir::IntegerType::Signless);
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

//===----------------------------------------------------------------------===//
// MO_ConcatOp support
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  TensorType outType = getResult().getType();

  // Enforce rank 1 so we don't need to check axis or deal with complex
  // stride stuff.
  if (!outType.hasRank() || outType.getRank() != 1 || !outType.hasStaticShape())
    return {};

  SmallVector<APInt> concatTensor;

  DType outDtype = outType.getDtype();
  for (Attribute attr : adaptor.getInputs()) {
    // Enforce that it only runs on ints.
    auto asIntAttr = dyn_cast_or_null<IntArrayElementsAttr>(attr);
    if (!asIntAttr)
      return {};

    auto maybeValue =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(asIntAttr), outDtype);
    if (!maybeValue)
      return {};

    // Append the values to the output tensor. We can do this since it is
    // rank 1.
    for (size_t idx = 0, e = maybeValue->size(); idx < e; idx++)
      concatTensor.push_back(APInt(64, (*maybeValue)[idx], true));
  }

  auto rankedResTy =
      RankedTensorType::get(outType.getIntDims(), outType.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, concatTensor);
}

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

  return success();
}

LogicalResult ConcatOp::canonicalize(ConcatOp op, PatternRewriter &rewriter) {
  Operation::operand_range inputs = op.getInputs();
  SmallVector<Value> nonEmptyInputs;
  for (Value arg : inputs) {
    TensorType argTy = cast<TensorType>(arg.getType());
    auto isNonZero = [](int64_t dim) { return dim != 0; };
    if (!argTy.hasRank() || llvm::all_of(argTy.getIntDims(), isNonZero))
      nonEmptyInputs.emplace_back(arg);
  }

  // TODO: implement this
  if (nonEmptyInputs.empty())
    return failure();

  if (nonEmptyInputs.size() == 1) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(),
                                          nonEmptyInputs.front());
    return success();
  }

  if (nonEmptyInputs.size() != inputs.size()) {
    op.getInputsMutable().assign(nonEmptyInputs);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MO_SplitOp support
//===----------------------------------------------------------------------===//

/// Verify that the sum of `outputs` shapes along `splitAxis` matches the input
/// shape, and that the other dimensions match.
static ErrorOrSuccess verifyMatchingSplitShapes(ArrayRef<TensorValue> outputs,
                                                TensorType inputTy,
                                                int64_t splitAxis) {
  int64_t splitDimSum = 0;
  for (auto res : outputs) {
    TensorType resTy = res.getType();
    for (auto [i, dimSize] : llvm::enumerate(resTy.getIntDims())) {
      if ((int64_t)i == splitAxis) {
        splitDimSum += dimSize;
      } else if (dimSize != inputTy.getIntDim(i)) {
        return Error(
            "input and output static shapes must match except split axis");
      }
    }
  }

  if (splitDimSum != inputTy.getIntDim(splitAxis)) {
    return Error(
        "sum of output shapes along split axis must match input shape");
  }

  return success();
}

LogicalResult SplitOp::verify() {
  SmallVector<TensorValue> tensors;
  tensors.emplace_back(getInput());
  for (Value output : getResults())
    tensors.emplace_back(cast<TensorValue>(output));

  if (failed(impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  TensorType inputTy = getInput().getType();
  TensorType resultTy = cast<TensorType>(getResults().front().getType());
  if (inputTy.getDtype() != resultTy.getDtype())
    return emitError("input and output element types must match");

  // If the input has a static shape and the axis is a constant, then verify
  // that the axis is in range.
  // Then, if the results also have static shapes, verify that the sum of the
  // output shapes along the split axis matches the input shape, and that the
  // other dimensions match.
  std::optional<int64_t> axisOr = tryGetConstantInt(getAxis());
  if (inputTy.hasStaticShape() && axisOr) {
    int64_t axis = *axisOr;
    int64_t inputRank = inputTy.getRank();
    int64_t normalizedAxis = (axis < 0) ? axis + inputRank : axis;
    if ((normalizedAxis < 0) || (inputRank <= normalizedAxis))
      return emitError("split axis must be in range [-rank, rank)");

    if (resultTy.hasStaticShape()) {
      ErrorOrSuccess result = verifyMatchingSplitShapes(
          ArrayRef<TensorValue>(tensors).drop_front(), inputTy, normalizedAxis);
      if (result.isError())
        return emitError(result.getError());
    }
  }

  return success();
}

LogicalResult SplitOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {
  // If it is a single value 1D tensor then it is trivially foldable as it can
  // only be an identity. (note scalars are not legal inputs).
  TensorType inType = getInput().getType();
  if (inType.hasRank() && inType.getRank() == 1 && inType.getIntDim(0) == 1) {
    results.push_back(getInput());
    return success();
  }

  // If the split is splitting into one tensor then it is identity.
  auto splits = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getSplitSizes());
  if (splits) {
    auto asArr = splits.asArrayRef<int64_t>();
    if (asArr.size() == 1 && asArr[0] == 1) {
      results.push_back(getInput());
      return success();
    }
  }

  return failure();
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

  auto inputElems = dyn_cast_or_null<ElementsAttr>(adaptor.getInput());
  auto idxElems = dyn_cast_or_null<ElementsAttr>(adaptor.getIndices());

  // The verifier should ensure that indices is empty, but we don't check it.
  if (inputElems && inputElems.empty())
    return getEmptyArray(inputElems.getElementType());

  if (idxElems && idxElems.empty())
    return getEmptyArray(idxElems.getElementType());

  if (!inputElems || !idxElems)
    return {};

  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();

  if (!resTy.hasRank() || !inTy.hasRank())
    return {};

  // We only perform this on rank 1 gathers.
  if (resTy.getRank() != inTy.getRank() || inTy.getRank() != 1 ||
      !resTy.hasStaticShape() || !inTy.hasStaticShape())
    return {};

  DType dataTy = getInput().getType().getDtype();
  DType indicesTy = getIndices().getType().getDtype();

  // If the input/indices/axis is known this is something we can fold.
  // We have to enforce that they are only ints for now.
  // We only run on rank 1 so we know the only possible value for axis is 1.
  auto dataAttr = dyn_cast<IntArrayElementsAttr>(inputElems);
  auto iAttr = dyn_cast<IntArrayElementsAttr>(idxElems);
  if (!dataAttr || !iAttr)
    return {};

  auto maybeData =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(dataAttr), dataTy);
  auto maybeInd =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(iAttr), indicesTy);
  if (!maybeData.has_value() || !maybeInd.has_value())
    return {};

  ArrayRef<int64_t> data = *maybeData;
  ArrayRef<int64_t> indices = *maybeInd;

  SmallVector<APInt> gatheredValues(indices.size());
  for (size_t idx = 0, e = indices.size(); idx < e; idx++) {
    int64_t index = indices[idx];
    if (index < 0 || static_cast<size_t>(index) > data.size())
      return {};

    int64_t val = data[index];
    gatheredValues[idx] = APInt(64, val, /*isSigned=*/true);
  }

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  return IntArrayElementsAttr::get(rankedResTy, gatheredValues);
}

LogicalResult GatherOp::verify() {
  TensorType paramsType = getInput().getType();
  TensorType indicesType = getIndices().getType();
  TensorType resultType = getResult().getType();

  if (paramsType.getDtype() != resultType.getDtype())
    return emitError("input and output must have the same element type");

  // Nothing we can say if either input or indices is dynamically ranked
  if (!paramsType.hasRank() || !indicesType.hasRank())
    return success();

  if (!resultType.hasRank()) {
    return emitError(
        "result must be ranked if both param and indices are ranked");
  }

  if (paramsType.getRank() - 1 + indicesType.getRank() !=
      resultType.getRank()) {
    return emitError("result rank must be 1 less than the sum of param and "
                     "indices ranks if they are all static");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MO_CastOp support
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  // Cast to the same type is just identity.
  TensorType resTy = getResult().getType();
  TensorType inTy = getInput().getType();
  if (inTy == resTy)
    return getInput();

  // TODO(#10811): support other types and do this using a MOJO kernel?
  Attribute inputAttr = adaptor.getInput();
  if (!inputAttr)
    return {};

  auto rankedResTy =
      RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
  auto resElemTy = rankedResTy.getElementType();
  if (auto intElemsAttr = dyn_cast_or_null<IntArrayElementsAttr>(inputAttr)) {
    if (isa<IntegerType>(resElemTy)) {
      // This is inefficient, but simple.
      SmallVector<APInt> res = llvm::map_to_vector(
          intElemsAttr.getValues(), [](const llvm::APInt &dim) {
            return APInt(64, dim.getSExtValue(), /*isSigned=*/true);
          });

      return IntArrayElementsAttr::get(rankedResTy, res);
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// MO_SelectOp support
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?
  // TODO refactor parts of this with other folders

  // If the x and y are the same it's an identity op regardless of condition.
  if (getX() == getY())
    return getX();

  auto condAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getCond());
  auto xAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getX());
  auto yAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getY());
  if (!condAttr)
    return {};

  DType condDType = getCond().getType().getDtype();

  // If either or both inputs are unknown we can still fold symbolically. If
  // the condition is known to be all of one value then we can just return X if
  // the cond is true or Y if the cond is false.
  if (!xAttr || !yAttr) {
    auto maybeCond =
        tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
    if (!maybeCond.has_value())
      return {};

    ArrayRef<int64_t> condArr = *maybeCond;

    int64_t value = condArr[0];

    // Check the values are all the same otherwise exit.
    for (size_t idx = 1, e = condAttr.size(); idx < e; idx++) {
      if (value != condArr[idx])
        return {};
    }
    return value ? getX() : getY();
  }

  TensorType resTy = getType();
  DType resDType = resTy.getDtype();
  auto maybeCond =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(condAttr), condDType);
  auto maybeX =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(xAttr), resDType);
  auto maybeY =
      tryGetConstantBlob<int64_t>(cast<ElementsAttr>(yAttr), resDType);
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

LogicalResult SelectOp::verify() {
  DType xDType = getX().getType().getDtype();
  DType yDType = getY().getType().getDtype();
  DType resultDType = getResult().getType().getDtype();

  if (xDType != yDType)
    return emitError("operands must have the same element type");
  if (xDType != resultDType)
    return emitError(
        "operands and result type must have the same element type");

  return success();
}

//===----------------------------------------------------------------------===//
// MO_WhereOp support
//===----------------------------------------------------------------------===//

OpFoldResult WhereOp::fold(FoldAdaptor adaptor) {
  TensorType inTy = getInput().getType();
  TensorType outTy = getType();

  // TODO(#10811): support other types and do this using a MOJO kernel?
  auto floatInputAttr =
      dyn_cast_or_null<FloatArrayElementsAttr>(adaptor.getInput());
  bool inputIsF32Vector =
      inTy.hasRank() && inTy.getRank() == 1 && inTy.getElementType().isF32();
  if (floatInputAttr && inputIsF32Vector) {
    SmallVector<APInt> res;
    for (auto [idx, val] : llvm::enumerate(getFloatBlob(floatInputAttr)))
      if (val != 0)
        res.push_back(APInt(64, idx, /*isSigned=*/false));

    SmallVector<int64_t> outShape{static_cast<int64_t>(res.size()), 1};
    auto rankedOutTy = RankedTensorType::get(outShape, outTy.getElementType());
    auto foldedAttr = IntArrayElementsAttr::get(rankedOutTy, res);

    if (outTy.hasStaticShape()) {
      assert(outTy.getIntDims() == outShape &&
             "Annotated output type doesn't match folded output type");
      return foldedAttr;
    }

    // TODO refactor with `RangeOp::fold`
    Location loc = getLoc();
    OpBuilder builder(*this); // insert before this op
    auto resTy = TensorType::get(outShape, outTy.getElementType());
    auto constantOp =
        builder.create<ConstantOp>(loc, resTy, cast<ElementsAttr>(foldedAttr));
    auto rebindOp =
        builder.create<RebindOp>(loc, outTy, constantOp, getOutputParamDecls());
    setOutputParamDecls({});
    return rebindOp.getResult();
  }
  return {};
}

LogicalResult WhereOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();

  if (!resTy.hasRank() || resTy.getRank() != 2)
    return emitError("mo.where: invalid output shape: rank must be 2");

  if ((inputTy.hasRank() && resTy.getIntDim(1) != (int64_t)inputTy.getRank()) ||
      (!inputTy.hasRank() && !ShapedType::isDynamic(resTy.getIntDim(1))))
    return emitError("mo.where: invalid output shape: inner_dim != input_rank");

  return success();
}

//===----------------------------------------------------------------------===//
// MO::ArgMaxOp support
//===----------------------------------------------------------------------===//

static LogicalResult verifyWithRankedInput(ArgMaxOp *op, int64_t axis,
                                           int64_t resDim, TensorType inputTy,
                                           size_t &dynamicOutDims,
                                           size_t &reducedDims) {
  auto inDim = inputTy.getTypedShapeAttr().getIntDim(axis);
  if (mlir::ShapedType::isDynamic(resDim) ||
      mlir::ShapedType::isDynamic(inDim)) {
    dynamicOutDims++;
    return success();
  }
  if (inDim != resDim) {
    if (resDim != 1)
      return op->emitOpError(
          "mo.arg_max: output dimensions must match input "
          "dimensions, except at the reduction axis where the "
          "output dimension must be 1");
    reducedDims++;
  }
  return success();
};

static void verifyWithUnrankedInput(int64_t axis, int64_t resDim,
                                    size_t &dynamicOutDims,
                                    size_t &reducedDims) {
  if (mlir::ShapedType::isDynamic(resDim)) {
    dynamicOutDims++;
    return;
  }
  if (resDim == 1)
    reducedDims++;
};

LogicalResult ArgMaxOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();
  TensorType axisTy = getAxis().getType();

  // TODO(#12939): Refactor this into verifyReduction.

  if (!axisTy.getElementType().isSignedInteger(64))
    return emitOpError("axis type must be si64");

  if ((inputTy.hasRank() && resTy.hasRank()) &&
      (inputTy.getRank() != resTy.getRank()))
    return emitOpError("mo.arg_max: output rank must match input rank");

  if (resTy.hasRank()) {
    auto optAxis = tryGetConstantInt(getAxis());
    size_t reducedDims = 0;
    size_t dynamicOutDims = 0;

    for (int64_t axis = 0, rank = resTy.getRank(); axis < rank; axis++) {
      int64_t resDim = resTy.getTypedShapeAttr().getIntDim(axis);

      if (optAxis.has_value() && (axis == optAxis.value()) && (resDim != 1) &&
          !mlir::ShapedType::isDynamic(resDim))
        return emitOpError(
            "mo.arg_max: output dimensions must be 1 at the reduction axis");
      // Note: we don't check that non-reduction axes match directly. The code
      // below does it, which handles the dynamic cases as well.

      if (inputTy.hasRank()) {
        if (failed(verifyWithRankedInput(this, axis, resDim, inputTy,
                                         dynamicOutDims, reducedDims)))
          return failure();
      } else {
        verifyWithUnrankedInput(axis, resDim, dynamicOutDims, reducedDims);
      }
    }

    if ((reducedDims > 1) || (!reducedDims && !dynamicOutDims)) {
      return emitOpError(
          "mo.arg_max: output dimensions must match input "
          "dimensions, except for exactly one dimension, the reduction axis");
    }
  }

  return success();
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
  // The sliding window interface already checks that the input and result
  // have matching ranks, so we can be more specific with the error message
  // here.
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

  return success();
}

// Default builder which leaves the attributes empty.
void MatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     Type resultTy, Value input_a, Value input_b) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({resultTy});
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

  return success();
}

//===----------------------------------------------------------------------===//
// MO_PadOp support
//===----------------------------------------------------------------------===//

LogicalResult PadOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType paddingsTy = getPaddings().getType();
  TensorType resTy = getType();
  if (inputTy.hasRank()) {
    if (!resTy.hasRank())
      return emitOpError("result must be ranked if input is ranked");
    if ((int64_t)inputTy.getRank() != paddingsTy.getIntDim(0))
      return emitOpError("paddings shape does not match static input rank");
  } else {
    if (resTy.hasRank())
      return emitOpError("result must be unranked if input is unranked");
    if (isa<IntegerAttr>(paddingsTy.getDim(0)))
      return emitOpError("paddings shape does not match dynamic input rank");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MO_RangeOp support
//===----------------------------------------------------------------------===//

OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto startAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getStart());
  auto limitAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getLimit());
  auto stepAttr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getStep());
  if (!startAttr || !limitAttr || !stepAttr)
    return {};

  int64_t start = getIntBlob(startAttr).front();
  int64_t limit = getIntBlob(limitAttr).front();
  int64_t step = getIntBlob(stepAttr).front();

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
  TensorType outTy = getType();
  if (outTy.hasStaticShape())
    return foldedAttr;

  Location loc = getLoc();
  OpBuilder builder(*this); // insert before this op
  auto resTy = TensorType::get({numResElements}, outTy.getElementType());
  auto constantOp =
      builder.create<ConstantOp>(loc, resTy, cast<ElementsAttr>(foldedAttr));
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
    if (isa<ShapeType>(newDecl.getType()) &&
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
      //   kgen.param.declare D1 = <apply("mosh.get_dim", Sh0, 1)>
      //   rebind : Sh0 -> [D0, D1, 44]
      for (auto [axis, outputDimAttr] :
           llvm::enumerate(outputDimAttrs.getValues())) {
        KGEN::ParamDeclAttr paramDecl;
        auto outputDimParamRef =
            dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr);
        if (outputDimParamRef &&
            (paramDecl = nameToParamDecl.lookup(outputDimParamRef.getName()))) {
          TypedAttr mlirOpCallAttr =
              MOSH::GetDimOp::emitStaticCall(inputShapeParamRef, axis);
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

  // We materialize the parameter declarations explicitly.
  LogicalResult paramsResolved = materializeShapeInference(op, rewriter);

  // Likely already processed, or doesn't help shape inference.
  if (op.getInput().getDefiningOp<RebindOp>())
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
  if (op.getInput().getType().hasUnknownShape()) {
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
// MO_ReductionOp support
//===----------------------------------------------------------------------===//

template <typename T>
LogicalResult canonicalizeStaticNegativeIndicesInPlace(T op, int operandIndex) {
  // Use at top of T::fold(), which is run before any other canonicalizations.
  // If successful, then if axis and rank can be gotten at compile time, it is
  // now guaranteed to be in [0, rank).
  TensorValue axisArr = op.getAxis();
  TensorType inputTy = op.getInput().getType();
  std::optional<int64_t> maybeAxis = tryGetConstantInt(axisArr);
  if (!maybeAxis.has_value() || !inputTy.hasRank())
    return failure();

  int64_t axis = maybeAxis.value();
  if (axis >= 0)
    return failure();

  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  int64_t new_axis = axis + inputTy.getRank();
  auto newAxis = builder.create<ConstantOp>(op.getLoc(), new_axis);
  op->setOperand(operandIndex, newAxis);
  return success();
}

//===----------------------------------------------------------------------===//
// MO_MeanOp support
//===----------------------------------------------------------------------===//

LogicalResult MeanOp::canonicalize(MeanOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1);
}

//===----------------------------------------------------------------------===//
// MO_ReduceMulOp support
//===----------------------------------------------------------------------===//

LogicalResult ReduceMulOp::canonicalize(ReduceMulOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1);
}

OpFoldResult ReduceMulOp::fold(FoldAdaptor adaptor) {
  // TODO(#10811): support other types and do this using a MOJO kernel?

  auto inputArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInput());
  auto axisArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getAxis());
  if (!inputArr || !axisArr)
    return {};

  // We only support rank-1 integers (i.e. shape-like).
  if (inputArr.getType().getRank() != 1)
    return {};

  int64_t axis = getIntBlob(axisArr).front();
  if (axis != 0)
    return {};

  TensorType resTy = getType();
  auto maybeInput = tryGetConstantBlob<int64_t>(cast<ElementsAttr>(inputArr),
                                                resTy.getDtype());
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
// MO_ReduceAddOp support
//===----------------------------------------------------------------------===//

LogicalResult ReduceAddOp::canonicalize(ReduceAddOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1);
}

OpFoldResult ReduceAddOp::fold(FoldAdaptor adaptor) {
  auto axisArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getAxis());
  TensorType inputTy = getType();

  if (!axisArr || !inputTy.hasRank())
    return {};

  int64_t axis = getIntBlob(axisArr).front();
  int64_t rank = inputTy.getRank();
  axis = (axis + rank) % rank;

  // If we are reducing along a dimension of 1
  if (inputTy.getIntDim(axis) == 1)
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// MO_ReduceMinOp support
//===----------------------------------------------------------------------===//

LogicalResult ReduceMinOp::canonicalize(ReduceMinOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1);
}

//===----------------------------------------------------------------------===//
// MO_ReduceMaxOp support
//===----------------------------------------------------------------------===//

LogicalResult ReduceMaxOp::canonicalize(ReduceMaxOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1);
}

//===----------------------------------------------------------------------===//
// MO_ReshapeOp support
//===----------------------------------------------------------------------===//

template <typename T>
inline LogicalResult checkReshapeInOutTypesCompatible(T op) {
  // checks whether the input and output shapes are compatible
  // e.g. may have the same number of elements and same dtype.
  TensorType inputTy = op.getInput().getType();
  TensorType resultTy = op.getType();
  if (inputTy.getDtype() != resultTy.getDtype())
    return op.emitOpError("expect input and output dtypes to match");

  std::optional<int64_t> inputNumEles = inputTy.getNumElements();
  std::optional<int64_t> resultNumEles = resultTy.getNumElements();
  if (!inputNumEles.has_value() || !resultNumEles.has_value()) {
    // Here, at least one of inputTy or resultTy has parameters in the shape
    // expression. Assume you can always set some parameters to equal the other
    // side. e.g. 5 * N = 10, N could be 2 to make things valid, we can't check
    // deeper. Note: as parameters are integers, some expressions could
    // potentially be invalidated. E.g. 5 * N = 8 is clearly impossible for
    // integer N. However, for now, this is sufficient.
    return success();
  }

  if (inputNumEles.value() != resultNumEles.value()) {
    return op.emitOpError("input and output elements do not match: ")
           << inputNumEles.value() << " vs " << resultNumEles.value();
  }
  return success();
}

LogicalResult ReshapeOp::verify() {
  LogicalResult result = checkReshapeInOutTypesCompatible(*this);
  if (result.failed())
    return result;

  auto shapeInput = dyn_cast_or_null<ConstantOp>(getNewShape().getDefiningOp());
  if (!shapeInput)
    return success();

  ElementsAttr attr = shapeInput.getValue();
  auto dimIntegers =
      tryGetConstantBlob<int64_t>(attr, getNewShape().getType().getDtype());
  if (!dimIntegers.has_value())
    return success();

  int64_t requestedElements = 1;
  bool requestShapeHasNegOne = false;
  for (int64_t dim : dimIntegers.value()) {
    if (dim == -1)
      requestShapeHasNegOne = true;
    requestedElements *= dim;
  }

  std::optional<int64_t> resultElements =
      getResult().getType().getNumElements();
  if (resultElements.has_value() &&
      resultElements.value() != requestedElements && !requestShapeHasNegOne) {
    return emitOpError("Expected requested shape and output shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << resultElements.value();
  }

  std::optional<int64_t> inputElements = getInput().getType().getNumElements();
  if (inputElements.has_value() && inputElements.value() != requestedElements &&
      !requestShapeHasNegOne) {
    return emitOpError("Expected requested shape and input shape to have the "
                       "same number of elements requested ")
           << requestedElements << " but expected " << inputElements.value();
  }

  return success();
}

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
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  // [?] --> []
  if (inputTy.getRank() == 1 && resTy.getRank() == 0) {
    auto inferredResTy =
        TensorType::get(SmallVector<int64_t>({}), resTy.getElementType());
    auto newReshapeOp = rewriter.create<Static::ReshapeOp>(
        op.getLoc(), inferredResTy, inputVal);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newReshapeOp,
                                          op.getOutputParamDecls());
    return success();
  }

  if (resTy.hasUnknownShape() || !op.getOutputParamDecls().empty())
    return failure();

  rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, resTy, inputVal);
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

  // If we are preceded by ranked -> ranked rebind, we fold that into this op.
  if (auto rebindOp = inputVal.getDefiningOp<RebindOp>()) {
    TensorValue rebIn = rebindOp.getInput();
    TensorType rebInTy = rebIn.getType();
    if (rebInTy.hasRank() && !rebInTy.hasUnknownShape()) {
      op.setOperand(rebIn);
      return success();
    }
  }

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank == 0 && resRank == 1) {
    // We can fold static reshapes from scalars to singletons.
    if (TypedAttr param = getParamConstThrough<IndexToTensorOp>(inputVal)) {
      auto shape = ShapeAttr::get(SmallVector<TypedAttr>{param},
                                  rewriter.getType<ShapeType>());
      auto newConstOp =
          rewriter.create<KGEN::ParamConstantOp>(op->getLoc(), shape);
      rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, newConstOp);
      return success();
    }
  }
  if (inputRank == 1 && resRank == 0) {
    // We can fold static reshapes from singletons to scalars.
    if (TypedAttr param = getParamConstThrough<Shape::ToTensorOp>(inputVal)) {
      if (auto shape = dyn_cast<ShapeAttr>(param);
          shape && shape.getRank() == 1) {
        auto newConstOp = rewriter.create<KGEN::ParamConstantOp>(
            op->getLoc(), shape.getValues()[0]);
        rewriter.replaceOpWithNewOp<IndexToTensorOp>(op, resTy, newConstOp);
        return success();
      }
    }
  }

  if (inputRank != resRank)
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

LogicalResult Static::ReshapeOp::verify() {
  LogicalResult typesCompatible = checkReshapeInOutTypesCompatible(*this);
  if (typesCompatible.failed()) {
    return typesCompatible;
  }

  TensorType resultTy = getResult().getType();
  if (resultTy.hasRank()) {
    for (auto curResDim : resultTy.getTypedShapeAttr().getValues()) {
      auto intCurResDim = dyn_cast<IntegerAttr>(curResDim);
      if (intCurResDim && intCurResDim.getInt() == -1)
        return emitOpError("static reshape can not have inferred dimensions");
    }
  }

  return success();
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

OpFoldResult SliceOp::fold(FoldAdaptor adaptor) {
  // We must be able to see the steps are all `1` as a condition of folding.
  auto steps = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getStep());
  if (!steps)
    return {};

  TensorType inputTy = getInput().getType();

  // If the input and the output are the same and neither have ? in them then we
  // can maybe fold.
  if (getResult().getType() != inputTy || inputTy.hasUnknownShape())
    return {};

  // If all of the steps are `1` then this is an identity. We need to check this
  // as you could produce the same shape by walking backwards through the slice
  // i.e -1. So we must check they are 1 which means the only way to produce the
  // same shape as input is through start of 0 and end of -1. I.E an identity.
  auto stepsData = tryGetConstantBlob<int64_t>(cast<ElementsAttr>(steps),
                                               getStep().getType().getDtype());
  if (!stepsData.has_value())
    return {};

  for (int64_t step : *stepsData) {
    if (step != 1)
      return {};
  }

  return getInput();
}

//===----------------------------------------------------------------------===//
// MO_TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  auto maybePermVec = tryGetConstantIntVector(op.getPerm());
  if (!maybePermVec.has_value())
    return failure();

  if (!op.getType().hasRank())
    return failure();

  size_t opRank = op.getType().getRank();
  ArrayRef<int64_t> permVec(maybePermVec.value());
  auto identityPerm = llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
  if (llvm::equal(permVec, identityPerm)) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                          op.getOutputParamDecls());
    return success();
  }

  // Merge consecutive transposes by merging their constant permutations
  if (auto inputTransposeOp = op.getInput().getDefiningOp<TransposeOp>()) {
    auto maybeInputPermVec =
        tryGetConstantIntVector(inputTransposeOp.getPerm());
    if (!maybeInputPermVec.has_value())
      return failure();
    ArrayRef<int64_t> inputPermVec(maybeInputPermVec.value());
    // TODO(#14270) check `perm` must contain unique elements here
    // In general we might need a more principled way of erroring on invalid
    // inputs during constant folding.
    SmallVector<int64_t> mergedPerm = permute(inputPermVec, permVec);
    auto mergedPermVal = rewriter.create<ConstantOp>(op.getLoc(), mergedPerm);
    rewriter.replaceOpWithNewOp<TransposeOp>(
        op, op.getType(), inputTransposeOp.getInput(), mergedPermVal,
        op.getOutputParamDecls());
    return success();
  }

  // Transpose == Reshape if the relative order of non-1 dims remains the same
  // e.g. transpose from shape [8, 1, 1, 5, 1, 1, 1, 50] to
  // - [8, 1, 1, 5, 1, 1, 50, 1]
  // - [1, 1, 1, 8, 5, 50, 1, 1]
  // - [8, 5, 50, 1, 1, 1, 1, 1]
  // replacing with equivalent reshapes are all allowed
  int64_t indexLastNonOneDim = -1; // should always be increasing
  MOSH::ShapeAttr inputShape = op.getInput().getType().getTypedShapeAttr();
  bool meetsReshapeRewriteCriteria = inputShape != nullptr;
  for (int64_t inputDimIndex : permVec) {
    if (!meetsReshapeRewriteCriteria) {
      break;
    }
    int64_t dimInteger = inputShape.getIntDim(inputDimIndex);
    if (dimInteger == mlir::ShapedType::kDynamic) {
      // violates static reshape which only allows int dims
      meetsReshapeRewriteCriteria = false;
    } else if (dimInteger != 1) {
      if (inputDimIndex < indexLastNonOneDim)
        meetsReshapeRewriteCriteria = false;
      indexLastNonOneDim = inputDimIndex;
    }
  }
  if (meetsReshapeRewriteCriteria) {
    rewriter.replaceOpWithNewOp<Static::ReshapeOp>(op, op.getType(),
                                                   op.getInput());
    return success();
  }

  return failure();
}

LogicalResult TransposeOp::verify() {
  // The permutation has known shape iff the argument and result are
  // statically ranked.
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
// MO_TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::verify() {
  // The permutation has known shape iff the argument and result are
  // statically ranked.
  TensorType repeatsTy = getRepeats().getType();
  TensorType inputTy = getInput().getType();
  TensorType resTy = getResult().getType();
  if (!repeatsTy.hasStaticShape()) {
    if (inputTy.hasRank() || resTy.hasRank()) {
      return emitError("argument and result must be dynamically ranked if "
                       "repeats has unknown shape");
    }
    return success();
  }

  if (!inputTy.hasRank() || !resTy.hasRank()) {
    return emitError("argument and result must be statically ranked if "
                     "repeats has known shape");
  }

  if (inputTy.getRank() != resTy.getRank())
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)resTy.getRank() != repeatsTy.getIntDim(0))
    return emitError("input and result ranks do not match repeats size");
  return success();
}

//===----------------------------------------------------------------------===//
// Linear Algebra Ops
//===----------------------------------------------------------------------===//

LogicalResult LinAlg::BandPartOp::verify() {
  // TODO: Refactor this into a trait when we see another op that does this
  if (getInput().getType() != getResult().getType())
    return emitError("first operand and the result must have the same type");
  return success();
}

LogicalResult LinAlg::SolveOp::verify() {
  TensorType inType = getInput().getType();
  if (inType.hasStaticShape()) {
    ArrayRef<TypedAttr> inMatShape = inType.getDims().take_back(2);
    if (inMatShape[0] != inMatShape[1])
      return emitOpError("first operand must be a batch of square matrices");
  }
  return impl::checkMatmulCompatibility(getOperation(), getInput(), getResult(),
                                        getRhs());
}

OpFoldResult Shape::FromTensorOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getInput()))
    return ShapeAttr::get(input, ShapeType::get(getContext()));
  return {};
}

OpFoldResult Shape::ToTensorOp::fold(FoldAdaptor adaptor) {
  if (auto input = dyn_cast_or_null<ShapeAttr>(adaptor.getInput())) {
    TensorType resTy = getType();
    if (!resTy.hasStaticShape()) {
      OpBuilder builder(getOperation());
      builder.setInsertionPointAfter(getOperation());
      TensorType newTy = resTy.cloneWith(ShapeAttr::get(
          {(int64_t)input.getRank()}, builder.getType<ShapeType>()));
      Location loc = getLoc();
      auto newOp = builder.create<Shape::ToTensorOp>(loc, newTy, getInput());
      auto rebindOp = builder.create<RebindOp>(loc, resTy, newOp);
      getResult().replaceAllUsesExcept(rebindOp, rebindOp.getOperation());
      return rebindOp.getResult();
    }

    if (input.isConstant()) {
      auto outTy =
          RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
      SmallVector<APInt> res =
          llvm::map_to_vector(input.getIntDims(), [](int64_t dim) {
            return APInt(64, dim, /*isSigned=*/true);
          });
      return IntArrayElementsAttr::get(outTy, res);
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
