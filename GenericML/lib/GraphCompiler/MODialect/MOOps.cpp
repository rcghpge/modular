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
#include "GenericML/GraphCompiler/MODialect/MODevices.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHOps.h"
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/Permutation.h"
#include "Support/ML/RangeUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ODSSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <numeric>

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {
ParseResult parseParamDecls(OpAsmParser &p,
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

void printParamDecls(OpAsmPrinter &p, Operation *,
                     KGEN::ParamDeclArrayAttr paramDecls) {
  if (!paramDecls.empty()) {
    p << "<() -> ";
    printParamDecls(p, paramDecls);
    p << '>';
  }
}
} // namespace M::MO

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

  // TODO(#37571)
  while (isIdentityLike(inputOp))
    inputOp = inputOp->getOperand(0).getDefiningOp();

  // Handle concatenations of constant tensors.
  if (auto concatOp = dyn_cast_or_null<ConcatOp>(inputOp)) {
    for (Value operand : concatOp.getInputs()) {
      TensorValue tensorOperand = cast<TensorValue>(operand);
      SmallVector<ConstantOp> operandConstantOps =
          tryGetConstantOpsOfRank(tensorOperand, rank);
      if (operandConstantOps.empty())
        return {};
      llvm::append_range(constantOps, operandConstantOps);
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
  if (auto maybeFloatBlob = tryGetConstantOfRank<float>(val, 0))
    return maybeFloatBlob.value().front();
  return std::nullopt;
}

std::optional<double> M::MO::tryGetConstantDouble(TensorValue val) {
  if (auto maybeDoubleBlobl = tryGetConstantOfRank<double>(val, 0))
    return maybeDoubleBlobl.value().front();
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

std::optional<SmallVector<int64_t>>
M::MO::tryGetConstantIntVector(ListValue val) {
  auto listCreateOp = dyn_cast_or_null<ListCreateOp>(val.getDefiningOp());
  if (!listCreateOp)
    return std::nullopt;

  SmallVector<int64_t> vec;
  for (auto inp : listCreateOp->getOperands()) {
    auto moTensorVal = dyn_cast<TensorValue>(inp);
    if (!moTensorVal)
      return std::nullopt;
    auto maybeVal = tryGetConstantInt(moTensorVal);
    if (!maybeVal)
      return std::nullopt;
    vec.push_back((*maybeVal));
  }
  return vec;
}

std::optional<SmallVector<bool>>
M::MO::tryGetConstantBoolVector(TensorValue val) {
  if (val.getType().getElementType().isInteger(1))
    return tryGetConstantOfRank<bool>(val, 1);
  return std::nullopt;
}

/// Assume `val` is of `MO::TensorType`, try to the value of `MOSH::ShapeType`
/// that feeds into the creation of `val`.  Otherwise return null.
static Value tryGetShapeValue(Value val) {
  // Skip proceeding rebinds (there could be more than one).
  while (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto toTensorOp = val.getDefiningOp<Shape::ToTensorOp>())
    return toTensorOp.getInput();

  return {};
}

/// Assume `val` is of `MO::TensorType`, try to find the constant integers that
/// created `val`, and return a `ShapeAttr` whose values are those constant
/// integers. Otherwise return null.
static ShapeAttr tryGetConstantShapeAttr(Value val) {
  // Skip at most a single intermediate rebind
  if (auto rebindOp = val.getDefiningOp<RebindOp>())
    val = rebindOp.getInput();

  if (auto tensorVal = dyn_cast<TensorValue>(val)) {
    if (auto maybeIntVec = tryGetConstantIntVector(tensorVal)) {
      MLIRContext *ctx = val.getContext();
      auto shapeType = ShapeType::get(ctx);
      return ShapeAttr::get(maybeIntVec.value(), shapeType);
    }
  }
  return {};
}

Value M::MO::tryConvertToShapeVal(Value val, Location loc) {
  if (Value shapeVal = tryGetShapeValue(val))
    return shapeVal;

  if (ShapeAttr shapeAttr = tryGetConstantShapeAttr(val)) {
    // We don't insert via pattern rewriter because it might prevent rewriter
    // from reaching convergence (e.g., if this is called by LowerConcat, and
    // some other concat inputs aren't constant/shape)
    OpBuilder builder(val.getContext());
    builder.setInsertionPointAfterValue(val);
    return builder.create<MOSH::ParamToValueOp>(loc, shapeAttr);
  }

  return {};
}

ShapeAttr M::MO::tryGetShapeAttr(Value val) {
  if (Value shapeVal = tryGetShapeValue(val)) {
    if (auto paramToValueOp = shapeVal.getDefiningOp<MOSH::ParamToValueOp>())
      return dyn_cast<ShapeAttr>(paramToValueOp.getValue());

    return {};
  }

  return tryGetConstantShapeAttr(val);
}

TypedAttr M::MO::tryGetDimAttr(Value val) {
  auto tensorVal = dyn_cast<TensorValue>(val);
  if (!tensorVal)
    return {};

  if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(tensorVal))
    return param;

  std::optional<int64_t> maybeConst = tryGetConstantInt(tensorVal);
  if (!maybeConst.has_value())
    return {};

  OpBuilder builder(val.getContext());
  return builder.getIndexAttr(*maybeConst);
}

/// If `op`'s axis is constant and input rank is known, modify `op`'s `axis` to
/// be the canonical version, i.e., it's guaranteed to be in [0, rank).
/// NOTE this should be used as the first canonicalization, so other logic in
/// canonicalization will be able to utilize it.
LogicalResult
canonicalizeStaticNegativeIndicesInPlace(Reduction op, int operandIndex,
                                         PatternRewriter &rewriter) {
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
  rewriter.modifyOpInPlace(op, [&] { op->setOperand(operandIndex, newAxis); });
  return success();
}

void M::MO::synchronizeMetadataUpdates(GraphOp op) {
  MLIRContext *ctx = op.getContext();
  TypeRange newArgTypes = op.getBody().getArgumentTypes();
  SmallVector<Type> newResTypes;

  for (auto [newOutTy, funcOutTy] :
       llvm::zip(op.getOutputOp().getOperandTypes(), op.getResultTypes())) {
    // By default, assume nothing changed.
    Type newResTy = funcOutTy;

    if (auto funcOutTensorTy = dyn_cast<TensorType>(funcOutTy)) {
      auto newOutTensorTy = cast<TensorType>(newOutTy);
      DictionaryAttr newMetadata = newOutTensorTy.getMetadata();
      if (funcOutTensorTy.getMetadata() != newMetadata)
        newResTy = funcOutTensorTy.cloneWith(newMetadata);
    }

    newResTypes.push_back(newResTy);
  }

  auto funcType = FunctionType::get(ctx, newArgTypes, newResTypes);
  SignatureType signature = KGEN::SignatureType::remapToSignature(
      op.getInputParams(), op.getResultParams(), funcType);
  NamedAttrList attrs = op->getAttrDictionary();
  attrs.set(op.getSignatureAttrName(), TypeAttr::get(signature));
  attrs.set(op.getFunctionTypeAttrName(), TypeAttr::get(funcType));
  op->setAttrs(attrs.getDictionary(ctx));
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

LogicalResult TransferOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  ASSERT_STREAM(operands.size() == 1, << "mismatched operand arity");
  auto sourceType = dyn_cast<TensorType>(operands[0].getType());
  ASSERT_STREAM(sourceType, << "mismatched operand types");
  auto destDeviceAttr = dyn_cast_if_present<DeviceRefAttr>(
      attributes.get(getDestDeviceAttrName(name)));
  ASSERT_STREAM(destDeviceAttr, << "mismatched operand types");
  inferredReturnTypes.emplace_back(sourceType.cloneWith(destDeviceAttr));
  return success();
}

LogicalResult TransferOp::verify() {
  if (!getInput().getType().getDeviceRef())
    return emitError()
           << "expected input tensor type to have a #M.device_ref annotation";
  if (!getResult().getType().getDeviceRef())
    return emitError()
           << "expected result tensor type to have a #M.device_ref annotation";
  if (getResult().getType().getDeviceRef() != getDestDeviceAttr())
    return emitError() << "expected result tensor type to have same "
                          "#M.device_ref annotation as transfer destination ("
                       << getDestDeviceAttr() << " vs "
                       << getResult().getType().getDeviceRef() << ").";
  return success();
}

LogicalResult TransferOp::canonicalize(TransferOp outerOp,
                                       PatternRewriter &rewriter) {
  // transfer(transfer(input)) = input, provided devices line up.
  TypedValue<TensorType> intermediate = outerOp.getInput();
  auto innerOp = intermediate.getDefiningOp<TransferOp>();
  if (!innerOp)
    return failure();
  TypedValue<TensorType> input = innerOp.getInput();
  if (outerOp.getDestDeviceAttr() !=
      MO::getDeviceOrDefault(outerOp->getParentOfType<GraphOp>(),
                             input.getType()))
    return failure();

  rewriter.replaceOp(outerOp, input);
  return success();
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

LogicalResult IndexToTensorOp::canonicalize(IndexToTensorOp op,
                                            PatternRewriter &rewriter) {
  auto paramToValueOp = op.getInput().getDefiningOp<MOSH::ParamToValueOp>();
  if (!paramToValueOp)
    return failure();

  auto inputIntAttr = dyn_cast<IntegerAttr>(paramToValueOp.getValue());
  if (!inputIntAttr)
    return failure();

  SmallVector<APInt> res{inputIntAttr.getValue()};
  auto rankedResTy = RankedTensorType::get({}, op.getType().getElementType());
  auto elementsAttr =
      cast<ElementsAttr>(IntArrayElementsAttr::get(rankedResTy, res));
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
  return success();
}

//===----------------------------------------------------------------------===//
// MO_GraphOp support
//===----------------------------------------------------------------------===//

/// Name of optional attribute holding the #M.device_spec_collection describing
/// the (virtual) devices available to the graph.
constexpr StringRef kGraphDeviceSpecCollectionAttrName = "device_specs";

/// Traverse the given types and collect unique parameter references, then
/// return a set of parameter declaration for those references.
static llvm::SetVector<KGEN::ParamDeclAttr>
getParamDeclsFromShapes(TypeRange types) {
  llvm::SetVector<KGEN::ParamDeclAttr> uniqueDecls;
  for (Type type : types) {
    type.walk([&](KGEN::ParamDeclRefAttr declRef) {
      uniqueDecls.insert(KGEN::ParamDeclAttr::get(declRef));
    });
  }
  return uniqueDecls;
}

void GraphOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                    TypeRange inputTypes, TypeRange resultTypes) {
  // Below we basically inline and specialize `buildWithEntryBlock`, which
  // doesn't work because we use `signature` and not `function_type`.
  state.addAttribute(getNameAttrName(state.name), builder.getStringAttr(name));

  // Add signature attribute
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::SignatureType::get(funcType);
  state.addAttribute(getSignatureAttrName(state.name), TypeAttr::get(sig));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));

  // Declare input/result parameters, if any is used in the input/result types.
  MLIRContext *ctx = builder.getContext();
  auto inputDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(inputTypes).getArrayRef());
  auto resultDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(resultTypes).getArrayRef());
  state.addAttribute(getInputParamsAttrName(state.name), inputDecls);
  state.addAttribute(getResultParamsAttrName(state.name), resultDecls);

  // Add the function body.
  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

static ParseResult parseGraphOp(OpAsmParser &p, StringAttr &name,
                                TypeAttr &signatureAttr,
                                TypeAttr &functionTypeAttr,
                                KGEN::ParamDeclArrayAttr &inputParams,
                                KGEN::ParamDeclArrayAttr &resultParams,
                                NamedAttrList &attrs, Region &body) {
  if (p.parseSymbolName(name))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  KGEN::SignatureType signature;
  FunctionType functionType;
  if (KGEN::parseFunctionSignature(p, args, inputParams, resultParams,
                                   functionType, signature))
    return failure();
  signatureAttr = TypeAttr::get(signature);
  functionTypeAttr = TypeAttr::get(functionType);

  if (p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();
  return success();
}

static void printGraphOp(OpAsmPrinter &p, Operation *op, StringAttr name,
                         TypeAttr signature, TypeAttr functionType,
                         KGEN::ParamDeclArrayAttr inputParams,
                         KGEN::ParamDeclArrayAttr resultParams,
                         DictionaryAttr attrs, Region &body) {
  p.printSymbolName(name);
  printFunctionSignature(p, &body, inputParams, resultParams,
                         cast<FunctionType>(functionType.getValue()),
                         cast<SignatureType>(signature.getValue()));

  auto graph = cast<GraphOp>(op);
  SmallVector<StringRef, 6> elidedAttrs{
      graph.getNameAttrName(), graph.getSignatureAttrName(),
      graph.getFunctionTypeAttrName(), graph.getInputParamsAttrName(),
      graph.getResultParamsAttrName()};
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

LogicalResult GraphOp::verify() {
  mlir::SymbolTableAnalysis symTabAnalysis(getOperation()->getParentOp());
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

KGEN::ParamDeclAttr GraphOp::getNewDimParam(const Twine &prefix) {
  return getNewParamOfType(prefix, IndexType::get(getContext()));
}

KGEN::ParamDeclAttr GraphOp::getNewShapeParam(const Twine &prefix) {
  return getNewParamOfType(prefix, ShapeType::get(getContext()));
}

KGEN::ParamDeclAttr GraphOp::getNewParamOfType(const Twine &prefix, Type type) {
  return KGEN::ParamDeclAttr::get(getNewParamName(prefix), type);
}

std::string GraphOp::getNewParamName(const Twine &prefix) {
  // TODO(#8829) avoid duplication with existing parameters in the input graph
  int64_t newCounter = getProperties().getCounter() + 1;
  getProperties().setCounter(newCounter);
  return (prefix + Twine(newCounter)).str();
}

DeviceSpecCollectionAttr GraphOp::getDeviceSpecs() {
  return dyn_cast_if_present<DeviceSpecCollectionAttr>(
      getOperation()->getAttr(kGraphDeviceSpecCollectionAttrName));
}

void GraphOp::setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs) {
  getOperation()->setAttr(kGraphDeviceSpecCollectionAttrName, deviceSpecs);
}

//===----------------------------------------------------------------------===//
// MO_OutputOp support
//===----------------------------------------------------------------------===//

/// Parse the output values for `mo.output`, sugaring `?` for unknown values.
static ParseResult parseOutputValues(AsmParser &p,
                                     KGEN::ParameterExprArrayAttr &valuesAttr) {
  SmallVector<TypedAttr> values;
  auto parseFn = [&]() -> ParseResult {
    Type type = p.getBuilder().getIndexType();
    if (succeeded(p.parseOptionalColon()) && KGEN::parseKGENType(p, type))
      return failure();
    if (succeeded(p.parseOptionalQuestion())) {
      values.push_back(KGEN::UnknownAttr::get(type));
      return success();
    }
    return KGEN::parseParamValue(p, values.emplace_back(), type);
  };
  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::OptionalLessGreater,
                                parseFn))
    return failure();
  valuesAttr = ParameterExprArrayAttr::get(p.getContext(), values);
  return success();
}

/// Print the output values for `mo.output`, sugaring `?` for unknown values.
static void printOutputValues(AsmPrinter &p, Operation *op,
                              ArrayRef<TypedAttr> values) {
  if (values.empty())
    return;
  p << '<';
  llvm::interleaveComma(values, p, [&](TypedAttr value) {
    if (Type type = value.getType(); !type.isIndex()) {
      p << ':';
      KGEN::printKGENType(p, type);
      p << ' ';
    }
    if (isa<KGEN::UnknownAttr>(value))
      p << '?';
    else
      KGEN::printParamValue(p, value);
  });
  p << '>';
}

void OutputOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {
  auto graph = cast<GraphOp>(getOperation()->getParentOp());
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParams();
  ArrayRef<TypedAttr> defs = this->getParameters();

  // Doesn't make sense to walk, graph is in a bad state.
  if (decls.size() != defs.size()) {
    emitError("[OutputOp::walkDefinitions] output bindings and result "
              "parameters must have the same size");
    return;
  }

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
// IfOp
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
  auto thenTerminator = thenBlock->getTerminator();
  if (!isa<YieldOp>(thenTerminator))
    return emitOpError("'then' block expected to be terminated by a YieldOp");
  auto thenOutput = cast<YieldOp>(thenTerminator);
  if (failed(checkOperandTypes(thenOutput, getResultTypes())))
    return failure();

  // The 'else' block results must match the mo.if results.
  auto elseTerminator = elseBlock->getTerminator();
  if (!isa<YieldOp>(elseTerminator))
    return emitOpError("'else' block expected to be terminated by a YieldOp");
  auto elseOutput = cast<YieldOp>(elseTerminator);
  return checkOperandTypes(elseOutput, getResultTypes());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> argumentInfo;
  Region *condRegion = result.addRegion();
  Region *bodyRegion = result.addRegion();

  if (parseParenOperandListWithShadowing(parser, result, argumentInfo,
                                         /*isIsolatedFromAbove=*/false) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parseRegion(parser, argumentInfo, *condRegion,
                  /*isIsolatedFromAbove=*/false) ||
      parser.parseKeyword("do") ||
      parseRegion(parser, argumentInfo, *bodyRegion,
                  /*isIsolatedFromAbove=*/false))
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
                                     getCondRegion().getArguments(),
                                     /*isIsolatedFromAbove=*/false);
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  printRegion(p, getInputs(), getCondRegion(), /*isIsolatedFromAbove=*/false);
  p << " do ";
  printRegion(p, getInputs(), getBodyRegion(), /*isIsolatedFromAbove=*/false);
}

/// Returns success if yieldOp's single operand is of the MO scalar boolean
/// type.
static LogicalResult outputsOneBool(YieldOp yieldOp) {
  if (yieldOp->getNumOperands() != 1)
    return failure();
  auto tensorType = dyn_cast<TensorType>(yieldOp->getOperand(0).getType());
  if (!tensorType)
    return failure();
  if (!tensorType.hasRank() || tensorType.getRank() != 0)
    return failure();
  if (tensorType.getElementType() != IntegerType::get(yieldOp->getContext(), 1))
    return failure();
  return success();
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
  // We don't construct a scalar bool type and use checkOperandTypes because
  // it is valid for the type to have additional type attributes (such as
  // for a storage device).
  auto condTerminator = condBlock->getTerminator();
  if (!isa<YieldOp>(condTerminator))
    return emitOpError("cond block expected to be terminated by a YieldOp");
  auto condOutput = cast<YieldOp>(condTerminator);
  if (failed(outputsOneBool(condOutput)))
    return condOutput->emitOpError()
           << "expected single operand of type '!mo.tensor<[], bool>'.";

  // The 'body' block returns a result for each mo.while result.
  auto bodyTerminator = bodyBlock->getTerminator();
  if (!isa<YieldOp>(bodyTerminator))
    return emitOpError("body block expected to be terminated by a YieldOp");
  auto bodyOutput = cast<YieldOp>(bodyTerminator);
  if (failed(checkOperandTypes(bodyOutput, getResultTypes())))
    return failure();

  // It would be nice to check that no lhs SSA value of an 'as clause
  // appears free in the body. However, common passes, such as
  // canonicalization, may replace locally bound SSA values with outer SSA
  // values which also happen to be passed as inputs, which would violates
  // this rule.

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

LogicalResult BroadcastToOp::canonicalize(BroadcastToOp op,
                                          PatternRewriter &rewriter) {
  TensorValue inputVal = op.getInput();
  TensorType inputTy = inputVal.getType();
  TensorType resTy = op.getType();

  // If we have a known input and an known output with no parameter decls then
  // we can convert this to static.
  if (!resTy.hasUnknownShape() && !inputTy.hasUnknownShape() &&
      op.getOutputParamDecls().empty()) {
    rewriter.replaceOpWithNewOp<Static::BroadcastToOp>(op, op.getType(),
                                                       inputVal);
  }

  return failure();
}

void BroadcastToOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TensorValue input, TensorValue broadcastLike,
                          TensorValue broadcastShape) {
  TensorType resultTy =
      input.getType().cloneWith(broadcastLike.getType().getShapeAttr());

  // Note broadcastShape may be null
  if (!broadcastShape)
    broadcastShape =
        odsBuilder.create<MO::ShapeOfOp>(input.getLoc(), broadcastLike);

  build(odsBuilder, odsState, resultTy, input, broadcastShape);
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
    // Be careful not to drop any non-shape info like metadata in TensorType.
    if (resTy == inputTy)
      rewriter.replaceAllUsesExcept(op, inputVal, op);
    else
      rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, inputVal);

    return success();
  }

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

LogicalResult Static::BroadcastToOp::verify() {
  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), getType()))
    return emitError(err.getError());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp support
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
    return success();
  }
  return failure();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  ASSERT_STREAM(operands.empty(), << "constant has no operands");
  return getValueAttr();
}

LogicalResult ConstantOp::verify() {
  auto valType = cast<ShapedType>(getValue().getType()).getElementType();
  auto resType = getResult().getType().getElementType();
  // TODO(#28523): Restrict the value type of booleans to i8.
  if (resType.isInteger(1)) {
    if (!(valType.isInteger(1) || valType.isInteger(8) ||
          valType.isInteger(32) || valType.isInteger(64))) {
      return emitOpError("for boolean tensors, the element type of the value "
                         "must be an integer type, but got ")
             << valType;
    }
    return success();
  }
  if (valType != resType) {
    llvm::errs() << " not equal emitting error\n";
    return emitOpError("element type of the value (")
           << valType << ") must match the element type of the result ("
           << resType << ")";
  }
  return success();
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
  auto rankedAttrTy =
      RankedTensorType::get(values.size(), odsBuilder.getI8Type());
  SmallVector<APInt> APIntVals =
      llvm::map_to_vector(values, [&](bool val) { return APInt(8, val); });
  auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, APIntVals);
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
  auto rankedAttrTy = RankedTensorType::get(1, odsBuilder.getI8Type());
  auto valueAttr = IntArrayElementsAttr::get(
      rankedAttrTy, ArrayRef<APInt>(APInt(8, value, /*isSigned=*/false)));
  build(odsBuilder, odsState, type, cast<ElementsAttr>(valueAttr));
}

bool ConstantOp::preferResultsDevice() { return true; }

//===----------------------------------------------------------------------===//
// MO_ConstantScalarOp support
//===----------------------------------------------------------------------===//

OpFoldResult ConstantScalarOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  ASSERT_STREAM(operands.empty(), << "constant.scalar has no operands");
  return getValueAttr();
}

LogicalResult ConstantScalarOp::verify() {
  auto scalarDType = cast<MO::ScalarType>(getType()).getDType();
  if (auto value = dyn_cast<IntegerAttr>(getValue())) {
    if (!isa<IntegerType>(scalarDType))
      return emitError(
          "mo.constant.scalar expected the dtype to be an integer");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  } else if (auto value = dyn_cast<FloatAttr>(getValue())) {
    if (!isa<FloatType>(scalarDType))
      return emitError("mo.constant.scalar expected the dtype to be a float");
    if (value.getType().getIntOrFloatBitWidth() !=
        scalarDType.getIntOrFloatBitWidth())
      return emitError("mo.constant.scalar expected bitwidth of the dtype and "
                       "attribute to match");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult ConcatOp::fold(FoldAdaptor adaptor) {
  TensorType outType = getResult().getType();

  Value axis = getAxis();

  SplitOp split = nullptr;

  // If the concat is the sole product of a split along the same axis we can
  // fold and just use the split input.
  size_t operandIndex = 0;
  for (Value operand : getInputs()) {
    auto definingOp = operand.getDefiningOp<MO::SplitOp>();
    if (!definingOp) {
      split = nullptr;
      break;
    }

    // Check if we have a match, all operands must have the same defining op.
    if (!split) {
      split = definingOp;
    } else if (split != definingOp) {
      split = nullptr;
      break;
    }

    // Split must have same axis.
    if (definingOp.getAxis() != axis) {
      split = nullptr;
      break;
    }

    if (cast<OpResult>(operand).getResultNumber() != operandIndex) {
      split = nullptr;
      break;
    }

    // Each split operand must be in the same order as they appear here.
    operandIndex++;
  }

  // If we are a concat followed by split we can fold and use the split input
  // directly.
  if (split && split.getInput().getType() == outType &&
      split.getNumResults() == getInputs().size())
    return split.getInput();

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
  if (numInputs < 1)
    return emitError("concat must have at least 1 input tensor");

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

  if (!op.getOutputParamDecls().empty())
    return Error("Cannot fold concat op with param decls.");

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

DeviceRefAttr ConcatOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// MO_BandPartOp support
//===----------------------------------------------------------------------===//

DeviceRefAttr LinAlg::BandPartOp::getOperandDevice(size_t index) {
  return (index == getNumLowerMutable().getOperandNumber() ||
          index == getNumUpperMutable().getOperandNumber() ||
          index == getExcludeMutable().getOperandNumber())
             ? MO::getHostDevice(getOperation())
             : nullptr;
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
    int64_t normalizedAxis = normalizeAxis(axis, inputRank);
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
  // If the split is splitting into one tensor then it is identity.
  if (getNumResults() == 1 && getInput().getType() == getResult(0).getType()) {
    results.push_back(getInput());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// GatherOp support
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
    // DType of empty array must match
    return getEmptyArray(getResult().getType().getElementType());

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

DeviceRefAttr GatherOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
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

  // TODO(#10811): support other types and do this using a MOJO kernel? Waiting
  // on #23197
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
          intElemsAttr.getValues(), [&](const llvm::APInt &dim) {
            if (inTy.getDtype().isSInt())
              return APInt(64, dim.getSExtValue(), /*isSigned=*/true);

            // Else is unsigned integer.
            return APInt(64, dim.getZExtValue(), /*isSigned=*/false);
          });

      return IntArrayElementsAttr::get(rankedResTy, res);
    }
  }

  return {};
}

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
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
// MO_ArgNonzeroOp support
//===----------------------------------------------------------------------===//

LogicalResult ArgNonzeroOp::verify() {
  TensorType resTy = getResult().getType();
  TensorType inputTy = getInput().getType();

  if (!resTy.hasRank() || resTy.getRank() != 2)
    return emitError("mo.arg_nonzero: invalid output shape: rank must be 2");

  if ((inputTy.hasRank() && (int64_t)inputTy.getRank() > 0 &&
       resTy.getIntDim(1) != (int64_t)inputTy.getRank()) ||
      (!inputTy.hasRank() && !ShapedType::isDynamic(resTy.getIntDim(1))))
    return emitError(
        "mo.arg_nonzero: invalid output shape: inner_dim != input_rank");

  return success();
}

//===----------------------------------------------------------------------===//
// MO::ArgMaxOp support
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult
verifyWithRankedInput(Op *op, int64_t axis, int64_t resDim, TensorType inputTy,
                      size_t &dynamicOutDims, size_t &reducedDims) {
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
}

static void verifyWithUnrankedInput(int64_t axis, int64_t resDim,
                                    size_t &dynamicOutDims,
                                    size_t &reducedDims) {
  if (mlir::ShapedType::isDynamic(resDim)) {
    dynamicOutDims++;
    return;
  }
  if (resDim == 1)
    reducedDims++;
}

template <typename Op>
static LogicalResult argVerify(Op *op) {
  static_assert(std::is_same_v<Op, ArgMaxOp> || std::is_same_v<Op, ArgMinOp>,
                "input op must be either an ArgMax or ArgMin op");

  TensorType resTy = op->getResult().getType();
  TensorType inputTy = op->getInput().getType();
  TensorType axisTy = op->getAxis().getType();

  // TODO(#12939): Refactor this into verifyReduction.

  if (!axisTy.getElementType().isSignedInteger(32) &&
      !axisTy.getElementType().isSignedInteger(64))
    return op->emitOpError("axis type must be si32 or si64");

  if ((inputTy.hasRank() && resTy.hasRank()) &&
      (inputTy.getRank() != resTy.getRank()))
    return op->emitOpError(Op::getOperationName() +
                           ": output rank must match input rank");

  if (resTy.hasRank()) {
    auto optAxis = tryGetConstantInt(op->getAxis());
    size_t reducedDims = 0;
    size_t dynamicOutDims = 0;

    for (int64_t axis = 0, rank = resTy.getRank(); axis < rank; axis++) {
      int64_t resDim = resTy.getTypedShapeAttr().getIntDim(axis);

      if (optAxis.has_value() && (axis == optAxis.value()) && (resDim != 1) &&
          !mlir::ShapedType::isDynamic(resDim))
        return op->emitOpError(
            Op::getOperationName() +
            ": output dimensions must be 1 at the reduction axis");

      // Note: we don't check that non-reduction axes match directly. The code
      // below does it, which handles the dynamic cases as well.

      if (inputTy.hasRank()) {
        if (failed(verifyWithRankedInput(op, axis, resDim, inputTy,
                                         dynamicOutDims, reducedDims)))
          return failure();
      } else {
        verifyWithUnrankedInput(axis, resDim, dynamicOutDims, reducedDims);
      }
    }

    if ((reducedDims > 1) || (!reducedDims && !dynamicOutDims)) {
      return op->emitOpError(
          Op::getOperationName() +
          ": output dimensions must match input "
          "dimensions, except for exactly one dimension, the reduction axis");
    }
  }

  return success();
}

template <typename TOp>
void buildReductionOp(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, TensorValue axis,
                      TensorType outputTy) {
  if (input.getType().hasRank() && !outputTy) {
    outputTy = input.getType().cloneWithUnknownDims(input.getType().getRank());
  } else if (!outputTy) {
    outputTy = input.getType().cloneWithUnknownRank();
  }

  TOp::build(odsBuilder, odsState, outputTy, input, axis);
}

template <typename TOp>
void buildReductionOp(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<TOp>(odsBuilder, odsState, input,
                        odsBuilder.create<ConstantOp>(input.getLoc(), axis),
                        outputTy);
}

LogicalResult ArgMaxOp::canonicalize(ArgMaxOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

LogicalResult ArgMaxOp::verify() { return argVerify(this); }

void ArgMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, TensorValue axis, TensorType outputTy) {
  buildReductionOp<ArgMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ArgMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// MO::ArgMinOp
//===----------------------------------------------------------------------===//

LogicalResult ArgMinOp::canonicalize(ArgMinOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

LogicalResult ArgMinOp::verify() { return argVerify(this); }

void ArgMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, TensorValue axis, TensorType outputTy) {
  buildReductionOp<ArgMinOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ArgMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                     TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ArgMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// MO::CumSumOp
//===----------------------------------------------------------------------===//

LogicalResult CumSumOp::verify() {
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (inputType.getDtype() != resultType.getDtype())
    return emitOpError("Expect input and result DTypes to be the same.");

  auto inDimAttrs = dyn_cast<ShapeAttr>(inputType.getShapeAttr());
  auto outDimAttrs = dyn_cast<ShapeAttr>(resultType.getShapeAttr());
  if (inDimAttrs && outDimAttrs) {
    for (auto [inDimAttr, outDimAttr] :
         llvm::zip(inDimAttrs.getValues(), outDimAttrs.getValues())) {
      auto inDimIntAttr = dyn_cast<IntegerAttr>(inDimAttr);
      auto outDimIntAttr = dyn_cast<IntegerAttr>(outDimAttr);
      if (inDimIntAttr && outDimIntAttr && inDimIntAttr != outDimIntAttr)
        return emitError("statically known dimensions must match");
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

LogicalResult AvgPoolCeilOp::verify() {
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

LogicalResult MaxPoolCeilOp::verify() {
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

LogicalResult ConvTransposeOp::verify() {
  // The sliding window interface already checks that the input and result
  // have matching ranks, so we can be more specific with the error message
  // here.
  if (failed(impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes(
      {getDilations(), getStrides(), getOutputPaddings()});
}

//===----------------------------------------------------------------------===//
// Matmul definitions.
//===----------------------------------------------------------------------===//
void BatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type resultTy, Value input_a, Value input_b) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({resultTy});
}

void BatchMatmulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Type resultTy, Value input_a, Value input_b,
                          ArrayRef<KGEN::ParamDeclAttr> decls) {
  odsState.addOperands(input_a);
  odsState.addOperands(input_b);
  odsState.addTypes({resultTy});
  if (!decls.empty()) {
    odsState.addAttribute(
        getOutputParamDeclsAttrName(odsState.name),
        KGEN::ParamDeclArrayAttr::get(odsBuilder.getContext(), decls));
  }
}

LogicalResult BatchMatmulOp::verify() {
  TensorType aType = getTensorA().getType();
  TensorType bType = getTensorB().getType();
  TensorType cType = getTensorResult().getType();

  // Note MO_MatmulLike interface verifies ranks are consistent between ops.
  // Ensure all ranked tensors are rank >= 3
  auto violatedRankProperty = [&](TensorType ty) -> bool {
    return ty.hasRank() && ty.getRank() < 3;
  };
  if (violatedRankProperty(aType) || violatedRankProperty(bType) ||
      violatedRankProperty(cType)) {
    return emitError("All operands and results must be of at least rank 3.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MO_Pad.*Op support
//===----------------------------------------------------------------------===//

LogicalResult canonicalizePadLikeOp(PadLike op, PatternRewriter &rewriter) {
  // Try to extract constant paddings
  auto maybePaddings = tryGetConstantIntVector(op.getPaddings());
  if (!maybePaddings.has_value())
    return failure();
  ArrayRef<int64_t> paddings = maybePaddings.value();

  // If paddings are all 0s, the pad operation is identity. We replace it with a
  // rebind from the input because the output shape of `mo.pad` might not always
  // equal to the input shape, e.g., it could be fully dynamic from the
  // frontends, e.g.,
  //     mo.pad(input, zeros, val) : ... -> tensor<[?, ?, ?], f32>
  if (llvm::all_of(paddings, [](int64_t n) { return n == 0; })) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                          op.getOutputParamDecls());
    return success();
  }

  return failure();
}

LogicalResult PadConstantOp::canonicalize(PadConstantOp op,
                                          PatternRewriter &rewriter) {
  return canonicalizePadLikeOp(op, rewriter);
}

LogicalResult PadReflectOp::canonicalize(PadReflectOp op,
                                         PatternRewriter &rewriter) {
  return canonicalizePadLikeOp(op, rewriter);
}

LogicalResult PadRepeatOp::canonicalize(PadRepeatOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizePadLikeOp(op, rewriter);
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
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
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
      rewriter.modifyOpInPlace(op, [&] { op.setOutputParamDecls({}); });
      return success();
    }

    if (auto inputDimAttrs = dyn_cast<ShapeAttr>(inputShapeAttr)) {
      // Example:
      //   rebind<() -> D4> : [D0, D1, D2] -> [D3, D4, 42]
      // --->
      //   kgen.param.declare D4 = <D1>
      //   rebind : [D0, D1, D2] -> [D0, D4, 42]
      llvm::DenseSet<KGEN::ParamDeclAttr> unknownParamDecls;

      // Avoid redeclaring multiple parameters. Example:
      //  rebind<() -> C> : [A, B] -> [C, C]
      // --->
      //   kgen.param.declare C = <A>
      //   rebind : [A, B] -> [C, C]
      llvm::SmallSetVector<KGEN::ParamDeclAttr, 4> declaredParams;
      for (auto [inputDimAttr, outputDimAttr] :
           llvm::zip(inputDimAttrs.getValues(), outputDimAttrs.getValues())) {
        if (auto outputDimParamRef =
                dyn_cast<KGEN::ParamDeclRefAttr>(outputDimAttr)) {
          if (KGEN::ParamDeclAttr paramDecl =
                  nameToParamDecl.lookup(outputDimParamRef.getName())) {
            if (isa<KGEN::UnknownAttr>(inputDimAttr)) {
              unknownParamDecls.insert(paramDecl);
            } else if (!declaredParams.contains(paramDecl)) {
              declaredParams.insert(paramDecl);
              rewriter.create<KGEN::ParamDeclareOp>(loc, paramDecl,
                                                    inputDimAttr);
            }
          }
        }
      }
      if (unknownParamDecls.size() == paramDecls.size())
        return failure();
      rewriter.modifyOpInPlace(op, [&] {
        op.setOutputParamDecls(SmallVector<KGEN::ParamDeclAttr>(
            unknownParamDecls.begin(), unknownParamDecls.end()));
      });
      return success();
    }
  }

  // TODO: support materialization for other rebinds
  return failure();
}

LogicalResult RebindOp::canonicalize(RebindOp op,
                                     PatternRewriter &origRewriter) {
  DeviceAwareRewriter rewriter(op, origRewriter);

  // We fold if the result and operand types are the same.
  if (op.getInput().getType() == op.getType()) {
    assert(op.getOutputParamDecls().empty());
    rewriter.replaceOp(op, op.getInput());
    return success();
  }

  // We materialize the parameter declarations explicitly.
  LogicalResult paramsResolved = materializeShapeInference(op, origRewriter);

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
// MeanOp
// ===----------------------------------------------------------------------===//

LogicalResult MeanOp::canonicalize(MeanOp op, PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

void MeanOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, TensorValue axis, TensorType outputTy) {
  buildReductionOp<MeanOp>(odsBuilder, odsState, input, axis, outputTy);
}
void MeanOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<MeanOp>(odsBuilder, odsState, input, axis, outputTy);
}

DeviceRefAttr MeanOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

LogicalResult RandomNormalOp::verify() {
  auto meanTy = getMean().getType();
  auto varTy = getVariance().getType();
  if (meanTy.getElementType() != varTy.getElementType())
    return emitOpError("Expect mean and variance DTypes to be the same");

  return success();
}

LogicalResult RandomNormalOp::canonicalize(RandomNormalOp op,
                                           PatternRewriter &rewriter) {
  TensorType resTy = op.getType();
  if (!resTy.hasUnknownShape() && op.getOutputParamDecls().empty()) {
    rewriter.replaceOpWithNewOp<Static::RandomNormalOp>(
        op, op.getType(), op.getMean(), op.getVariance(), op.getSeed());
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ReduceMulOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceMulOp::canonicalize(ReduceMulOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

void ReduceMulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMulOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceMulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMulOp>(odsBuilder, odsState, input, axis, outputTy);
}

DeviceRefAttr ReduceMulOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// ReduceAddOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceAddOp::canonicalize(ReduceAddOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

OpFoldResult ReduceAddOp::fold(FoldAdaptor adaptor) {
  auto axisArr = dyn_cast_or_null<IntArrayElementsAttr>(adaptor.getAxis());
  TensorType inputTy = getInput().getType();

  if (!axisArr || !inputTy.hasRank())
    return {};

  int64_t axis = getIntBlob(axisArr).front();
  int64_t rank = inputTy.getRank();
  axis = (axis + rank) % rank;

  // If we are reducing along a dimension of 1
  if (inputTy.getIntDim(axis) == 1 &&
      getInput().getType() == getResult().getType())
    return getInput();
  return {};
}

void ReduceAddOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceAddOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceAddOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceAddOp>(odsBuilder, odsState, input, axis, outputTy);
}

DeviceRefAttr ReduceAddOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// ReduceMinOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceMinOp::canonicalize(ReduceMinOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

void ReduceMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

void ReduceMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

DeviceRefAttr ReduceMinOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// ReduceMaxOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceMaxOp::canonicalize(ReduceMaxOp op,
                                        PatternRewriter &rewriter) {
  return canonicalizeStaticNegativeIndicesInPlace(op, 1, rewriter);
}

void ReduceMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}

DeviceRefAttr ReduceMaxOp::getOperandDevice(size_t index) {
  return index == getAxisMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// TopK / BottomK
//===----------------------------------------------------------------------===//

template <typename OrderKOpT>
LogicalResult verifyOrderedKOp(OrderKOpT op) {
  auto output1Shape = op.getValues().getType().getShapeAttr();
  auto output2Shape = op.getIndices().getType().getShapeAttr();
  if (output1Shape != output2Shape) {
    return op.emitOpError(
               "values and indices should have the same shape but got: ")
           << output1Shape << " vs " << output2Shape << "\n";
  }
  return success();
}

LogicalResult TopKOp::verify() { return verifyOrderedKOp(*this); }

LogicalResult BottomKOp::verify() { return verifyOrderedKOp(*this); }

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

template <typename T>
inline LogicalResult checkReshapeInOutTypesCompatible(T op) {
  // checks whether the input and output shapes are compatible
  // e.g. may have the same number of elements and same dtype.
  TensorType inputTy = op.getInput().getType();
  TensorType resultTy = op.getType();

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

bool ReshapeOp::preferResultsDevice() { return true; }

//===----------------------------------------------------------------------===//
// MO_UnsqueezeShapeOp support
//===----------------------------------------------------------------------===//
LogicalResult UnsqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  // Check indices don't point to the same index
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getPaddingIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    llvm::sort(indices);
    for (size_t i = 0; i < indices.size() - 1; i++) {
      if (indices[i] == indices[i + 1])
        return emitOpError("Padding indices pad the same dimension twice");
    }
  }

  // Check out number of elements equal to sum of size of input elements
  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();
  std::optional<int64_t> paddingIndicesSize =
      getPaddingIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize.has_value() && paddingIndicesSize.has_value() &&
      outputSize.has_value() &&
      inputShapeSize.value() + paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the sum of the input sizes");
  }

  return success();
}

LogicalResult UnsqueezeShapeOp::canonicalize(UnsqueezeShapeOp op,
                                             PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getPaddingIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();

  SmallVector<int64_t> &indices = maybeIndices.value();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    if (index < 0) {
      index += inputShapeType.getNumElements().value() + indices.size();
      changedIndices = true;
    }
  }

  bool isSorted = true;
  for (size_t i = 0; i < indices.size() - 1 && isSorted; i++)
    isSorted = indices[i] <= indices[i + 1];
  if (!isSorted) {
    llvm::sort(indices);
    changedIndices = true;
  }

  if (!changedIndices)
    return failure();

  // Construct new op with same dtypes for indices
  TensorValue newIndices = rewriter.create<ConstantOp>(op->getLoc(), indices);
  newIndices = rewriter.create<CastOp>(op->getLoc(), newIndices,
                                       indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<UnsqueezeShapeOp>(op, op.getResult().getType(),
                                                op.getInputShape(), newIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// MO_SqueezeShapeOp support
//===----------------------------------------------------------------------===//

LogicalResult SqueezeShapeOp::verify() {
  if (getInputShape().getType().getDtype() != getResult().getType().getDtype())
    return emitOpError("Expect in and out DTypes to be the same type");

  // Check indices don't point to the same index
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(getRemoveIndices());
  if (maybeIndices) {
    SmallVector<int64_t> &indices = maybeIndices.value();
    llvm::sort(indices);

    // Note: canonicalize will change negative indices into positive if possible
    // which will catch the case where negative and positive indices point to
    // the same index.
    for (size_t i = 0, e = indices.size() - 1; i < e; ++i)
      if (indices[i] == indices[i + 1])
        return emitOpError("Padding indices pad the same dimension twice");
  }

  // Check number of elements equal to difference of input element sizes
  std::optional<int64_t> inputShapeSize =
      getInputShape().getType().getNumElements();
  std::optional<int64_t> paddingIndicesSize =
      getRemoveIndices().getType().getNumElements();
  std::optional<int64_t> outputSize = getResult().getType().getNumElements();
  if (inputShapeSize && paddingIndicesSize && outputSize &&
      inputShapeSize.value() - paddingIndicesSize.value() !=
          outputSize.value()) {
    return emitOpError("Output size is not the difference of the input sizes");
  }

  return success();
}

LogicalResult SqueezeShapeOp::canonicalize(SqueezeShapeOp op,
                                           PatternRewriter &rewriter) {
  // Normalize negative indices and make result in sorted order
  TensorValue indicesValue = op.getRemoveIndices();
  TensorType inputShapeType = op.getInputShape().getType();
  std::optional<SmallVector<int64_t>> maybeIndices =
      tryGetConstantIntVector(indicesValue);
  if (!maybeIndices || !inputShapeType.getNumElements().has_value())
    return failure();

  SmallVector<int64_t> &indices = maybeIndices.value();
  bool changedIndices = false;
  for (int64_t &index : indices) {
    if (index < 0) {
      index += inputShapeType.getNumElements().value();
      changedIndices = true;
    }
  }

  bool isSorted = true;
  for (size_t i = 0, e = indices.size() - 1; i < e; ++i)
    isSorted = indices[i] <= indices[i + 1];
  if (!isSorted) {
    llvm::sort(indices);
    changedIndices = true;
  }

  if (!changedIndices)
    return failure();

  // Construct new op with same dtypes for indices
  TensorValue newIndices = rewriter.create<ConstantOp>(op->getLoc(), indices);
  newIndices = rewriter.create<CastOp>(op->getLoc(), newIndices,
                                       indicesValue.getType().getDtype());
  rewriter.replaceOpWithNewOp<SqueezeShapeOp>(op, op.getResult().getType(),
                                              op.getInputShape(), newIndices);
  return success();
}

//===----------------------------------------------------------------------===//
// Static::RandomNormalOp
//===----------------------------------------------------------------------===//

LogicalResult Static::RandomNormalOp::verify() {
  auto meanTy = getMean().getType();
  auto varTy = getVariance().getType();
  if (meanTy.getElementType() != varTy.getElementType())
    return emitOpError("Expect mean and variance DTypes to be the same");

  if (getResult().getType().hasUnknownShape())
    return emitOpError("Output shape cannot be unknown");

  return success();
}

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult Static::ReshapeOp::canonicalize(Static::ReshapeOp op,
                                              PatternRewriter &rewriter) {
  // We fold consecutive static reshapes into a single reshape.
  if (auto precedingOp = op.getInput().getDefiningOp<Static::ReshapeOp>()) {
    rewriter.modifyOpInPlace(op,
                             [&] { op.setOperand(precedingOp.getInput()); });
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
      rewriter.modifyOpInPlace(op, [&] { op.setOperand(rebIn); });
      return success();
    }
  }

  size_t inputRank = inputTy.getRank();
  size_t resRank = resTy.getRank();
  if (inputRank == 0 && resRank == 1) {
    // We can fold static reshapes from scalars to singletons.
    if (TypedAttr param = getParamAttrThrough<IndexToTensorOp>(inputVal)) {
      auto shape = ShapeAttr::get(SmallVector<TypedAttr>{param},
                                  rewriter.getType<ShapeType>());
      auto newConstOp =
          rewriter.create<MOSH::ParamToValueOp>(op->getLoc(), shape);
      rewriter.replaceOpWithNewOp<Shape::ToTensorOp>(op, resTy, newConstOp);
      return success();
    }
  }
  if (inputRank == 1 && resRank == 0) {
    // We can fold static reshapes from singletons to scalars.
    if (TypedAttr param = getParamAttrThrough<Shape::ToTensorOp>(inputVal)) {
      if (auto shape = dyn_cast<ShapeAttr>(param);
          shape && shape.getRank() == 1) {
        auto newConstOp =
            rewriter.create<MOSH::ParamToValueOp>(op->getLoc(), shape[0]);
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

bool Static::ReshapeOp::preferResultsDevice() { return true; }

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

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input, int64_t width) {
  ASSERT_STREAM(llvm::is_contained({32, 64}, width),
                << "`mo.shape_of` returns si32 or si64");
  Type intTy = odsBuilder.getIntegerType(width, /*isSigned=*/true);

  int64_t outputSize = mlir::ShapedType::kDynamic;
  if (input.getType().hasRank())
    outputSize = input.getType().getRank();

  // TODO(#18749): don't introduce `?` with operations.
  build(odsBuilder, odsState, MO::TensorType::get({outputSize}, intTy), input);
}

void ShapeOfOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TensorValue input) {
  build(odsBuilder, odsState, input, /*width=*/32);
}

//===----------------------------------------------------------------------===//
// SliceOp
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

  // If the input and the output are the same and neither have ? in them then
  // we can maybe fold.
  if (getResult().getType() != inputTy || inputTy.hasUnknownShape())
    return {};

  // If all of the steps are `1` then this is an identity. We need to check
  // this as you could produce the same shape by walking backwards through the
  // slice i.e -1. So we must check they are 1 which means the only way to
  // produce the same shape as input is through start of 0 and end of -1. I.E
  // an identity.
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

DeviceRefAttr SliceOp::getOperandDevice(size_t index) {
  return (index == getStartMutable().getOperandNumber() ||
          index == getStopMutable().getOperandNumber() ||
          index == getStepMutable().getOperandNumber())
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::canonicalize(TransposeOp op,
                                        PatternRewriter &rewriter) {
  // Nothing we can do if op is dynamically ranked.
  if (!op.getType().hasRank())
    return failure();
  size_t opRank = op.getType().getRank();

  // Extract constant permutation.
  auto maybePermVec = tryGetConstantIntVector(op.getPerm());
  if (!maybePermVec.has_value())
    return failure();
  ArrayRef<int64_t> permVec(maybePermVec.value());

  // Fold away identity permutation, e.g., [0, 1, 2] for 3D transpose.
  auto identityPerm = llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
  if (llvm::equal(permVec, identityPerm)) {
    rewriter.replaceOpWithNewOp<RebindOp>(op, op.getType(), op.getInput(),
                                          op.getOutputParamDecls());
    return success();
  }

  // NOTE The following (transposes merging or transpose-->reshape) assumes
  // the verifier checks against invalid permutations.

  // Merge consecutive transposes by merging their constant permutations
  if (auto inputTransposeOp = op.getInput().getDefiningOp<TransposeOp>()) {
    auto maybeInputPermVec =
        tryGetConstantIntVector(inputTransposeOp.getPerm());
    if (!maybeInputPermVec.has_value())
      return failure();
    ArrayRef<int64_t> inputPermVec(maybeInputPermVec.value());

    // Compute the merged permutation, and replace the consecutive transpose
    // with a new transpose with the merged permutation.
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

  size_t opRank = resTy.getRank();
  if (inputTy.getRank() != opRank)
    return emitError("input and result ranks do not match");

  // ODS constraints ensure that perm has rank = 1.
  if ((int64_t)opRank != permTy.getIntDim(0))
    return emitError("input and result ranks do not match perm size");

  // A valid perm must be a permutation of [0, ..., rank - 1].
  if (auto maybePermVec = tryGetConstantIntVector(getPerm())) {
    auto permVec = maybePermVec.value();
    auto identityPerm =
        llvm::iota_range<int64_t>(0, opRank, /*Inclusive=*/false);
    SmallVector<int64_t> sortedPerm(permVec);
    llvm::sort(sortedPerm);
    if (!llvm::equal(sortedPerm, identityPerm))
      return emitError("invalid permutations");

    if (inputTy.hasStaticShape() && resTy.hasStaticShape() &&
        !llvm::equal(resTy.getTypedShapeAttr().getValues(),
                     permute(inputTy.getTypedShapeAttr().getValues(), permVec)))
      return emitError("output shape is invalid based on constant permutation");
  }

  return success();
}

DeviceRefAttr TransposeOp::getOperandDevice(size_t index) {
  return index == getPermMutable().getOperandNumber()
             ? MO::getHostDevice(getOperation())
             : nullptr;
}

//===----------------------------------------------------------------------===//
// MO_TileOp
//===----------------------------------------------------------------------===//

LogicalResult TileOp::canonicalize(TileOp op, PatternRewriter &rewriter) {
  TensorType inputTy = op.getInput().getType();
  if (!inputTy.hasRank())
    return failure();

  // Extract constant repeats.
  auto maybeRepeatsVec = tryGetConstantIntVector(op.getRepeats());
  if (!maybeRepeatsVec.has_value())
    return failure();
  ArrayRef<int64_t> repeatsVec(maybeRepeatsVec.value());

  // Check if the tile operation can be replaced with a broadcast. Each repeats
  // value leaves the input dimension unchanged or the input dimension must be
  // a broadcast.
  for (auto [inDim, repeats] : llvm::zip(inputTy.getIntDims(), repeatsVec)) {
    if (repeats < 1 || (repeats > 1 && inDim != 1))
      return failure();
  }

  TensorType shapeTy =
      TensorType::get({(int64_t)inputTy.getRank()},
                      rewriter.getIntegerType(64, /*isSigned=*/true));
  TensorValue inputShape =
      rewriter.create<ShapeOfOp>(op.getLoc(), shapeTy, op.getInput());
  TensorValue bcastShape = rewriter.create<BroadcastShapeOp>(
      op.getLoc(), shapeTy, inputShape, op.getRepeats());
  rewriter.replaceOpWithNewOp<BroadcastToOp>(
      op, op.getType(), op.getInput(), bcastShape, op.getOutputParamDecls());
  return success();
}

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

LogicalResult Shape::ToTensorOp::canonicalize(Shape::ToTensorOp op,
                                              PatternRewriter &rewriter) {
  auto paramToValueOp = op.getInput().getDefiningOp<MOSH::ParamToValueOp>();
  if (!paramToValueOp)
    return failure();

  auto inputShapeAttr = dyn_cast<MOSH::ShapeAttr>(paramToValueOp.getValue());
  if (!inputShapeAttr)
    return failure();

  TensorType resTy = op.getType();
  if (!resTy.hasStaticShape()) {
    TensorType newTy = resTy.cloneWith(ShapeAttr::get(
        {(int64_t)inputShapeAttr.getRank()}, rewriter.getType<ShapeType>()));
    Location loc = op.getLoc();
    auto newOp = rewriter.create<Shape::ToTensorOp>(loc, newTy, paramToValueOp);
    rewriter.replaceOpWithNewOp<RebindOp>(op, resTy, newOp);
    return success();
  }

  if (inputShapeAttr.isConstant()) {
    auto outTy =
        RankedTensorType::get(resTy.getIntDims(), resTy.getElementType());
    SmallVector<APInt> res =
        llvm::map_to_vector(inputShapeAttr.getIntDims(), [](int64_t dim) {
          return APInt(64, dim, /*isSigned=*/true);
        });
    auto elementsAttr =
        cast<ElementsAttr>(IntArrayElementsAttr::get(outTy, res));
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), elementsAttr);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// MO_LayoutTransformOp
//===----------------------------------------------------------------------===//

LogicalResult LayoutTransformOp::verify() {
  if (!getInput().getType().getLayoutAttr())
    return emitError() << "input type must contain layout annotation";

  if (!getResult().getType().getLayoutAttr())
    return emitError() << "output type must contain layout annotation";

  return success();
}

LogicalResult LayoutTransformOp::canonicalize(LayoutTransformOp op,
                                              PatternRewriter &rewriter) {
  LayoutAttr inputLayout = op.getInput().getType().getLayoutAttr();
  LayoutAttr outputLayout = op.getResult().getType().getLayoutAttr();
  if (inputLayout == outputLayout) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

/// This method is misnamed and should be 'prefer result or operands'. In
/// this case we just want AssignDevicesPass to place the print on whatever
/// device the operand is on.
bool DebugTensorPrintOp::preferResultsDevice() { return true; }

//===----------------------------------------------------------------------===//
// List ops
//===----------------------------------------------------------------------===//

/// Return whether `ty1` is a subtype of `ty2`, i.e., the set of possible
/// tensors represented by `ty1` is a subset for that of `ty2`.
static bool isSubtype(TensorType ty1, TensorType ty2) {
  // !mo.tensor<..., {layout = #mo.layout<KN>}>  v.s. !mo.tensor<..., {layout =
  // #mo.layout<NK>}>

  // Check for matching metadata outside of devices
  // TODO fix with #37598
  size_t numMatches = 0;
  const auto &ty2Metadata = ty2.getMetadata();
  for (const auto &entry : ty1.getMetadata()) {
    const auto &ty1Name = entry.getName();
    const auto &ty1Value = entry.getValue();
    if (ty2Metadata.contains(ty1Name))
      numMatches++;

    if (ty1Name == "device")
      continue;
    if (ty2Metadata.get(ty1Name) != ty1Value)
      return false;
  }

  // Handle case where ty2 metadata has more entries than ty1
  if (ty2Metadata.size() != numMatches)
    return false;

  // !mo.tensor<..., f32>  v.s. !mo.tensor<..., si32>
  if (ty1.getDtype() != ty2.getDtype())
    return false;

  TypedAttr shapeAttr1 = ty1.getShapeAttr();
  TypedAttr shapeAttr2 = ty2.getShapeAttr();

  if (!ty2.hasRank()) { // ty2: !mo.tensor<?|Sh0, f32>
    // If `ty2` has unknown shape `?`, any `ty1` will be a subtype of `ty2`;
    // otherwise the shape must match textually, e.g., `Sh0 == Sh0`.
    if (isa<KGEN::UnknownAttr>(shapeAttr2))
      return true;
    else
      return shapeAttr1 != shapeAttr2;
  }

  // Otherwise `ty2` is ranked, i.e., ty2: !mo.tensor<[...], f32>

  // ty1: !mo.tensor<?, f32>
  if (!ty1.hasRank())
    return false;

  // ty1: !mo.tensor<[...], f32>
  // 1. Make sure # of dims match.
  if (ty1.getRank() != ty2.getRank())
    return false;

  // 2. Make sure lhs' individual dims are more general.
  for (auto [dimAttr1, dimAttr2] :
       llvm::zip(cast<ShapeAttr>(shapeAttr1).getValues(),
                 cast<ShapeAttr>(shapeAttr2).getValues())) {
    // If `dimAttr2` is an unknown dimension `?`, `dimAttr1` can be anything in
    // order for `ty1` to be a subtype of `ty2`; otherwise the dimensions must
    // match textually, e.g., `42 == 42`.
    if (!isa<KGEN::UnknownAttr>(dimAttr2) && dimAttr1 != dimAttr2)
      return false;
  }

  return true;
}

/// Return whether `ty1` is a subtype of `ty2`, i.e., the set of possible
/// values represented by `ty1` is a subset for that of `ty2`. Currently only
/// support nontrivial subtyping for `TensorType`.
static bool isSubtype(Type ty1, Type ty2) {
  auto tensorTy1 = dyn_cast<TensorType>(ty1);
  auto tensorTy2 = dyn_cast<TensorType>(ty2);
  if (tensorTy1 && tensorTy2)
    return isSubtype(tensorTy1, tensorTy2);

  return ty1 == ty2;
}

LogicalResult ListAppendOp::verify() {
  Type inputListElementTy = getInput().getType().getElementType();
  Type outputListElementTy = getType().getElementType();
  if (!isSubtype(inputListElementTy, outputListElementTy)) {
    return emitError() << "input list element type must be a subtype of output "
                          "list element type: "
                       << inputListElementTy << " v.s. " << outputListElementTy;
  }

  TensorType newElementTy = getElement().getType();
  if (!isSubtype(newElementTy, outputListElementTy)) {
    return emitError()
           << "new element type must be a subtype of output list element type: "
           << newElementTy << " v.s. " << outputListElementTy;
  }

  return success();
}

LogicalResult ListConcatOp::verify() {

  Type lhsListElementTy = getLhs().getType().getElementType();
  Type rhsListElementTy = getRhs().getType().getElementType();

  Type resultListElementTy = getType().getElementType();

  if (!isSubtype(lhsListElementTy, resultListElementTy)) {
    return emitError()
           << "LHS input type must be a subtype of result list element type: "
           << lhsListElementTy << " v.s. " << resultListElementTy;
  }

  if (!isSubtype(rhsListElementTy, resultListElementTy)) {
    return emitError()
           << "RHS input type must be a subtype of result list element type: "
           << rhsListElementTy << " v.s. " << resultListElementTy;
  }

  return success();
}

LogicalResult ListCreateOp::verify() {
  auto resTy = dyn_cast<TensorType>(getType().getElementType());
  if (!resTy) {
    return emitError() << "output list must contain tensors, but got: "
                       << resTy;
  }

  for (Value input : getOperands()) {
    auto inTy = cast<TensorType>(input.getType());
    if (!isSubtype(inTy, resTy)) {
      return emitError()
             << "input type must be a subtype of list element type: " << inTy
             << " v.s. " << resTy;
    }
  }
  return success();
}

LogicalResult ListGetOp::verify() {
  Type listElementTy = getInput().getType().getElementType();
  Type outputTy = getType();
  if (listElementTy != outputTy) {
    return emitError() << "list element type must match output type: "
                       << listElementTy << " v.s. " << outputTy;
  }
  return success();
}

LogicalResult ListInsertOp::verify() {
  Type inputListElementTy = getInput().getType().getElementType();
  Type outputListElementTy = getType().getElementType();
  if (!isSubtype(inputListElementTy, outputListElementTy)) {
    return emitError() << "input list element type must be a subtype of output "
                          "list element type: "
                       << inputListElementTy << " v.s. " << outputListElementTy;
  }

  TensorType newElementTy = getElement().getType();
  if (!isSubtype(newElementTy, outputListElementTy)) {
    return emitError()
           << "new element type must be a subtype of output list element type: "
           << newElementTy << " v.s. " << outputListElementTy;
  }

  return success();
}

LogicalResult ListSliceOp::verify() {
  ListType inputTy = getInput().getType();
  ListType resultTy = getResult().getType();

  if (!isSubtype(inputTy.getElementType(), resultTy.getElementType())) {
    return emitError() << "Input list element type must be a subtype of output "
                          "list element type: "
                       << inputTy.getElementType() << " v.s. "
                       << resultTy.getElementType();
  }

  return success();
}

LogicalResult ConcatFromListOp::verify() {
  auto inputListElementTy =
      dyn_cast<TensorType>(getInput().getType().getElementType());
  if (!inputListElementTy)
    return emitError("input list must contain tensors");

  TensorType outputTy = getType();
  if (inputListElementTy.getElementType() != outputTy.getElementType()) {
    return emitError()
           << "input list tensor dtype must match output tensor dtype: "
           << inputListElementTy << " v.s. " << outputTy;
  }

  return success();
}

LogicalResult ConcatFromListOp::canonicalize(ConcatFromListOp op,
                                             PatternRewriter &rewriter) {
  if (auto constList = tryGetConstantIntVector(op.getInput())) {
    rewriter.replaceOpWithNewOp<ConstantOp>(op, *constList);
    return success();
  }
  return failure();
}

LogicalResult GuardOp::verify() {
  if (getInputs().size() != getResults().size()) {
    return emitError() << "Expected the number of inputs and outputs to be "
                          "exactly the same";
  }
  return success();
}

LogicalResult GuardOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {

  for (Value val : llvm::drop_begin(operands))
    inferredReturnTypes.push_back(val.getType());

  return success();
}

/// Folds away the guard op when its input chain comes from a:
/// mosh.param.to_value: !mo.chain = <#mo.chain>
/// Such cases occur after successful shape inference and the folder for
/// mo.chain.create has run (see its docs for more details)
LogicalResult GuardOp::fold(FoldAdaptor adaptor,
                            SmallVectorImpl<OpFoldResult> &results) {

  auto chainDefOp = getChain().getDefiningOp<MOSH::ParamToValueOp>();

  if (!chainDefOp)
    return failure();

  bool isTrivialChain = isa<MO::ChainAttr>(chainDefOp.getValue());

  if (!isTrivialChain)
    return failure();

  results.insert(results.begin(), getInputs().begin(), getInputs().end());

  return success();
}

/// Remove inputs that are the results of mosh.param.to_value on constant
/// mo.chains (i.e. #mo.chain).
///
/// Such a structure can occur after shape inference passes run. However,
/// this will only modify the op if *some* of the inputs are such chains.
///
/// In cases where they are all these kinds of constant chains the op is folded
/// away and replaced with the trivial chain construction:
/// mosh.param.to_value: !mo.chain = <#mo.chain>).
OpFoldResult ChainCreateOp::fold(FoldAdaptor adaptor) {

  if (getInputs().empty())
    return nullptr;

  SmallVector<Value> keptValues;

  for (Value val : getInputs()) {
    auto chainDefOp = val.getDefiningOp<MOSH::ParamToValueOp>();

    if (!chainDefOp) {
      keptValues.push_back(val);
      continue;
    }

    auto isConstantChain = isa<MO::ChainAttr>(chainDefOp.getValue());

    if (!isConstantChain) {
      keptValues.push_back(val);
      continue;
    }
  }

  // In this case all the inputs were from 'constant' chains, that is:
  // mosh.param.to_value: !mo.chain = <#mo.chain>
  //
  // replace the output of this op with one of these
  if (keptValues.empty())
    return getInputs().front();

  // In this case none of the inputs were constant chains so leave the op
  // unperturbed.
  if (keptValues.size() == getInputs().size())
    return nullptr;

  getOperation()->setOperands(keptValues);

  return nullptr;
}

void AssertOp::build(OpBuilder &builder, OperationState &state, Value cond,
                     const Twine &message) {
  auto attr = StringAttr::get(message.str(),
                              KGEN::StringType::get(builder.getContext()));
  build(builder, state, {}, cond, cast<TypedAttr>(attr));
}

static TypedAttr emitStaticCallImpl(MLIRContext *ctx,
                                    ArrayRef<TypedAttr> operands,
                                    const Twine &message) {
  SmallVector<std::pair<StringAttr (*)(mlir::OperationName), Attribute>>
      msgAttrNameFuncAndArg = {
          {AssertOp::getMessageAttrName,
           StringAttr::get(message, KGEN::StringType::get(ctx))}};
  auto chainTy = ChainType::get(ctx);
  return KGEN::emitMLIROperationCall(AssertOp::getOperationName(),
                                     msgAttrNameFuncAndArg, operands, chainTy);
}

TypedAttr AssertOp::emitStaticCall(TypedAttr cond, const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {cond}, message);
}

static TypedAttr emitStaticCall(TypedAttr chain, TypedAttr cond,
                                const Twine &message) {
  return emitStaticCallImpl(cond.getContext(), {chain, cond}, message);
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

static void linkAssertPair(mlir::RewriterBase &rewriter,
                           Operation *prevAssertOp,
                           Operation *currentAssertOp) {

  MLIRContext *ctx = currentAssertOp->getContext();

  auto prevAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(prevAssertOp);
  auto currentAssertDeclareOp = dyn_cast<KGEN::ParamDeclareOp>(currentAssertOp);

  // Grab the names of some relevant attrs so we don't hard code them later on.
  auto paramDeclOpName =
      mlir::OperationName(KGEN::ParamDeclareOp::getOperationName(), ctx);
  auto assertOpName = mlir::OperationName(AssertOp::getOperationName(), ctx);

  auto prevChain = prevAssertDeclareOp.getValue();

  auto operatorAttr = cast<KGEN::ParamOperatorAttr>(prevChain);
  auto operands = operatorAttr.getOperands();

  auto mlirOp = cast<KGEN::MLIROpAttr>(operands[0]);
  auto dictAttr = mlirOp.getAttrs();
  auto message = dictAttr.get(AssertOp::getMessageAttrName(assertOpName));

  StringAttr messageAttr = cast<mlir::StringAttr>(message);
  auto errorMsg = messageAttr.strref();

  auto currentOperands =
      cast<KGEN::ParamOperatorAttr>(currentAssertDeclareOp.getValue())
          .getOperands()
          .drop_front();

  // finally create the new assert that accepts prevAssertOp's output chain
  auto newAttr = emitStaticCall(
      KGEN::ParamDeclRefAttr::get(prevAssertDeclareOp.getParamDecl()),
      currentOperands[0], errorMsg);

  rewriter.startOpModification(currentAssertDeclareOp);
  currentAssertDeclareOp->setAttr(
      KGEN::ParamDeclareOp::getValueAttrName(paramDeclOpName), newAttr);
  rewriter.finalizeOpModification(currentAssertDeclareOp);
}

static void linkAssertSequence(mlir::RewriterBase &rewriter,
                               ArrayRef<KGEN::ParamDeclareOp> asserts) {

  ASSERT_STREAM(!asserts.empty(),
                << "linkAssertSequence received an empty sequence of asserts!");

  KGEN::ParamDeclareOp prevAssert = asserts.front();

  for (auto assert : asserts.drop_front()) {
    linkAssertPair(rewriter, prevAssert, assert);
    prevAssert = assert;
  }
}

void AssertOp::materializeAllAssertLogic(
    mlir::RewriterBase &rewriter, Operation *op,
    ArrayRef<KGEN::ParamDeclareOp> asserts) {

  if (asserts.empty())
    return;

  linkAssertSequence(rewriter, asserts);

  mlir::RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // the following block takes all the KGEN param level chains and adds ops to
  // make them actual SSA values
  SmallVector<Value> allChains;
  for (auto assert : asserts) {
    auto declRefAttr = KGEN::ParamDeclRefAttr::get(
        cast<KGEN::ParamDeclAttr>(assert.getParamDecl()));

    allChains.push_back(
        rewriter.create<MOSH::ParamToValueOp>(op->getLoc(), declRefAttr));
  }

  // fan in all the assertion chains as the guard op takes a single chain
  auto guardChain =
      rewriter.create<MO::ChainCreateOp>(op->getLoc(), ValueRange{allChains});

  auto guardOp =
      rewriter.create<MO::GuardOp>(op->getLoc(), guardChain, op->getOperands());

  rewriter.modifyOpInPlace(op, [&guardOp, &op] {
    for (auto &&[replacement, old] :
         llvm::zip(guardOp->getResults(), op->getOpOperands())) {
      old.set(replacement);
    }
  });
}

//===----------------------------------------------------------------------===//
// MO_CustomOp
//===----------------------------------------------------------------------===//

DeviceRefAttr CustomOp::getOperandDevice(size_t index) {
  for (auto deviceIndex : getOperandDevices())
    if (deviceIndex.getIndex().getInt() == static_cast<int64_t>(index))
      return deviceIndex.getDevice();

  return nullptr;
}

LogicalResult CustomOp::verify() {
  SmallVector<bool> assignedIndex(getNumOperands(), false);

  // Ensure device placements for operands are correct
  for (auto deviceIndex : getOperandDevices()) {
    int64_t index = deviceIndex.getIndex().getInt();
    if (index < 0 || index >= getNumOperands())
      return emitError(
          "Device operand index out of bounds of [0, numOperands)");

    if (assignedIndex[index])
      return emitError("Device operand index " + std::to_string(index) +
                       " defined twice!");

    assignedIndex[index] = true;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MO Mutable tensor operations
//===----------------------------------------------------------------------===//

LogicalResult MutableLoad::verify() {

  for (auto &&[idx, result] : llvm::enumerate(getResults())) {
    auto numUses = llvm::range_size(result.getUses());

    if (numUses > 1) {
      return emitError() << llvm::formatv(
                 "Results from {0} are only allowed "
                 "to have at most one use -- found {1} for result #{1}",
                 getOperationName(), idx, numUses);
    }
  }

  return success();
}

LogicalResult MutableStore::verify() {

  auto numUses = llvm::range_size(getResult().getUses());

  if (numUses > 1) {
    return emitError() << llvm::formatv(
               "The output chain from {0} is only allowed "
               "to have at most one use -- found: {1}",
               getOperationName(), numUses);
  }

  // NOTE: might be able to relax this constraint in the future if we have a
  // full lowering path, lambda based fusion, and copying of results when
  // multiple uses exist
  auto valueTensorNumUses = llvm::range_size(getInTensor().getUses());

  if (valueTensorNumUses > 1) {
    return emitError() << llvm::formatv("The value semantic tensor copied from "
                                        "has more than one use, found: {0}",
                                        valueTensorNumUses);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

// Needed because the type and attribute constraints in the generated file are
// not placed in the correct namespace by tablegen.
using namespace Shape;
using M::KGEN::parseStringParam;
using M::KGEN::printStringParam;

void MODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "GenericML/GraphCompiler/MODialect/MO.cpp.inc"
