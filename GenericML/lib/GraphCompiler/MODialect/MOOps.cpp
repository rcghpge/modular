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
#include "GenericML/GraphCompiler/Matchers.h"
#include "GenericML/Support/Dims.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/ML/Permutation.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace M;
using namespace MO;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace M::MO {

static SmallVector<ChainValue> getChainValues(ValueRange values) {
  SmallVector<ChainValue> results;

  for (auto value : values) {
    if (auto chain = dyn_cast<ChainValue>(value))
      results.push_back(chain);
  }

  return results;
}

static ParseResult
parseParamDeclElement(OpAsmParser &parser,
                      SmallVectorImpl<ParamDeclAttr> &decls) {
  auto dimType = ShapeAttr::getDimType(parser.getContext());

  StringAttr name;
  Type type;
  if (KGEN::parseParamName(parser, name) ||
      KGEN::parseColonTypeOrDefault(parser, type, dimType))
    return failure();

  decls.push_back(ParamDeclAttr::get(name, type));
  return success();
}

static void printParamDeclElement(OpAsmPrinter &printer,
                                  KGEN::ParamDeclAttr decl) {
  auto dimType = ShapeAttr::getDimType(decl.getContext());
  KGEN::printParamName(printer, decl.getName());
  KGEN::printColonTypeOrDefault(printer, decl.getType(), dimType);
}

ParseResult parseParamDecls(OpAsmParser &p,
                            KGEN::ParamDeclArrayAttr &paramDecls) {
  auto parserHook = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    return parseParamDeclElement(p, decls);
  };

  if (succeeded(p.parseOptionalLess()) && failed(p.parseOptionalGreater())) {
    if (p.parseLParen() || p.parseRParen() || p.parseArrow() ||
        KGEN::parseParamDecls(p, paramDecls, parserHook) || p.parseGreater())
      return failure();
  } else {
    paramDecls = KGEN::ParamDeclArrayAttr::get(p.getContext(), {});
  }
  return success();
}

void printParamDecls(OpAsmPrinter &p, Operation *,
                     KGEN::ParamDeclArrayAttr paramDecls) {
  auto printerHook = [&](KGEN::ParamDeclAttr decl) {
    printParamDeclElement(p, decl);
  };

  if (!paramDecls.empty()) {
    p << "<() -> ";
    KGEN::printParamDecls(p, paramDecls, printerHook);
    p << '>';
  }
}

/// Print the result tensor device in MO::TransferOp.
ParseResult parseTransferResult(AsmParser &p, Type inputType,
                                Type &resultType) {
  DeviceRefAttr deviceRef;
  Attribute parsedDevice = DeviceRefAttr::parse(p, /*odsType=*/{});
  if (!parsedDevice)
    return failure();
  deviceRef = cast<DeviceRefAttr>(parsedDevice);
  resultType = cast<TensorType>(inputType).cloneWith(deviceRef);
  return success();
}

/// Print the result tensor device in MO::TransferOp.
void printTransferResult(AsmPrinter &p, Operation *op, TensorType inputType,
                         TensorType resultType) {
  DeviceRefAttr deviceRef = resultType.getDeviceRef();
  ASSERT_STREAM(deviceRef, "DeviceRefAttr should be present.");
  deviceRef.print(p);
}

} // namespace M::MO

SmallVector<ConstantOp> M::MO::tryGetConstantOpsOfRank(TensorValue val,
                                                       size_t rank) {
  SmallVector<ConstantOp> constantOps;
  Operation *inputOp = val.getDefiningOp();

  // Handle identity chains.
  auto isIdentityLike = [&](Operation *op) {
    if (isa_and_nonnull<RebindOp>(op))
      return true;
    if (isa_and_nonnull<mlir::UnrealizedConversionCastOp>(op))
      return true;
    if (isa_and_nonnull<ReshapeOp>(op)) {
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
  if (auto concatOp = dyn_cast_if_present<ConcatOp>(inputOp)) {
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
  auto constantOp = dyn_cast_if_present<ConstantOp>(inputOp);
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
  if (auto maybeDoubleBlob = tryGetConstantOfRank<double>(val, 0))
    return maybeDoubleBlob.value().front();
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

std::optional<APFloat> M::MO::tryGetConstantAPFloat(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APFloat>(val, 0))
    return maybeBlob.value().front();
  return std::nullopt;
}

std::optional<APInt> M::MO::tryGetConstantAPInt(TensorValue val) {
  if (auto maybeBlob = tryGetConstantOfRank<APInt>(val, 0))
    return maybeBlob.value().front();
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
  auto listCreateOp = dyn_cast_if_present<ListCreateOp>(val.getDefiningOp());
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

std::optional<SmallVector<APFloat>>
M::MO::tryGetConstantAPFloatVector(ListValue val) {
  auto listCreateOp = dyn_cast_if_present<ListCreateOp>(val.getDefiningOp());
  if (!listCreateOp)
    return std::nullopt;

  SmallVector<APFloat> vec;
  for (auto inp : listCreateOp->getOperands()) {
    auto moTensorVal = dyn_cast<TensorValue>(inp);
    if (!moTensorVal)
      return std::nullopt;
    auto maybeVal = tryGetConstantAPFloat(moTensorVal);
    if (!maybeVal)
      return std::nullopt;
    vec.push_back((*maybeVal));
  }
  return vec;
}

std::optional<SmallVector<APInt>>
M::MO::tryGetConstantAPIntVector(ListValue val) {
  auto listCreateOp = dyn_cast_if_present<ListCreateOp>(val.getDefiningOp());
  if (!listCreateOp)
    return std::nullopt;

  SmallVector<APInt> vec;
  for (auto inp : listCreateOp->getOperands()) {
    auto moTensorVal = dyn_cast<TensorValue>(inp);
    if (!moTensorVal)
      return std::nullopt;
    auto maybeVal = tryGetConstantAPInt(moTensorVal);
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

  return MOSH::ShapeAttr::getDim(val.getContext(), *maybeConst);
}

/// Verifies that the first block of the given `region` is terminated by a
/// TerminatorTy. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(Operation *op, Region &region,
                                           StringRef errorMessage) {
  Operation *terminatorOperation = nullptr;
  if (!region.empty() && !region.front().empty()) {
    terminatorOperation = &region.front().back();
    if (auto yield = dyn_cast_if_present<TerminatorTy>(terminatorOperation))
      return yield;
  }
  auto diag = op->emitOpError(errorMessage);
  if (terminatorOperation)
    diag.attachNote(terminatorOperation->getLoc()) << "terminator here";
  return nullptr;
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
      newResTy = TensorType::get(
          funcOutTensorTy.getShapeAttr(), funcOutTensorTy.getDtype(),
          newOutTensorTy.getDeviceRef(), newOutTensorTy.getMetadata());
    }

    if (auto funcOutBufferTy = dyn_cast<BufferType>(funcOutTy)) {
      auto newOutBufferTy = cast<BufferType>(newOutTy);

      newResTy = BufferType::get(
          funcOutBufferTy.getShapeAttr(), funcOutBufferTy.getDtype(),
          newOutBufferTy.getDeviceRef(), newOutBufferTy.getMetadata());
    }

    if (auto funcOutListTy = dyn_cast<ListType>(funcOutTy)) {
      auto newOutListTy = cast<ListType>(newOutTy);
      newResTy = ListType::get(funcOutListTy.getContext(),
                               newOutListTy.getElementType());
    }

    newResTypes.push_back(newResTy);
  }

  auto funcType = FunctionType::get(ctx, newArgTypes, newResTypes);
  FuncTypeGeneratorType signature =
      KGEN::FuncTypeGeneratorType::remapToFuncTypeGenerator(
          op.getInputParameters(), funcType);
  op.setSignature(signature);
  op.setFunctionType(funcType);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto graphOp =
      symbolTable.lookupNearestSymbolFrom<GraphOp>(*this, getCalleeAttr());
  if (!graphOp)
    return emitError("'") << getCalleeAttr()
                          << "' does not reference a valid graph";

  if (!graphOp.getIsSubgraph())
    return emitError() << "Only subgraphs can be called";

  auto fnType = graphOp.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitError() << "Subgraph " << graphOp.getName()
                       << " has wrong number of arguments";

  auto emitErrorFn = [&](size_t idx, Type graphArgType, Type operandType) {
    auto diag = emitError() << "Subgraph " << graphOp.getName()
                            << " has wrong type for argument " << idx
                            << " (function type: " << graphArgType
                            << ", operand type: " << operandType << ")";
    diag.attachNote(graphOp.getLoc()) << "graph op here";
    return failure();
  };

  for (auto [idx, graphArgType, operandType] : llvm::enumerate(
           graphOp.getFunctionType().getInputs(), getOperandTypes())) {
    if (isa<TensorType>(graphArgType) && isa<TensorType>(operandType)) {
      auto graphArgTensorType = cast<TensorType>(graphArgType);
      auto operandTensorType = cast<TensorType>(operandType);
      if (graphArgTensorType.cloneWithoutLayout() !=
          operandTensorType.cloneWithoutLayout()) {
        return emitErrorFn(idx, graphArgType, operandType);
      }
    } else if (graphArgType != operandType) {
      return emitErrorFn(idx, graphArgType, operandType);
    }
  }

  return success();
}

LogicalResult CallOp::verify() {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();

  // Add this logic here to ensure eager verification of call operations
  // at build time in the Graph API.
  mlir::SymbolTableAnalysis symTabAnalysis(moduleOp);
  return verifySymbolUses(symTabAnalysis.getSymbolTables());
}

bool CallOp::isInPlace() {
  return !getInChains().empty() && !getOutChains().empty();
}

SmallVector<ChainValue> CallOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CallOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> CallOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

void TransferOp::build(OpBuilder &builder, OperationState &state,
                       TensorValue inputValue, DeviceRefAttr destDevice,
                       ChainValue inChain, bool alwaysElideSameDeviceCopy) {
  auto resultTy = inputValue.getType().cloneWith(destDevice);
  auto chainTy = ChainType::get(builder.getContext());
  build(builder, state, resultTy, chainTy, inputValue,
        alwaysElideSameDeviceCopy, inChain);
}

LogicalResult TransferOp::verify() {
  TensorType inputType = getInput().getType();
  TensorType resultType = getResult().getType();
  TensorType resTypeWithInputDevice =
      resultType.cloneWith(inputType.getDeviceRef());
  if (resTypeWithInputDevice != inputType) {
    return emitError()
           << "expected result type to only differ from input type in device ("
           << inputType << " vs " << resultType << ").";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

/// Name of optional attribute holding the #M.device_spec_collection describing
/// the (virtual) devices available to the graph.
constexpr StringRef kGraphDeviceSpecCollectionAttrName = "device_specs";

constexpr StringRef kGraphTargetInfoAttrName = "target_info";

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
                    TypeRange inputTypes, TypeRange resultTypes,
                    bool isSubgraph) {
  // Below we basically inline and specialize `buildWithEntryBlock`, which
  // doesn't work because we use `signature` and not `function_type`.
  state.addAttribute(getSymNameAttrName(state.name),
                     builder.getStringAttr(name));
  if (isSubgraph) {
    state.addAttribute(getIsSubgraphAttrName(state.name),
                       builder.getUnitAttr());
  }
  // Add signature attribute
  FunctionType funcType = builder.getFunctionType(inputTypes, resultTypes);
  auto sig = KGEN::FuncTypeGeneratorType::get(/*inputParamTypes=*/{}, funcType);
  state.addAttribute(getSignatureAttrName(state.name), TypeAttr::get(sig));
  state.addAttribute(getFunctionTypeAttrName(state.name),
                     TypeAttr::get(funcType));

  // Declare input/result parameters, if any is used in the input/result types.
  MLIRContext *ctx = builder.getContext();
  auto inputDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(inputTypes).getArrayRef());
  auto resultDecls = KGEN::ParamDeclArrayAttr::get(
      ctx, getParamDeclsFromShapes(resultTypes).getArrayRef());
  state.addAttribute(getInputParametersAttrName(state.name), inputDecls);
  state.addAttribute(getResultParametersAttrName(state.name), resultDecls);

  // Add the function body.
  Region *bodyRegion = state.addRegion();
  Block &body = bodyRegion->emplaceBlock();
  for (Type input : inputTypes)
    body.addArgument(input, state.location);
}

// GraphOp contains a property called `counter` that is used to generate
// unique symbolic dimension names. For printing/parsing, we simply place it in
// the list of misc. attributes attached to the graph and extract/insert it as
// needed.
static constexpr StringLiteral GRAPH_OP_COUNTER_NAME = "counter";
static ParseResult
parseGraphOp(OpAsmParser &p, StringAttr &name, TypeAttr &signatureAttr,
             TypeAttr &functionTypeAttr, KGEN::ParamDeclArrayAttr &inputParams,
             KGEN::ParamDeclArrayAttr &resultParams, NamedAttrList &attrs,
             Region &body, int64_t &counter) {
  if (p.parseSymbolName(name))
    return failure();

  auto parserHook = [&](SmallVectorImpl<ParamDeclAttr> &decls) -> ParseResult {
    return parseParamDeclElement(p, decls);
  };

  SmallVector<OpAsmParser::Argument> args;
  KGEN::FuncTypeGeneratorType signature;
  FunctionType functionType;
  if (KGEN::parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                           functionType, signature,
                                           /*parseDeclElt=*/parserHook))
    return failure();
  signatureAttr = TypeAttr::get(signature);
  functionTypeAttr = TypeAttr::get(functionType);

  if (p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();

  // Extract counter
  NamedAttrList extraAttrs;
  if (p.parseOptionalAttrDict(extraAttrs))
    return failure();

  counter = 0;
  if (!extraAttrs.empty()) {
    ASSERT_STREAM(extraAttrs.getAttrs().size() == 1,
                  << "Expect 'counter' to be only extra attribute.");
    NamedAttribute attr = *extraAttrs.begin();
    ASSERT_STREAM(attr.getName() == GRAPH_OP_COUNTER_NAME,
                  "Expect 'counter' to be only extra attribute.");
    counter = cast<IntegerAttr>(attr.getValue()).getInt();
  }

  return success();
}

static void printGraphOp(OpAsmPrinter &p, Operation *op, StringAttr name,
                         TypeAttr signature, TypeAttr functionType,
                         KGEN::ParamDeclArrayAttr inputParams,
                         KGEN::ParamDeclArrayAttr resultParams,
                         DictionaryAttr attrs, Region &body, int64_t counter) {
  auto printerHook = [&](KGEN::ParamDeclAttr decl) {
    printParamDeclElement(p, decl);
  };

  p.printSymbolName(name);
  KGEN::printFunctionFuncTypeGenerator(
      p, &body, inputParams, resultParams,
      cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()),
      /*printInputElt=*/printerHook,
      /*printResultElt=*/printerHook);

  auto graph = cast<GraphOp>(op);
  SmallVector<StringRef, 6> elidedAttrs{
      graph.getSymNameAttrName(), graph.getSignatureAttrName(),
      graph.getFunctionTypeAttrName(), graph.getInputParametersAttrName(),
      graph.getResultParametersAttrName()};
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);

  // Extra metadata at end for use internally in compiler
  SmallVector<NamedAttribute> extraAttrs;
  if (counter != 0) {
    auto counterAttr =
        IntegerAttr::get(IntegerType::get(op->getContext(), 64), counter);
    auto counterAttrName =
        StringAttr::get(op->getContext(), GRAPH_OP_COUNTER_NAME);
    extraAttrs.push_back(NamedAttribute(counterAttrName, counterAttr));
  }
  p.printOptionalAttrDict(extraAttrs);
}

OutputOp GraphOp::getOutputOp() {
  return cast<OutputOp>(getBodyBlock()->getTerminator());
}

LogicalResult GraphOp::verify() {
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

  return useDefGraph.verify(nullptr, cache);
}

KGEN::ParamDeclAttr GraphOp::getNewDimParam(const Twine &prefix) {
  auto dimType = MOSH::ShapeAttr::getDimType(getContext());
  return getNewParamOfType(prefix, dimType);
}

KGEN::ParamDeclAttr GraphOp::getNewShapeParam(const Twine &prefix) {
  return getNewParamOfType(prefix, ShapeType::get(getContext()));
}

KGEN::ParamDeclAttr GraphOp::getNewParamOfType(const Twine &prefix, Type type) {
  return KGEN::ParamDeclAttr::get(getNewParamName(prefix), type);
}

std::string GraphOp::getNewParamName(const Twine &prefix) {
  // TODO(#8829) avoid duplication with existing parameters in the input graph
  int64_t newCounter = getCounter() + 1;
  setCounter(newCounter);
  return (prefix + Twine(newCounter)).str();
}

void GraphOp::setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs) {
  getOperation()->setAttr(kGraphDeviceSpecCollectionAttrName, deviceSpecs);
}

TargetInfoAttr GraphOp::getDeviceTargetInfo() {
  return dyn_cast_if_present<TargetInfoAttr>(
      getOperation()->getAttr(kGraphTargetInfoAttrName));
}
void GraphOp::setDeviceTargetInfo(TargetInfoAttr targetInfo) {
  getOperation()->setAttr(kGraphTargetInfoAttrName, targetInfo);
}

ArrayRef<ParamDeclAttr> GraphOp::getInputParams() {
  return getInputParameters();
}

SmallVector<ParamDeclAttr> GraphOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getInputParameters().begin(), getInputParameters().end());
  params.append(getResultParameters().begin(), getResultParameters().end());
  return params;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

/// Parse the output values for `mo.output`, sugaring `?` for unknown values.
static ParseResult parseOutputValues(AsmParser &p,
                                     KGEN::ParameterExprArrayAttr &valuesAttr) {
  SmallVector<TypedAttr> values;
  auto parseFn = [&]() -> ParseResult {
    Type type = ShapeAttr::getDimType(p.getContext());
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
    if (Type type = value.getType(); !ShapeAttr::isDimType(type)) {
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
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
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
  ArrayRef<KGEN::ParamDeclAttr> decls = graph.getResultParameters();
  ArrayRef<TypedAttr> defs = this->getParameters();
  if (decls.size() != defs.size()) {
    return emitError(
        "output bindings and result parameters must have the same size");
  }

  // Build a mapping from result param references to their binding attribute,
  // example:
  //  { (D1, 42), (Sh0, [D0, 10]) }
  DenseMap<KGEN::ParamDeclRefAttr, TypedAttr> resParamRefToBindingAttr;
  for (auto [resParamDecl, bindingParam] : llvm::zip(decls, defs)) {
    resParamRefToBindingAttr.insert(
        {KGEN::ParamDeclRefAttr::get(resParamDecl), bindingParam});
  }

  // Substitute param references inside graph result types using the map, which
  // gives us the expected output operands type.
  SmallVector<Type> expectedOutputOperandTypes;
  for (Type ty : graph.getResultTypes()) {
    if (auto tensorTy = dyn_cast<TensorType>(ty)) {
      TypedAttr newShapeAttr = cast<TypedAttr>(tensorTy.getShapeAttr().replace(
          [&resParamRefToBindingAttr](KGEN::ParamDeclRefAttr paramRef) {
            TypedAttr resolvedAttr = resParamRefToBindingAttr.lookup(paramRef);
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
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  Operation *parentOp = getOperation()->getParentOp();
  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    if (decls.size() != defs.size()) {
      return emitOpError(
                 "[YieldOp::walkDefinitions] yield parameter bindings (")
             << defs.size() << ") and if output parameters (" << decls.size()
             << ") must have the same size";
    }
  }
  return success();
}

void YieldOp::walkDefinitions(
    function_ref<void(KGEN::ParamDeclAttr, const KGEN::ParamDefValue &)>
        walkDef) {

  Operation *parentOp = getOperation()->getParentOp();

  if (auto ifOp = dyn_cast<IfOp>(parentOp)) {
    ArrayRef<KGEN::ParamDeclAttr> decls = ifOp.getOutputParamDecls();
    ArrayRef<TypedAttr> defs = this->getParameters();
    for (auto [decl, def] : llvm::zip(decls, defs))
      walkDef(decl, def);
  }
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> IfOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> IfOp::getAllParams() {
  SmallVector<ParamDeclAttr> params;
  params.append(getOutputParamDecls().begin(), getOutputParamDecls().end());
  return params;
}

Block *IfOp::getThenBlock() { return &getThenRegion().front(); }

Block *IfOp::getElseBlock() { return &getElseRegion().front(); }

void IfOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The `then` and the `else` region branch back to the parent operation.
  if (!point.isParent()) {
    regions.emplace_back(getOperation(), getResults());
    return;
  }

  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

void IfOp::getEntrySuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  regions.emplace_back(&getThenRegion());
  regions.emplace_back(&getElseRegion());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

ArrayRef<ParamDeclAttr> WhileOp::getInputParams() { return {}; }
SmallVector<ParamDeclAttr> WhileOp::getAllParams() { return {}; }

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

OperandRange
WhileOp::getEntrySuccessorOperands(mlir::RegionSuccessor successor) {
  assert(successor.getSuccessor() == &getCondRegion() &&
         "WhileOp is expected to branch only to the first region");
  return getOperands();
}

void WhileOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // The parent op always branches to the condition region.
  if (point.isParent()) {
    regions.emplace_back(&getCondRegion(), getCondRegion().getArguments());
    return;
  }

  assert(llvm::is_contained(
             {&getCondRegion(), &getBodyRegion()},
             point.getTerminatorPredecessorOrNull()->getParentRegion()) &&
         "there are only two regions in a WhileOp");
  // The body region always branches back to the condition region.
  if (point.getTerminatorPredecessorOrNull()->getParentRegion() ==
      &getBodyRegion()) {
    regions.emplace_back(&getCondRegion(), getCondRegion().getArguments());
    return;
  }

  regions.emplace_back(getOperation(), getResults());
  regions.emplace_back(&getBodyRegion(), getBodyRegion().getArguments());
}

MutableArrayRef<OpOperand> WhileOp::getInitsMutable() {
  return getInputsMutable();
}

std::optional<MutableArrayRef<OpOperand>> WhileOp::getYieldedValuesMutable() {
  return getBodyTerminator().getOperandsMutable();
}

Block::BlockArgListType WhileOp::getRegionIterArgs() {
  return getCondRegion().getArguments();
}

SmallVector<Region *> WhileOp::getLoopRegions() {
  return {&getCondRegion(), &getBodyRegion()};
}

LogicalResult WhileOp::verify() {
  auto beforeTerminator = verifyAndGetTerminator<WhileConditionOp>(
      *this, getCondRegion(),
      "expects the 'before' region to terminate with 'mo.while.condition'");
  if (!beforeTerminator)
    return failure();

  auto afterTerminator = verifyAndGetTerminator<YieldOp>(
      *this, getBodyRegion(),
      "expects the 'after' region to terminate with 'mo.yield'");
  return success(afterTerminator != nullptr);
}

WhileConditionOp WhileOp::getCondTerminator() {
  return cast<WhileConditionOp>(getCondRegion().front().getTerminator());
}

YieldOp WhileOp::getBodyTerminator() {
  return cast<YieldOp>(getBodyRegion().front().getTerminator());
}

Block *WhileOp::getCondBlock() { return &getCondRegion().front(); }

Block *WhileOp::getBodyBlock() { return &getBodyRegion().front(); }

Block::BlockArgListType WhileOp::getCondArguments() {
  return getCondRegion().getArguments();
}

Block::BlockArgListType WhileOp::getBodyArguments() {
  return getBodyRegion().getArguments();
}

SmallVector<ChainValue> WhileOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> WhileOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> WhileOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getInputsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

void WhileConditionOp::getSuccessorRegions(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  WhileOp whileOp = getParentOp();

  // Condition can either lead to the after region or back to the parent op
  // depending on whether the condition is true or not.
  regions.emplace_back(&whileOp.getBodyRegion(),
                       whileOp.getBodyRegion().getArguments());
  regions.emplace_back(whileOp.getOperation(), whileOp.getResults());
}

MutableOperandRange
WhileConditionOp::getMutableSuccessorOperands(mlir::RegionSuccessor successor) {
  assert((successor.isParent() ||
          successor.getSuccessor() == &getParentOp().getBodyRegion()) &&
         "condition op can only exit the loop or branch to the after"
         "region");
  // Pass all operands except the condition to the successor region.
  return getArgsMutable();
}

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
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
// BroadcastToOp
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
        int64_t inDim = fromIntDim.getSInt();
        if (inDim > 1 && (inDim != toIntDim.getSInt()))
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

void BroadcastToOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TensorValue input, TensorValue broadcastLike,
                          TensorValue broadcastShape) {
  TensorType resultTy =
      input.getType().cloneWith(broadcastLike.getType().getShapeAttr());

  // Note broadcastShape may be null
  if (!broadcastShape)
    broadcastShape =
        MO::ShapeOfOp::create(odsBuilder, input.getLoc(), broadcastLike);

  build(odsBuilder, odsState, resultTy, input, broadcastShape);
}

//===----------------------------------------------------------------------===//
// Static::BroadcastToOp
//===----------------------------------------------------------------------===//

LogicalResult Static::BroadcastToOp::verify() {
  if (auto err = verifyBroadcastStaticRanks(getInput().getType(), getType()))
    return emitError(err.getError());
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

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
    return emitOpError("element type of the value (")
           << valType << ") must match the element type of the result ("
           << resType << ")";
  }
  return success();
}

// Create an ElementsAttr from a list of primitive values.
namespace {
template <typename T>
ElementsAttr createElementsAttr(OpBuilder &odsBuilder, ArrayRef<T> values,
                                Type elementType) {
  if constexpr (std::is_same_v<T, bool>) {
    // For bool, we convert to i8 type.
    auto rankedAttrTy =
        RankedTensorType::get(values.size(), odsBuilder.getI8Type());
    auto apIntVals = llvm::map_to_vector(
        values, [](bool val) { return APInt(8, val, /*isSigned=*/false); });
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, apIntVals);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_integral_v<T>) {
    return cast<ElementsAttr>(odsBuilder.getAttr<IntArrayElementsAttr>(
        values, mlir::IntegerType::Signed));
  } else if constexpr (std::is_floating_point_v<T>) {
    auto apFloatValues =
        llvm::map_to_vector(values, [](T value) { return APFloat(value); });
    return cast<ElementsAttr>(
        FloatArrayElementsAttr::get(apFloatValues, elementType));
  } else if constexpr (std::is_same_v<T, APInt>) {
    auto rankedAttrTy = RankedTensorType::get(values.size(), elementType);
    auto valueAttr = IntArrayElementsAttr::get(rankedAttrTy, values);
    return cast<ElementsAttr>(valueAttr);
  } else if constexpr (std::is_same_v<T, APFloat>) {
    return cast<ElementsAttr>(FloatArrayElementsAttr::get(values, elementType));
  }
}

template <typename T>
void buildConstant(OpBuilder &odsBuilder, OperationState &odsState,
                   ArrayRef<T> values, Type elementType,
                   DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get({(int64_t)values.size()}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(odsBuilder, values, elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}

template <typename T>
void buildConstantScalar(OpBuilder &odsBuilder, OperationState &odsState,
                         T value, Type elementType, DeviceRefAttr deviceRef) {
  if (!deviceRef)
    deviceRef = getHostDevice(odsBuilder.getContext());
  auto type =
      MO::TensorType::get(SmallVector<int64_t>{}, elementType, deviceRef);
  ElementsAttr valueAttr = createElementsAttr(
      odsBuilder, ArrayRef<T>(SmallVector<T>{value}), elementType);
  ConstantOp::build(odsBuilder, odsState, type, valueAttr);
}
} // namespace

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Type resultType, ElementsAttr value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(resultType);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int64_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(64, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<int32_t> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values,
                odsBuilder.getIntegerType(32, /*isSigned=*/true), deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APInt> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<APFloat> values, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstant(odsBuilder, odsState, values, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       ArrayRef<bool> values, DeviceRefAttr deviceRef) {
  buildConstant(odsBuilder, odsState, values, odsBuilder.getI1Type(),
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(64, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int32_t value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value,
                      odsBuilder.getIntegerType(32, /*isSigned=*/true),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APFloat value, Type elementType,
                       DeviceRefAttr deviceRef) {
  assert(elementType.isIntOrFloat() && !elementType.isInteger());
  buildConstantScalar(odsBuilder, odsState, value, elementType, deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       APInt value, Type elementType, DeviceRefAttr deviceRef) {
  assert(elementType.isInteger());
  buildConstant(odsBuilder, odsState, ArrayRef<APInt>(value), elementType,
                deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       float value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF32Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       double value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getF64Type(),
                      deviceRef);
}

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       bool value, DeviceRefAttr deviceRef) {
  buildConstantScalar(odsBuilder, odsState, value, odsBuilder.getI1Type(),
                      deviceRef);
}

//===----------------------------------------------------------------------===//
// ConstantScalarOp
//===----------------------------------------------------------------------===//

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

void ConstantScalarOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             TypedAttr value) {
  auto type = value.getType();

  DType dtype;

  if (auto integerType = dyn_cast<IntegerType>(type))
    dtype = M::getEquivalentDType(integerType);
  else
    dtype = M::getEquivalentDType(cast<FloatType>(type));

  auto resultType = odsBuilder.getType<ScalarType>(dtype);
  build(odsBuilder, odsState, resultType, value);
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::verify() {
  size_t numInputs = getInputs().size();
  if (numInputs < 1)
    return emitError("concat must have at least 1 input tensor");

  SmallVector<TensorValue> tensors;
  for (Value input : getInputs())
    tensors.emplace_back(cast<TensorValue>(input));

  tensors.emplace_back(getResult());
  if (failed(::Impl::verifyMatchingRanks(tensors))) {
    return emitError(
        "input and output ranks must be all dynamic, or all static and match");
  }

  auto elementTypes = llvm::map_range(
      tensors, [](TensorValue val) { return val.getType().getElementType(); });

  if (!llvm::all_equal(elementTypes)) {
    return emitError("input and output element types must match");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SplitOp
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

  if (failed(::Impl::verifyMatchingRanks(tensors))) {
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

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

/// Return an empty array with the given element type.

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
// CastOp
//===----------------------------------------------------------------------===//

void CastOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, DType dtype) {
  build(odsBuilder, odsState, input.getType().cloneWith(dtype), input);
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

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
// ArgNonzeroOp
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
// ArgMaxOp
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
  buildReductionOp<TOp>(
      odsBuilder, odsState, input,
      ConstantOp::create(odsBuilder, input.getLoc(), axis,
                         getHostDevice(odsBuilder.getContext())),
      outputTy);
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
// ArgMinOp
//===----------------------------------------------------------------------===//

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
// CumSumOp
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

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

LogicalResult ConvOp::verify() {
  // The sliding window interface already checks that the input and result
  // have matching ranks, so we can be more specific with the error message
  // here.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
    return emitError("filter, input and output ranks must be all dynamic, or "
                     "all static and match");

  auto op = cast<SlidingWindow>(this->getOperation());
  return op.verifyWindowParamSizes({getDilations(), getStrides()});
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

LogicalResult ConvTransposeOp::verify() {
  // The sliding window interface already checks that the input and result
  // have matching ranks, so we can be more specific with the error message
  // here.
  if (failed(
          ::Impl::verifyMatchingRanks({getInput(), getResult(), getFilter()})))
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
// RangeOp
//===----------------------------------------------------------------------===//

LogicalResult RangeOp::verify() {
  ArrayRef<KGEN::ParamDeclAttr> paramDecls = getOutputParamDecls();
  if (!paramDecls.empty()) {
    if (paramDecls.size() != 1)
      return emitError("cannot declare more than one parameter");
    if (!ShapeAttr::isDimType(paramDecls[0].getType()))
      return emitError("can only declare dimension parameters");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

LogicalResult RebindOp::verify() {
  TensorType inTy = getInput().getType();
  TensorType outTy = getType();
  // Disallow rebind to a different device.
  if (inTy.getDeviceRef() && outTy.getDeviceRef() &&
      (inTy.getDeviceRef() != outTy.getDeviceRef()))
    return emitError("input device " + inTy.getDeviceRef().toString() +
                     " must match result device " +
                     outTy.getDeviceRef().toString() + " in rebind");

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

void MeanOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, TensorValue axis, TensorType outputTy) {
  buildReductionOp<MeanOp>(odsBuilder, odsState, input, axis, outputTy);
}
void MeanOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<MeanOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// ReduceMulOp
//===----------------------------------------------------------------------===//

void ReduceMulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMulOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceMulOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMulOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// ReduceAddOp
//===----------------------------------------------------------------------===//

void ReduceAddOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceAddOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceAddOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceAddOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// ReduceMinOp
//===----------------------------------------------------------------------===//

void ReduceMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

void ReduceMinOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMinOp>(odsBuilder, odsState, input, axis, outputTy);
}

//===----------------------------------------------------------------------===//
// ReduceMaxOp
//===----------------------------------------------------------------------===//

void ReduceMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, TensorValue axis,
                        TensorType outputTy) {
  buildReductionOp<ReduceMaxOp>(odsBuilder, odsState, input, axis, outputTy);
}
void ReduceMaxOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        TensorValue input, int64_t axis, TensorType outputTy) {
  buildReductionOp<ReduceMaxOp>(odsBuilder, odsState, input, axis, outputTy);
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

  auto shapeInput =
      dyn_cast_if_present<ConstantOp>(getNewShape().getDefiningOp());
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

//===----------------------------------------------------------------------===//
// AddSingletonDimOp
//===----------------------------------------------------------------------===//

LogicalResult AddSingletonDimOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType resultTy = getResult().getType();
  int64_t inputRank = inputTy.getRank();
  int64_t resultRank = resultTy.getRank();
  if (resultRank != inputRank + 1)
    return emitOpError("Output rank must be one greater than input.");

  TensorType newResultTy = inputTy.cloneWith(resultTy.getDims());
  if (auto layout = resultTy.getLayoutAttr())
    newResultTy = newResultTy.cloneWith(layout);
  else
    newResultTy = newResultTy.cloneWithoutLayout();

  if (newResultTy != resultTy)
    return emitOpError(
        "More than shape / layout has changed between input and output.");

  // Check surrounding dims around axis are the same
  int64_t axis = getAxis().getLimitedValue();
  for (int64_t i = 0; i < resultRank; i++) {
    if (i < axis) {
      if (resultTy.getDim(i) != inputTy.getDim(i))
        return emitOpError("Input and output dims do not match around axis.");
    } else if (i == axis) {
      if (resultTy.getIntDim(i) != 1)
        return emitOpError("Output dim does not have 1 for `axis`");
    } else {
      // i > axis
      if (resultTy.getDim(i) != inputTy.getDim(i - 1))
        return emitOpError("Input and output dims do not match around axis.");
    }
  }

  return success();
}

LogicalResult SplitDimOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType resultTy = getResult().getType();

  int64_t inputRank = inputTy.getRank();
  int64_t resultRank = resultTy.getRank();
  if (resultRank != inputRank + 1)
    return emitOpError("Output rank must be one greater than input.");

  TensorType newResultTy = inputTy.cloneWith(resultTy.getDims());
  if (auto layout = resultTy.getLayoutAttr())
    newResultTy = newResultTy.cloneWith(layout);
  else
    newResultTy = newResultTy.cloneWithoutLayout();

  if (newResultTy != resultTy)
    return emitOpError(
        "More than shape / layout has changed between input and output.");

  // Check surrounding dims around axis are the same
  int64_t axis = getAxis().getLimitedValue();
  for (int64_t i = 0; i < resultRank; i++) {
    if (i < axis) {
      if (resultTy.getDim(i) != inputTy.getDim(i))
        return emitOpError("Input and output dims do not match around axis.");
    } else if (i > axis + 1) {
      if (resultTy.getDim(i) != inputTy.getDim(i - 1))
        return emitOpError("Input and output dims do not match around axis.");
    }
  }

  // Make sure split dimensions are correct
  if (resultTy.isIntegerDim(axis) && resultTy.isIntegerDim(axis + 1) &&
      inputTy.isIntegerDim(axis)) {
    int64_t origDim = inputTy.getIntDim(axis);
    int64_t splitDimProduct =
        resultTy.getIntDim(axis) * resultTy.getIntDim(axis + 1);

    if (origDim != splitDimProduct)
      return emitOpError("Split dimensions do not multiply to final.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MergeDimOp
//===----------------------------------------------------------------------===//

LogicalResult MergeDimOp::verify() {
  TensorType inputTy = getInput().getType();
  TensorType resultTy = getResult().getType();

  int64_t inputRank = inputTy.getRank();
  int64_t resultRank = resultTy.getRank();
  if (resultRank != inputRank - 1)
    return emitOpError("Output rank must be one less than input.");

  TensorType newResultTy = inputTy.cloneWith(resultTy.getDims());
  if (auto layout = resultTy.getLayoutAttr())
    newResultTy = newResultTy.cloneWith(layout);
  else
    newResultTy = newResultTy.cloneWithoutLayout();

  if (newResultTy != resultTy)
    return emitOpError(
        "More than shape / layout has changed between input and output.");

  // Check surrounding dims around axis are the same
  int64_t axis = getAxis().getLimitedValue();
  for (int64_t i = 0; i < resultRank; i++) {
    if (i < axis) {
      if (resultTy.getDim(i) != inputTy.getDim(i))
        return emitOpError("Input and output dims do not match around axis.");
    } else if (i > axis) {
      if (resultTy.getDim(i) != inputTy.getDim(i + 1))
        return emitOpError("Input and output dims do not match around axis.");
    }
  }

  // Make sure merged dimensions are correct
  if (inputTy.isIntegerDim(axis) && inputTy.isIntegerDim(axis + 1) &&
      resultTy.isIntegerDim(axis)) {
    int64_t newDim = resultTy.getIntDim(axis);
    int64_t mergeDimProduct =
        inputTy.getIntDim(axis) * inputTy.getIntDim(axis + 1);

    if (newDim != mergeDimProduct)
      return emitOpError("Merge dimensions do not multiply to final.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UnsqueezeShapeOp
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

//===----------------------------------------------------------------------===//
// SqueezeShapeOp
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

//===----------------------------------------------------------------------===//
// Static::ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult Static::ReshapeOp::verify() {
  LogicalResult typesCompatible = checkReshapeInOutTypesCompatible(*this);
  if (typesCompatible.failed()) {
    return typesCompatible;
  }

  TensorType resultTy = getResult().getType();
  if (resultTy.hasRank()) {
    for (auto curResDim : resultTy.getTypedShapeAttr().getValues()) {
      auto intCurResDim = dyn_cast<IntegerAttr>(curResDim);
      if (intCurResDim && intCurResDim.getSInt() == -1)
        return emitOpError("static reshape can not have inferred dimensions");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ShapeOfOp
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
  build(odsBuilder, odsState,
        MO::TensorType::get({outputSize}, intTy,
                            getHostDevice(odsBuilder.getContext())),
        input);
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

//===----------------------------------------------------------------------===//
// SliceDimOp
//===----------------------------------------------------------------------===//

LogicalResult SliceDimOp::verify() {
  TensorType inputType = getInput().getType();
  if (inputType.hasRank())
    if ((size_t)getAxis().getSExtValue() >= inputType.getRank())
      return emitError("Slice axis is greater than rank of input");
  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// TileOp
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

//===----------------------------------------------------------------------===//
// InvokeShapeFuncOp
//===----------------------------------------------------------------------===//

LogicalResult InvokeShapeFuncOp::verify() {
  if (llvm::any_of(getOutputParamDecls(), [&](ParamDeclAttr decl) {
        return isa<MOSH::ShapeType>(decl.getType());
      }))
    return emitError() << "dynamic rank isn't supported in shape function";
  return success();
}

//===----------------------------------------------------------------------===//
// LayoutTransformOp
//===----------------------------------------------------------------------===//

LogicalResult LayoutTransformOp::verify() {
  if (!getInput().getType().getLayoutAttr())
    return emitError() << "input type must contain layout annotation";

  if (!getResult().getType().getLayoutAttr())
    return emitError() << "output type must contain layout annotation";

  return success();
}

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

LogicalResult
GuardOp::inferReturnTypes(MLIRContext *context,
                          std::optional<Location> location, Adaptor adaptor,
                          SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes = llvm::to_vector(TypeRange(adaptor.getInputs()));
  return success();
}

ParseResult ChainCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SmallVector<Type, 4> types;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(types))
      return failure();

    if (operands.size() != types.size())
      return parser.emitError(parser.getNameLoc(),
                              "mismatched number of operands and types");

    if (parser.resolveOperands(operands, types, parser.getNameLoc(),
                               result.operands))
      return failure();
  }

  result.types.push_back(ChainType::get(parser.getContext()));

  return success();
}

void ChainCreateOp::print(OpAsmPrinter &p) {
  p << "(";
  p.printOperands(getOperands());
  p << ")";
  p.printOptionalAttrDict(getOperation()->getAttrs());
  if (!getOperands().empty()) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
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
        MOSH::ParamToValueOp::create(rewriter, op->getLoc(), declRefAttr));
  }

  // fan in all the assertion chains as the guard op takes a single chain
  auto guardChain =
      MO::ChainCreateOp::create(rewriter, op->getLoc(), ValueRange{allChains});

  auto guardOp = MO::GuardOp::create(rewriter, op->getLoc(), guardChain,
                                     op->getOperands());

  rewriter.modifyOpInPlace(op, [&guardOp, &op] {
    for (auto &&[replacement, old] :
         llvm::zip(guardOp->getResults(), op->getOpOperands())) {
      old.set(replacement);
    }
  });
}

//===----------------------------------------------------------------------===//
// CustomOp
//===----------------------------------------------------------------------===//
//

/// Checks if an op's operands contains any mutable types: opaque or buffer.
static bool hasMutableOperand(OperandRange operands) {
  return llvm::any_of(operands, [](Value val) {
    return isa<MO::BufferType, MO::OpaqueType>(val.getType());
  });
}

bool CustomOp::isInPlace() {
  // TODO(GEX-1181): design kernel API/registration mechanism to refine this.
  return hasMutableOperand(getOperands()) && maybeGetInChain().has_value() &&
         maybeGetOutChain().has_value();
}

SmallVector<ChainValue> CustomOp::getInChains() {
  return getChainValues(getOperands());
}

SmallVector<ChainValue> CustomOp::getOutChains() {
  return getChainValues(getResults());
}

SmallVector<OpOperand *> CustomOp::getInChainsMutable() {
  SmallVector<OpOperand *> results;

  for (auto &value : getOperandsMutable()) {
    if (isa<ChainType>(value.get().getType()))
      results.push_back(&value);
  }

  return results;
}

std::optional<ChainValue> CustomOp::maybeGetInChain() {
  if (getNumOperands() == 0)
    return std::nullopt;

  auto lastOperand = getOperands().back();
  if (auto chain = dyn_cast<ChainValue>(lastOperand))
    return chain;

  return std::nullopt;
}

std::optional<ChainValue> CustomOp::maybeGetOutChain() {
  if (getNumResults() == 0)
    return std::nullopt;

  auto finalResult = getResults().back();
  if (auto chain = dyn_cast<ChainValue>(finalResult))
    return chain;

  return std::nullopt;
}

mlir::SmallVector<Value> CustomOp::getBufferOperands() {
  SmallVector<Value> result;

  for (Value val : getOperands()) {
    if (isa<MO::BufferType>(val.getType()))
      result.push_back(val);
  }

  return result;
}

void CustomOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  bool isInPlace = this->isInPlace();

  // The mutable spec isn't a required attr at the moment. If it's not set
  // conservatively assume that there's a read and write effect on all mutable
  // operands.
  if (isInPlace) {
    for (auto &&opOperand : getOperation()->getOpOperands()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), &opOperand);
      effects.emplace_back(mlir::MemoryEffects::Write::get(), &opOperand);
    }
  }

  // Handle generic parameterization logic.
  if (!getOutputParamDecls().empty())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

//===----------------------------------------------------------------------===//
// MO Mutable tensor operations
//===----------------------------------------------------------------------===//

void BufferTransferOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getSrcMutable());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), &getDstMutable());
}

void MutableLoadOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), &getInBufferMutable());
}

void MutableStoreOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

void MutableStoreSliceOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       &getInBufferMutable());
}

LogicalResult MutableStoreSliceOp::verify() {

  // Copied wholesale from the mo.slice verifier
  TensorType inputTy = castFromTensorLike(getInBuffer().getType());
  TensorType startTy = getStart().getType();
  TensorType stopTy = getStop().getType();
  TensorType stepTy = getStep().getType();
  int64_t inputRank = inputTy.getRank();
  if (startTy.hasStaticShape() && startTy.getIntDim(0) != inputRank)
    return emitError("start shape must be [inputRank] if it's static");

  if (stopTy.hasStaticShape() && stopTy.getIntDim(0) != inputRank)
    return emitError("stop shape must be [inputRank] if it's static");

  if (stepTy.hasStaticShape() && stepTy.getIntDim(0) != inputRank)
    return emitError("step shape must be [inputRank] if it's static");

  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceSumOp
//===----------------------------------------------------------------------===//

LogicalResult AllReduceSumOp::verify() {
  // Verify we have inputs.
  auto inputs = getInputs();
  if (inputs.empty())
    return emitError("allreduce requires at least one input");

  // Verify that all inputs have the same shape and element type.
  auto firstInputType = getInputTensor(0).getType();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto inputType = getInputTensor(i).getType();
    if (inputType.getTypedShapeAttr() != firstInputType.getTypedShapeAttr())
      return emitError("all inputs must have the same shape");
    if (inputType.getElementType() != firstInputType.getElementType())
      return emitError("all inputs must have the same element type");
  }

  // Verify that output type matches input type (shape and element type).
  auto outputType = getOutput().getType();
  if (outputType.getTypedShapeAttr() != firstInputType.getTypedShapeAttr())
    return emitError("output shape must match input shape");
  if (outputType.getElementType() != firstInputType.getElementType())
    return emitError("output element type must match input element type");

  // Verify that the device attribute matches the output device.
  auto deviceAttr = getDeviceAttr();
  auto outputDevice = outputType.getDeviceRef();
  if (outputDevice && deviceAttr != outputDevice)
    return emitError("device attribute must match output device");

  // Verify signal buffers match input count
  if (getSignalBuffers().size() != inputs.size())
    return emitError("number of signal buffers must match number of inputs");

  return success();
}

void AllReduceSumOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  // Checks input/output counts and that corresponding device types match.
  auto outputs = getOutputs();
  auto inputs = getInputs();
  auto signalBuffers = getSignalBuffers();

  // Check that number of signal buffers matches number of inputs
  if (signalBuffers.size() != inputs.size())
    return emitError(
        llvm::formatv("number of signal buffers should match number of inputs. "
                      "Expected {0} signal buffers but got {1}",
                      inputs.size(), signalBuffers.size()));

  if (outputs.size() != inputs.size() * inputs.size())
    return emitError(llvm::formatv(
        "number of outputs should be num_inputs * num_inputs for all-gather. "
        "Expected {0} outputs but got {1} (num_inputs={2})",
        inputs.size() * inputs.size(), outputs.size(), inputs.size()));
  return success();
}

void AllGatherOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
}

//===----------------------------------------------------------------------===//
// MatmulAllReduceOp
//===----------------------------------------------------------------------===//

void MatmulAllReduceOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  MutableOperandRange signalBuffers = getSignalBuffersMutable();

  for (auto &sig : signalBuffers)
    effects.emplace_back(mlir::MemoryEffects::Write::get(), &sig);
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
