//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MOP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOPDialect/MOPOps.h"
#include "GraphRT/BEFDialect/BEFOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Returns the align constraint for the given attribute, or none if it
/// has no such constraint.
static FailureOr<Optional<int64_t>>
getImpliedAlignment(ElementsAttr value,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (auto bufferAttr = dyn_cast<BufferAttr>(value))
    return bufferAttr.getAlign();

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    if (!blob)
      return emitError() << "requires the attribute resource to be initialized";
    if (blob->getData().size() != static_cast<size_t>(value.getNumElements()))
      // TODO: Why is this not enforced by the dense_resource attribute?
      return emitError() << "the attribute resource size does not match the "
                            "attribute elements size"
                         << " (" << blob->getData().size() << " vs "
                         << value.getNumElements() << ")";
    size_t blobAlign = blob->getDataAlignment();
    if (blobAlign == 0)
      // No align constraint.
      return {{}};
    // We have an align constraint.
    return Optional<int64_t>(static_cast<int64_t>(blobAlign));
  }

  // No align constraint.
  return {{}};
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
static LogicalResult
verifyBufferConstantOp(BufferConstantOp &op,
                       function_ref<InFlightDiagnostic()> emitError) {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = op.getValue();
  if (value.isSplat())
    return emitError() << "does not support splat attributes";
  ShapedType shape = value.getType();
  if (!shape.hasRank() || shape.getShape().size() != 1)
    return emitError() << "only supports rank-1 attributes";
  Type elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitError() << "only supports attributes of uint8_t elements"
                       << " (have " << elemType << ")";

  // The resulting constant's type must be sized, and the size must match
  // the attribute's size.
  BufferType bufferType = op.getResult().getType();
  if (!bufferType.getSize())
    return emitError() << "requires the buffer type have a fixed size";
  if (*bufferType.getSize() != value.getNumElements())
    return emitError() << "requires the buffer type have a size matching the "
                          "attribute size"
                       << " (" << *bufferType.getSize() << " vs "
                       << value.getNumElements() << ")";

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint.
  Optional<int64_t> typeAlign = bufferType.getAlign();
  FailureOr<Optional<int64_t>> failOrValueAlign =
      getImpliedAlignment(value, emitError);
  if (failed(failOrValueAlign))
    return failOrValueAlign;
  Optional<int64_t> valueAlign = *failOrValueAlign;
  if (typeAlign.has_value() != valueAlign.has_value())
    return emitError() << "requires the buffer type have an align "
                          "constraint if and only "
                          "if the attribute has an align constraint";
  if (typeAlign.has_value() && *typeAlign != *valueAlign)
    return emitError() << "requires the buffer type align match the "
                          "attribute align"
                       << " (" << *typeAlign << " vs " << *valueAlign << ")";

  return success();
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  return verifyBufferConstantOp(*this,
                                [this]() { return this->emitOpError(); });
}

//===----------------------------------------------------------------------===//
// ContextNewOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

/// Returns the implicit signature of the 'init' entry point.
static FunctionType implicitInitSignature(OpBuilder &builder) {
  return builder.getFunctionType(
      {}, {builder.getType<BEF::ChainType>(), builder.getType<ContextType>()});
}

/// Returns the implicit signature of the 'execute' entry point, before
/// any explicit inputs and results have been included.
static FunctionType implicitExecuteSignature(OpBuilder &builder) {
  return builder.getFunctionType(
      {builder.getType<BEF::ChainType>(), builder.getType<ContextType>()},
      {builder.getType<BEF::ChainType>()});
}

/// Returns true if requiredTypes is a prefix of actualTypes, in which case
/// the prefix is removed from actualTypes.
static bool checkAndRemovePrefix(ArrayRef<Type> requiredTypes,
                                 SmallVectorImpl<Type> &actualTypes) {
  // CAUTION: We are assuming post C++14 semantics for mismatch.
  auto [mismatchRequired, mismatchActual] = std::mismatch(
      requiredTypes.begin(), requiredTypes.end(), actualTypes.begin());
  if (mismatchRequired != requiredTypes.end())
    return false;
  actualTypes.erase(actualTypes.begin(),
                    actualTypes.begin() + requiredTypes.size());
  return true;
}

/// Creates a new model with empty 'init' and 'execute' bodies and with
/// `inputTypes` and `resultTypes` for the op's explicit signature.
/// `inputArgLocs` specifies the location information for the explicit
/// arguments to the 'execute' entry point, and thus should have the same
/// length as `inputTypes`.
void ModelOp::build(OpBuilder &builder, OperationState &state,
                    StringAttr sym_name, ArrayRef<Type> inputTypes,
                    ArrayRef<Type> resultTypes, ArrayRef<Location> inputArgLocs,
                    ArrayRef<NamedAttribute> extraAttrs) {
  build(builder, state, sym_name,
        DictionaryAttr::get(state.getContext(), extraAttrs),
        builder.getFunctionType(inputTypes, resultTypes));

  // The 'init' entry point has no arguments.
  assert(implicitInitSignature(builder).getInputs().empty());
  Region *initRegion = state.addRegion();
  initRegion->emplaceBlock();

  // The 'execute' entry point has both implicit and explicit arguments.
  Region *executeRegion = state.addRegion();
  executeRegion->emplaceBlock();
  for (auto type : implicitExecuteSignature(builder).getInputs())
    executeRegion->addArgument(type, state.location);
  for (auto [type, loc] : llvm::zip(inputTypes, inputArgLocs))
    executeRegion->addArgument(type, loc);
}

/// Parses a model.
///
/// `mop.model` symbol function-attributes?
///   `init` init-body
///   `execute` function-signature execute-body
///
ParseResult ModelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbol;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> resultTypes;
  llvm::SMLoc argLoc;
  Region *initRegion = result.addRegion();
  Region *executeRegion = result.addRegion();

  // We're using the standard symbol attribute name.
  assert(getSymNameAttrName(result.name) ==
         mlir::SymbolTable::getSymbolAttrName());
  mlir::NamedAttrList extraAttrs;

  if (parser.parseSymbolName(symbol, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDict(extraAttrs) ||

      // The init region has no explicit arguments.
      parser.parseKeyword("init") ||
      parser.parseRegion(*initRegion, {}, false) ||

      // The execute region should have explicit argument and result types.
      parser.parseKeyword("execute") || parser.getCurrentLocation(&argLoc) ||
      parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true) ||
      parser.parseArrowTypeList(resultTypes) ||
      parser.parseRegion(*executeRegion, args, true))
    return failure();

  result.addAttribute(getExtraAttrsAttrName(result.name),
                      DictionaryAttr::get(parser.getContext(), extraAttrs));

  // Calculate the signature for 'execute'. The first two operands must always
  // be chain and context, but they are not included in the model's function
  // signature. Similarly the first result must always be chani, but is also
  // not included in the signature.
  OpBuilder builder(parser.getContext());
  FunctionType implicitExecSig = implicitExecuteSignature(builder);
  SmallVector<Type> inputTypes(executeRegion->getArgumentTypes());
  if (!checkAndRemovePrefix(implicitExecSig.getInputs(), inputTypes))
    return parser.emitError(argLoc)
           << "expects the 'execute' region arguments to begin with types "
           << implicitExecSig.getInputs();
  if (!checkAndRemovePrefix(implicitExecSig.getResults(), resultTypes))
    return parser.emitError(argLoc)
           << "expects the 'execute' region results to begin with types "
           << implicitExecSig.getResults();

  result.addAttribute(getInputResultTypesAttrName(result.name),
                      TypeAttr::get(parser.getBuilder().getFunctionType(
                          inputTypes, resultTypes)));

  return success();
}

/// Prints the model. Collects argument and result types and passes them to
/// helper functions. Drops "void" result since it cannot be parsed back.
void ModelOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName((*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName()));
  p.printOptionalAttrDict(getExtraAttrsAttr().getValue());
  p << " init ";
  p.printRegion(getInit());
  p << " execute (";
  llvm::interleaveComma(getExecuteBlock()->getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printArrowTypeList(getExecuteBlock()->getTerminator()->getOperandTypes());
  p << " ";
  p.printRegion(getExecute(), false);
}

/// Verifies the init and execute regions of the model.
LogicalResult ModelOp::verifyRegions() {
  OpBuilder builder(getContext());

  // The type of the init region is
  //  () -> (!bef.chain, !mop.context)
  Block *initBlock = getInitBlock();
  if (auto output = dyn_cast<BEF::OutputOp>(initBlock->getTerminator())) {
    FunctionType implicitInitSig = implicitInitSignature(builder);
    assert(implicitInitSig.getInputs().empty());
    if (!initBlock->getArgumentTypes().empty())
      return emitOpError()
             << "expects no arguments to the 'init' body, but got "
             << initBlock->getArgumentTypes() << ".";

    if (failed(output.checkArgumentTypes(implicitInitSig.getResults())))
      return emitOpError()
             << "expects the output types of the 'init' body to be "
             << implicitInitSig.getResults() << ".";
  }

  // The type of the execute region is:
  //   (!bef.chain, !mop.context, arg_1, ... arg_n) ->
  //   (!bef.chain, res_1, ... res_m)
  // where the ModelOp's inputOutputTypes are:
  //   (arg_1, ... arg_n) -> (res_1, ... res_m)
  Block *execBlock = getExecuteBlock();
  if (auto output = dyn_cast<BEF::OutputOp>(execBlock->getTerminator())) {
    FunctionType implicitExecSig = implicitExecuteSignature(builder);
    ArrayRef<Type> modelArgTypes = getInputResultTypes().getInputs();

    SmallVector<Type> expectedArgTypes(implicitExecSig.getInputs());
    expectedArgTypes.append(modelArgTypes.begin(), modelArgTypes.end());
    if (execBlock->getArgumentTypes() != expectedArgTypes)
      return emitOpError() << "expects the arguments of the "
                              "'execute' body to be "
                           << expectedArgTypes << ", but have "
                           << execBlock->getArgumentTypes() << ".";

    ArrayRef<Type> modelResultTypes = getInputResultTypes().getResults();
    SmallVector<Type> expectedResultTypes(implicitExecSig.getResults());
    expectedResultTypes.append(modelResultTypes.begin(),
                               modelResultTypes.end());

    if (failed(output.checkArgumentTypes(expectedResultTypes)))
      return emitOpError()
             << "expected the results of the 'execute' body to be  "
             << expectedResultTypes << ", but have " << output->getResultTypes()
             << ".";
  }
  return success();
}

/// Verifies the input/output type signature.
LogicalResult ModelOp::verify() {
  // Ensure that ModelOp's arguments and results are ShapedTypes.
  // TODO: In short term, should replace ShapedType with !mop.tensor once
  // implemented since that will be our representation for M::Tensor.
  // TODO: In longer term, generalize executable model's calling convention
  // to support generic arguments and results.
  FunctionType modelType = getInputResultTypes();
  for (size_t i = 0, e = modelType.getNumInputs(); i < e; ++i) {
    Type type = modelType.getInput(i);
    if (!type.isa<ShapedType>())
      return emitOpError()
             << "expects the 'execute' entry point to take arguments "
                "of ShapedType, but argument "
             << i << " is of type " << type << ".";
  }
  for (size_t i = 0, e = modelType.getNumResults(); i < e; ++i) {
    Type type = modelType.getResult(i);
    if (!type.isa<ShapedType>())
      return emitOpError()
             << "expects the 'execute' entry point to return results "
                "of ShapedType, but result "
             << i << " is of type " << type << ".";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MOPDialect op support
//===----------------------------------------------------------------------===//

void MOPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MOPDialect/MOP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MOPDialect/MOP.cpp.inc"
