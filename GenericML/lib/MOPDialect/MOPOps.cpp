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
#include "Support/ErrorOr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Returns the align constraint for the given attribute, or none if it
/// has no such constraint.
static ErrorOr<Optional<int64_t>> getImpliedAlignment(ElementsAttr value) {
  if (auto bufferAttr = dyn_cast<BufferAttr>(value))
    return bufferAttr.getAlign();

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    Optional<int64_t> alignFromType;
    if (auto bufferType = dyn_cast<BufferType>(resourceAttr.getType()))
      alignFromType = bufferType.getAlign();

    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    // It appears resource blobs are not always initialized when verifying.
    // However if the blob is available then we can check its size and
    // alignment agree with what we expect.
    if (blob) {
      if (static_cast<int64_t>(blob->getData().size()) !=
          value.getNumElements())
        // TODO: Why is this not enforced by the dense_resource attribute?
        return Error(Twine("the attribute resource size does not match the "
                           "attribute elements size (" +
                           Twine(blob->getData().size()) + " vs " +
                           Twine(value.getNumElements()) + ")"));
      if (alignFromType &&
          *alignFromType != static_cast<int64_t>(blob->getDataAlignment()))
        return Error(Twine("the attribute resource alignment does not match "
                           "the alignment specified by the buffer type (" +
                           Twine(blob->getDataAlignment()) + " vs " +
                           Twine(*alignFromType) + ")"));
    }
    return alignFromType;
  }

  // No align constraint.
  return Optional<int64_t>();
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getValue();
  if (value.isSplat())
    return emitOpError() << "does not support splat attributes";
  ShapedType shape = value.getType();
  if (!shape.hasRank() || shape.getShape().size() != 1)
    return emitOpError() << "only supports rank-1 attributes";
  Type elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitOpError() << "only supports attributes of uint8_t elements"
                         << " (have " << elemType << ")";

  // The resulting constant's type must be sized, and the size must match
  // the attribute's size.
  BufferType bufferType = getResult().getType();
  if (!bufferType.getSize())
    return emitOpError() << "requires the buffer type have a fixed size";
  if (*bufferType.getSize() != value.getNumElements())
    return emitOpError() << "requires the buffer type have a size matching the "
                            "attribute size"
                         << " (" << *bufferType.getSize() << " vs "
                         << value.getNumElements() << ")";

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint. Check they are valid.
  ErrorOr<Optional<int64_t>> errorOrValueAlign = getImpliedAlignment(value);
  if (auto *error = errorOrValueAlign.getError())
    return emitOpError() << error;
  Optional<int64_t> valueAlign = *errorOrValueAlign;

  // If the value is aligned then the result type may include that
  // alignment. However, if the value is unaligned no such align
  // constraint can appear in the result type.
  Optional<int64_t> typeAlign = bufferType.getAlign();
  if (valueAlign) {
    if (typeAlign && typeAlign.has_value() && *typeAlign != *valueAlign)
      return emitOpError() << "requires the buffer type align match the "
                              "attribute align"
                           << " (" << *typeAlign << " vs " << *valueAlign
                           << ")";
  } else {
    if (typeAlign)
      return emitOpError()
             << "requires the buffer type to not have an align "
                "constraint since the attribute has no align constraint";
  }

  return success();
}

/// Infers the buffer type result from the attribute argument.
LogicalResult BufferConstantOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.empty() && "mismatched operand arity");
  auto attr = attributes.get(getValueAttrName(name));
  auto elementsAttr = dyn_cast_or_null<ElementsAttr>(attr);
  assert(elementsAttr && "mismatched operand types");
  int64_t size = elementsAttr.size();
  // The inferred result type does not propagate any align constraint.
  inferredReturnTypes.emplace_back(
      BufferType::get(context, size, /*align=*/{}));
  return success();
}

Optional<int64_t> BufferConstantOp::getAlign() {
  ErrorOr<Optional<int64_t>> errorOrAlign = getImpliedAlignment(getValue());
  assert(!errorOrAlign.isError());
  if (errorOrAlign.isError())
    return {};
  return *errorOrAlign;
}

//===----------------------------------------------------------------------===//
// ContextNewOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// TensorPackOp
//===----------------------------------------------------------------------===//

/// Checks the shape, buffer type and result tensor types are consistent.
LogicalResult TensorPackOp::verify() {
  TensorShapeType tensorShapeType = getTensorShape().getType();
  BufferType bufferType = getBuffer().getType();
  TensorType tensorType = getResult().getType();

  if (tensorShapeType != tensorType.getTensorShape())
    return emitOpError() << "requires shape to match result tensor shape"
                         << " (" << tensorShapeType << " vs "
                         << tensorType.getTensorShape() << ")";
  int64_t staticSize = tensorType.getSizeInBytes();
  if ((staticSize != ShapedType::kDynamicSize) !=
      bufferType.getSize().has_value())
    return emitOpError() << "requires the buffer to be sized if and only if "
                            "the tensor shape is fully static";
  if (bufferType.getSize() && staticSize != *bufferType.getSize())
    return emitOpError()
           << "requires the buffer size to match the tensor size in bytes"
           << " (" << *bufferType.getSize() << " vs " << staticSize << ")";
  return success();
}

DTypeType TensorPackOp::getDType() { return getResult().getType().getDType(); }

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
