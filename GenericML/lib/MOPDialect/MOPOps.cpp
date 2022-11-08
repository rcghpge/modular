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
#include "./ParsingUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/TypeUtilities.h"

#include "./Utils.h"

using namespace M;
using namespace MOP;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

/// Returns the align constraint for the given attribute, or none if it
/// has no such constraint.
static ErrorOr<Optional<uint64_t>> getImpliedAlignment(ElementsAttr value) {
  // TODO(#4424): Would be better to just look for a HasAlignmentAttrInterface
  // instead of looking for specific attribute types.
  if (auto alignedBufferAttr = dyn_cast<AlignedBytesAttr>(value))
    return alignedBufferAttr.getAlign();

  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    if (!blob)
      return Error("the resource blob must be initialized so that its "
                   "alignment and size can be determined.");

    if (static_cast<int64_t>(blob->getData().size()) != value.getNumElements())
      // TODO: Why is this not enforced by the dense_resource attribute?
      return Error(Twine("the resource size does not match the "
                         "attribute's element size (" +
                         Twine(blob->getData().size()) + " vs " +
                         Twine(value.getNumElements()) + ")"));
    Optional<uint64_t> align =
        blob->getDataAlignment() == 0
            ? Optional<uint64_t>()
            : static_cast<uint64_t>(blob->getDataAlignment());
    if (!isValidAlign(align))
      return Error(
          Twine("the resource alignment must be a power of two (have " +
                Twine(blob->getDataAlignment()) + ")."));
  }

  // No align constraint.
  return Optional<uint64_t>();
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getValue();
  if (value.isSplat())
    return emitOpError() << "does not support splat attributes";
  ShapedType shapedType = value.getType();
  if (!shapedType.hasRank() || shapedType.getShape().size() != 1)
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
  ErrorOr<Optional<uint64_t>> errorOrValueAlign = getImpliedAlignment(value);
  if (auto *error = errorOrValueAlign.getError())
    return emitOpError() << error;

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
  inferredReturnTypes.emplace_back(BufferType::get(context, size));
  return success();
}

Optional<uint64_t> BufferConstantOp::getAlign() {
  ErrorOr<Optional<uint64_t>> errorOrAlign = getImpliedAlignment(getValue());
  assert(!errorOrAlign.isError());
  if (errorOrAlign.isError())
    return {};
  return *errorOrAlign;
}

//===----------------------------------------------------------------------===//
// BufferCreateOp
//===----------------------------------------------------------------------===//

LogicalResult BufferCreateOp::verify() {
  BufferType bufferType = getResult().getType();
  const bool haveBufferSize = haveSize(bufferType.getSize());
  const bool haveRuntimeSize = getRuntimeSize() != nullptr;
  const bool haveStaticSize = haveSize(getSize());
  if (haveRuntimeSize == haveStaticSize)
    return emitOpError()
           << "expected a runtime size argument or a size attribute.";
  if (haveStaticSize != haveBufferSize)
    return emitOpError() << "expected result type to have a size constraint if "
                            "and only if a size attribute is given.";
  if (haveStaticSize && getSize() != *bufferType.getSize())
    return emitOpError()
           << "expected result type size to match attribute size ("
           << *bufferType.getSize() << " vs " << getSize() << ").";
  if (!isValidSize(getSize()))
    return emitOpError() << "expected size to be positive.";
  if (!isValidAlign(getAlign()))
    return emitOpError() << "expected align to be a power of two.";
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferCreateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_or_null<mlir::IntegerAttr>(
      attributes.get(getSizeAttrName(name)));
  Optional<int64_t> size =
      haveSize(sizeAttr) ? Optional<int64_t>(sizeAttr.getSInt()) : llvm::None;
  inferredReturnTypes.emplace_back(BufferType::get(context, size));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferRecreateOp
//===----------------------------------------------------------------------===//

// Ideally BufferRecreateOp would just extend BufferCreateOp with the slot
// and output index attributes. Instead we have this code dup.

LogicalResult BufferRecreateOp::verify() {
  BufferType bufferType = getResult().getType();
  const bool haveBufferSize = haveSize(bufferType.getSize());
  const bool haveRuntimeSize = getRuntimeSize() != nullptr;
  const bool haveStaticSize = haveSize(getSize());
  if (haveRuntimeSize == haveStaticSize)
    return emitOpError()
           << "expected a runtime size argument or a size attribute.";
  if (haveStaticSize != haveBufferSize)
    return emitOpError() << "expected result type to have a size constraint if "
                            "and only if a size attribute is given.";
  if (haveStaticSize && getSize() != *bufferType.getSize())
    return emitOpError()
           << "expected result type size to match attribute size ("
           << *bufferType.getSize() << " vs " << getSize() << ").";
  if (!isValidSize(getSize()))
    return emitOpError() << "expected size to be positive.";
  if (!isValidAlign(getAlign()))
    return emitOpError() << "expected align to be a power of two.";
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferRecreateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_or_null<mlir::IntegerAttr>(
      attributes.get(getSizeAttrName(name)));
  Optional<int64_t> size =
      haveSize(sizeAttr) ? Optional<int64_t>(sizeAttr.getSInt()) : llvm::None;
  inferredReturnTypes.emplace_back(BufferType::get(context, size));
  return success();
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// SpecWrapOp
//===----------------------------------------------------------------------===//

/// Parses a dtype, but represented as an MLIR TypeAttr.
static ParseResult parseDTypeAttr(AsmParser &p, TypeAttr &elementTypeAttr) {
  FailureOr<Type> elementType;
  if (parseDType(p, elementType))
    return failure();
  elementTypeAttr = TypeAttr::get(*elementType);
  return success();
}

/// Prints a dtype, but represented as an MLIR TypeAtr.
void printDTypeAttr(AsmPrinter &p, const SpecWrapOp &op,
                    TypeAttr elementTypeAttr) {
  printDType(p, elementTypeAttr.getValue());
}

LogicalResult SpecWrapOp::verify() {
  TensorSpecType actualSpecType = getResult().getType();
  if (mlirTypeToDType(getElementType()).isInvalid())
    return emitOpError() << "cannot represent " << getElementType()
                         << " as a DType.";
  TensorSpecType impliedSpecType =
      TensorSpecType::get(getContext(), getDims(), getElementType());
  if (actualSpecType != impliedSpecType)
    return emitOpError()
           << "expected result type is " << actualSpecType
           << ", but static dimensions and element type arguments "
              "suggest it should be "
           << impliedSpecType << ".";
  if (static_cast<int64_t>(getRuntimeDims().size()) !=
      impliedSpecType.getNumDynamicDims())
    return emitOpError() << "requires " << actualSpecType.getNumDynamicDims()
                         << " dimension operands to match the spec "
                         << actualSpecType << ", but given "
                         << getRuntimeDims().size() << ".";
  return success();
}

LogicalResult
SpecWrapOp::inferReturnTypes(MLIRContext *context, Optional<Location> location,
                             ValueRange operands, DictionaryAttr attributes,
                             mlir::RegionRange regions,
                             SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto dimsAttr =
      dyn_cast<IntArrayElementsAttr>(attributes.get(getDimsAttrName(name)));
  auto elementTypeAttr =
      dyn_cast<TypeAttr>(attributes.get(getElementTypeAttrName(name)));
  inferredReturnTypes.emplace_back(TensorSpecType::get(
      context, dimsAttr.asArrayRef<int64_t>(), elementTypeAttr.getValue()));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecUnwrapOp
//===----------------------------------------------------------------------===//

LogicalResult SpecUnwrapOp::verify() {
  TensorSpecType actualSpecType = getSpec().getType();
  TensorSpecType impliedSpecType = TensorSpecType::get(
      getContext(), getDims(), actualSpecType.getElementType());
  if (actualSpecType != impliedSpecType)
    return emitOpError() << "expected spec type is " << actualSpecType
                         << ", but static dimensions arguments "
                            "suggest it should be "
                         << impliedSpecType << ".";
  if (static_cast<int64_t>(getResults().size()) !=
      actualSpecType.getNumDynamicDims())
    return emitOpError() << "requires " << actualSpecType.getNumDynamicDims()
                         << " dimension results to match the tensor spc "
                         << actualSpecType << ", but given "
                         << getResults().size();
  return success();
}

LogicalResult SpecUnwrapOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.size() == 1 && "mismatched operand arity");
  auto specType = dyn_cast<TensorSpecType>(operands[0].getType());
  assert(specType && "mismatched operand type");
  for (int64_t i = 0, e = specType.getNumDynamicDims(); i < e; ++i)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecSizeOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// TensorWrapOp
//===----------------------------------------------------------------------===//

/// Checks the shape, buffer type and result tensor types are consistent.
LogicalResult TensorWrapOp::verify() {
  TensorSpecType tensorSpecType = getTensorSpec().getType();
  BufferType bufferType = getBuffer().getType();
  TensorType tensorType = getResult().getType();

  if (tensorSpecType != tensorType.getTensorSpec())
    return emitOpError() << "requires shape to match result tensor shape"
                         << " (" << tensorSpecType << " vs "
                         << tensorType.getTensorSpec() << ")";
  Optional<int64_t> staticSize = tensorType.getSizeInBytes();
  if (staticSize.has_value() != bufferType.getSize().has_value())
    return emitOpError() << "requires the buffer to be sized if and only if "
                            "the tensor shape is fully static";
  if (bufferType.getSize() && *staticSize != *bufferType.getSize())
    return emitOpError()
           << "requires the buffer size to match the tensor size in bytes"
           << " (" << *bufferType.getSize() << " vs " << *staticSize << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// TensorUnwrapOp
//===----------------------------------------------------------------------===//

LogicalResult TensorUnwrapOp::verify() {
  TensorType tensorType = getTensor().getType();
  if (tensorType.getTensorSpec() != getTensorSpecResult().getType())
    return emitOpError() << "requires the result shape to match tensor's shape"
                         << " (" << getTensorSpecResult().getType() << " vs "
                         << tensorType.getTensorSpec() << ")";
  Optional<int64_t> sizeInBytes = tensorType.getSizeInBytes();
  if (sizeInBytes.has_value() !=
      getBufferResult().getType().getSize().has_value())
    return emitOpError() << "requires the result buffer to have a size if and "
                            "only if the tensor has a fully static shape";
  if (sizeInBytes.has_value() &&
      *sizeInBytes != *getBufferResult().getType().getSize())
    return emitOpError() << "requires the result buffer to have a size "
                            "matching the byte size of the tensor"
                         << " (" << *getBufferResult().getType().getSize()
                         << " vs " << *sizeInBytes << ")";
  return success();
}

LogicalResult TensorUnwrapOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorType = dyn_cast<TensorType>(operands[0].getType());
  assert(tensorType && "mismatched operand type");
  Optional<int64_t> sizeInBytes = tensorType.getSizeInBytes();
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec());
  inferredReturnTypes.emplace_back(BufferType::get(context, sizeInBytes));
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
