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
    // If the blob is available then we can check its size and alignment
    // agree with what we expect.
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
// BufferCreateOp
//===----------------------------------------------------------------------===//

LogicalResult BufferCreateOp::verify() {
  BufferType bufferType = getBufferType();
  const bool haveRuntimeSize = getRuntimeSize() != nullptr;
  if (bufferType.getSize().has_value() == haveRuntimeSize) {
    return emitOpError()
           << "requires a runtime size argument if and only if the buffer size "
              "is unknown in the given buffer type";
  }
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferCreateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto bufferTypeAttr =
      dyn_cast<TypeAttr>(attributes.get(getBufferTypeAttrName(name)));
  assert(bufferTypeAttr && "mismatched operand types");
  auto bufferType = dyn_cast<BufferType>(bufferTypeAttr.getValue());
  assert(bufferType && "mismatched operand types");
  inferredReturnTypes.emplace_back(bufferType.forgetAlign());
  return success();
}

Optional<int64_t> BufferCreateOp::getAlign() {
  return getBufferType().getAlign();
}

//===----------------------------------------------------------------------===//
// BufferRecreateOp
//===----------------------------------------------------------------------===//

LogicalResult BufferRecreateOp::verify() {
  BufferType bufferType = getBufferType();
  if (bufferType.getSize().has_value() == (getRuntimeSize() != nullptr)) {
    return emitOpError()
           << "requires a runtime size argument if and only if the buffer size "
              "is unknown in the given buffer type";
  }
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferRecreateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto bufferTypeAttr =
      dyn_cast<TypeAttr>(attributes.get(getBufferTypeAttrName(name)));
  assert(bufferTypeAttr && "mismatched operand types");
  auto bufferType = dyn_cast<BufferType>(bufferTypeAttr.getValue());
  assert(bufferType && "mismatched operand types");
  inferredReturnTypes.emplace_back(bufferType.forgetAlign());
  return success();
}

Optional<int64_t> BufferRecreateOp::getAlign() {
  return getBufferType().getAlign();
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// ShapePackOp
//===----------------------------------------------------------------------===//

LogicalResult ShapePackOp::verify() {
  TensorShapeType shapeType = getResult().getType();
  if (static_cast<int64_t>(getRuntimeDims().size()) !=
      shapeType.getNumDynamicDims())
    return emitOpError() << "requires " << shapeType.getNumDynamicDims()
                         << " dimension operands to match the shape "
                         << shapeType << ", but given "
                         << getRuntimeDims().size();
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeUnpackOp
//===----------------------------------------------------------------------===//

LogicalResult ShapeUnpackOp::verify() {
  TensorShapeType shapeType = getShape().getType();
  if (static_cast<int64_t>(getResults().size()) !=
      shapeType.getNumDynamicDims())
    return emitOpError() << "requires " << shapeType.getNumDynamicDims()
                         << " dimension results to match the shape "
                         << shapeType << ", but given " << getResults().size();
  return success();
}

LogicalResult ShapeUnpackOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.size() == 1 && "mismatched operand arity");
  auto shapeType = dyn_cast<TensorShapeType>(operands[0].getType());
  assert(shapeType && "mismatched operand type");
  for (int64_t i = 0, e = shapeType.getNumDynamicDims(); i < e; ++i)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// ShapeSizeOp
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

DTypeType TensorPackOp::getDType() { return getResult().getType().getDType(); }

//===----------------------------------------------------------------------===//
// TensorUnpackOp
//===----------------------------------------------------------------------===//

LogicalResult TensorUnpackOp::verify() {
  TensorType tensorType = getTensor().getType();
  if (tensorType.getTensorShape() != getTensorShapeResult().getType())
    return emitOpError() << "requires the result shape to match tensor's shape"
                         << " (" << getTensorShapeResult().getType() << " vs "
                         << tensorType.getTensorShape() << ")";
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

LogicalResult TensorUnpackOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorType = dyn_cast<TensorType>(operands[0].getType());
  assert(tensorType && "mismatched operand type");
  Optional<int64_t> sizeInBytes = tensorType.getSizeInBytes();
  inferredReturnTypes.emplace_back(tensorType.getTensorShape());
  inferredReturnTypes.emplace_back(
      BufferType::get(context, sizeInBytes, /*align=*/{}));
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
