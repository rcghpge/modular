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
    return emitOpError() << "expected the buffer type have a fixed size";
  if (*bufferType.getSize() != static_cast<uint64_t>(value.getNumElements()))
    return emitOpError() << "expected the buffer type have a size matching the "
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
// BufferAllocOp
//===----------------------------------------------------------------------===//

LogicalResult BufferAllocOp::verify() {
  BufferType resultBufferType = getResult().getType();
  BufferType impliedBufferType = BufferType::get(getContext(), getSize());
  if (resultBufferType != impliedBufferType)
    return emitOpError()
           << "expected result buffer type to match type implied by operannds ("
           << resultBufferType << " vs " << impliedBufferType << ").";
  const bool haveRuntimeSize = getRuntimeSize() != nullptr;
  const bool haveStaticSize = resultBufferType.getSize().has_value();
  if (haveRuntimeSize == haveStaticSize)
    return emitOpError()
           << "expected a runtime size argument or a size attribute.";
  if (!isValidSize(getSize()))
    return emitOpError() << "expected size to be positive.";
  if (!isValidAlign(getAlign()))
    return emitOpError() << "expected align to be a power of two.";
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferAllocOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_or_null<mlir::IntegerAttr>(
      attributes.get(getSizeAttrName(name)));
  inferredReturnTypes.emplace_back(BufferType::get(context, sizeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferReallocOp
//===----------------------------------------------------------------------===//

// Ideally BufferReallocOp would just extend BufferAllocOp with the slot
// and output index attributes. Instead we have this code dup.

LogicalResult BufferReallocOp::verify() {
  BufferType resultBufferType = getResult().getType();
  BufferType impliedBufferType = BufferType::get(getContext(), getSize());
  if (resultBufferType != impliedBufferType)
    return emitOpError() << "expected result buffer type " << resultBufferType
                         << " to match " << impliedBufferType
                         << " given size argument";
  const bool haveRuntimeSize = getRuntimeSize() != nullptr;
  const bool haveStaticSize = resultBufferType.getSize().has_value();
  if (haveRuntimeSize == haveStaticSize)
    return emitOpError()
           << "expected a runtime size argument or a size attribute.";
  if (!isValidSize(getSize()))
    return emitOpError() << "expected size to be positive.";
  if (!isValidAlign(getAlign()))
    return emitOpError() << "expected align to be a power of two.";
  return success();
}

/// Constructs the buffer type result from the given buffer type attribute.
LogicalResult BufferReallocOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_or_null<mlir::IntegerAttr>(
      attributes.get(getSizeAttrName(name)));
  inferredReturnTypes.emplace_back(BufferType::get(context, sizeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// SpecCreateOp
//===----------------------------------------------------------------------===//

/// Parses a !mop.spec type where the element type must be known, but
/// represented in 'raw' attribute form.
static ParseResult parseRawShapeAndDType(AsmParser &p,
                                         IntArrayElementsAttr &rawShape,
                                         IntegerAttr &rawDType) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mop.spec type.";
  if (!tensorSpec.getDtype())
    return p.emitError(loc)
           << "expected !mop.spec type to have a known element type.";

  rawShape = tensorSpec.getRawShape();
  rawDType = tensorSpec.getRawDType();

  return success();
}

/// Prints a !mop.spec type, but represented in 'raw' attribute form.
static void printRawShapeAndDType(AsmPrinter &p, const SpecCreateOp &op,
                                  IntArrayElementsAttr rawShape,
                                  IntegerAttr rawDType) {
  p << TensorSpecType::get(rawShape, rawDType);
}

LogicalResult SpecCreateOp::verify() {
  auto impliedSpecType = TensorSpecType::getChecked(
      [this]() { return emitOpError(); }, getContext(), getFstRawShape(),
      getSndRawDType());
  if (impliedSpecType == nullptr)
    return failure();
  TensorSpecType actualSpecType = getResult().getType();
  if (actualSpecType != impliedSpecType)
    return emitOpError()
           << "expected result type is " << actualSpecType
           << ", but static dimensions and element type arguments "
              "suggest it should be "
           << impliedSpecType << ".";
  if (getRuntimeDims().size() != impliedSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension operands to match " << actualSpecType
                         << ", but given " << getRuntimeDims().size() << ".";
  return success();
}

LogicalResult SpecCreateOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawShapeAttr = dyn_cast<IntArrayElementsAttr>(
      attributes.get(getFstRawShapeAttrName(name)));
  auto rawDTypeAttr =
      dyn_cast<IntegerAttr>(attributes.get(getSndRawDTypeAttrName(name)));
  assert(rawShapeAttr && rawDTypeAttr && "mismatched operand type");
  inferredReturnTypes.emplace_back(
      TensorSpecType::get(rawShapeAttr, rawDTypeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecCreateDynamicOp
//===----------------------------------------------------------------------===//

/// Parses a !mop.spec type where the element type is unknown, but represented
/// in 'raw' attribute form.
static ParseResult parseRawShape(AsmParser &p, IntArrayElementsAttr &rawShape) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mop.spec type.";
  if (tensorSpec.getDtype())
    return p.emitError(loc)
           << "expected !mop.spec type to have an unknown element type.";

  rawShape = tensorSpec.getRawShape();

  return success();
}

/// Prints a !mop.spec type, but represented in 'raw' attribute form.
static void printRawShape(AsmPrinter &p, const SpecCreateDynamicOp &op,
                          IntArrayElementsAttr rawShape) {
  p << TensorSpecType::get(rawShape, /*rawDTypeOrNull=*/{});
}

LogicalResult SpecCreateDynamicOp::verify() {
  auto impliedSpecType =
      TensorSpecType::getChecked([this]() { return emitOpError(); },
                                 getContext(), getRawShape(), /*rawDType=*/{});
  if (impliedSpecType == nullptr)
    return failure();
  TensorSpecType actualSpecType = getResult().getType();
  if (actualSpecType != impliedSpecType)
    return emitOpError() << "expected result type is " << actualSpecType
                         << ", but static dimensions suggest it should be "
                         << impliedSpecType << ".";
  if (getRuntimeDims().size() != impliedSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension operands to match " << actualSpecType
                         << ", but given " << getRuntimeDims().size() << ".";
  return success();
}

LogicalResult SpecCreateDynamicOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawShapeAttr =
      dyn_cast<IntArrayElementsAttr>(attributes.get(getRawShapeAttrName(name)));
  assert(rawShapeAttr && "mismatched operand type");
  inferredReturnTypes.emplace_back(
      TensorSpecType::get(rawShapeAttr, /*rawDTypeOrNull=*/{}));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecExtractShapeOp
//===----------------------------------------------------------------------===//

/// Parses a TensorSpecType, however also sets the rawShape to capture it's
/// shape in 'raw' attribute form.
static ParseResult
parseTensorSpecTypeWithRawShape(AsmParser &p, Type &specType,
                                IntArrayElementsAttr &rawShape) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(specType))
    return failure();
  auto tensorSpec = dyn_cast<TensorSpecType>(specType);
  if (tensorSpec == nullptr)
    p.emitError(loc) << "expected !mop.spec type";
  rawShape = tensorSpec.getRawShape();
  return success();
}

static void printTensorSpecTypeWithRawShape(AsmPrinter &p,
                                            const SpecDimsDynamicOp &op,
                                            Type specType,
                                            IntArrayElementsAttr rawShape) {
  p << specType;
}

LogicalResult SpecDimsDynamicOp::verify() {
  TensorSpecType actualSpecType = getSpec().getType();
  TensorSpecType impliedSpecType = TensorSpecType::get(
      getContext(), getRawShape(), actualSpecType.getDtype());
  if (actualSpecType != impliedSpecType)
    return emitOpError() << "expected spec type is " << actualSpecType
                         << ", but static dimensions arguments "
                            "suggest it should be "
                         << impliedSpecType << ".";
  if (getResults().size() != actualSpecType.getNumDynamicDims())
    return emitOpError() << "expected " << actualSpecType.getNumDynamicDims()
                         << " dimension results to match the tensor spec "
                         << actualSpecType << ", but given "
                         << getResults().size();
  return success();
}

LogicalResult SpecDimsDynamicOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto specType = dyn_cast<TensorSpecType>(operands[0].getType());
  assert(specType && "mismatched operand type");
  for (size_t i = 0, e = specType.getNumDynamicDims(); i < e; ++i)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// SpecSizeOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// SpecNumElemsOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// SpecDTypeOp
//===----------------------------------------------------------------------===//

// Nothing to implement.

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

/// Checks the shape, buffer type and result tensor types are consistent.
LogicalResult TensorCreateOp::verify() {
  TensorSpecType tensorSpecType = getTensorSpec().getType();
  BufferType actualBufferType = getBuffer().getType();
  BufferType expectedBufferType = tensorSpecType.getExpectedBufferType();
  TensorType tensorType = getResult().getType();

  if (tensorSpecType != tensorType.getTensorSpec())
    return emitOpError() << "expected shape to match result tensor shape"
                         << " (" << tensorSpecType << " vs "
                         << tensorType.getTensorSpec() << ")";
  if (actualBufferType != expectedBufferType)
    return emitOpError()
           << "expected buffer type to match type implied by tensor spec ("
           << actualBufferType << " vs " << expectedBufferType << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// TensorExtractOp
//===----------------------------------------------------------------------===//

LogicalResult TensorExtractOp::verify() {
  TensorType tensorType = getTensor().getType();
  if (tensorType.getTensorSpec() != getTensorSpecResult().getType())
    return emitOpError() << "expected the result shape to match tensor's shape"
                         << " (" << getTensorSpecResult().getType() << " vs "
                         << tensorType.getTensorSpec() << ")";
  BufferType impliedBufferType =
      tensorType.getTensorSpec().getExpectedBufferType();
  if (impliedBufferType != getBufferResult().getType())
    return emitOpError() << "expected the result buffer to match the type "
                            "implied by the tensor type ("
                         << getBufferResult().getType() << " vs "
                         << impliedBufferType << ").";
  return success();
}

LogicalResult TensorExtractOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorType = dyn_cast<TensorType>(operands[0].getType());
  assert(tensorType && "mismatched operand type");
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec());
  inferredReturnTypes.emplace_back(
      tensorType.getTensorSpec().getExpectedBufferType());
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
