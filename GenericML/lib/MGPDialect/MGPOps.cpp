//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MGP dialect operations.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "Cache/CacheDialect/CacheAttrs.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/DType.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// CpuBufferConstantOp
//===----------------------------------------------------------------------===//

/// Returns the alignment constraint for the given attribute, or none if it
/// has no such constraint.
///
/// TODO(#4424): Would be better to just look for a HasAlignmentAttrInterface
/// instead of looking for specific attribute types.
static ErrorOr<std::optional<uint64_t>>
getImpliedAlignment(ElementsAttr value) {
  if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(value)) {
    DenseResourceElementsHandle handle = resourceAttr.getRawHandle();
    mlir::AsmResourceBlob *blob = handle.getBlob();
    if (!blob && !isa<AlignedBytesType>(resourceAttr.getType())) {
      // Report a user-friendly error in this situation.
      return Error(Twine("dense_resource blob is not available, so require "
                         "type annotation to supply an alignment"));
    }

    if (blob) {
      // Since the blob is available, check its size is consistent with the
      // attribute size.
      if (static_cast<int64_t>(blob->getData().size()) !=
          value.getNumElements())
        return Error(Twine("the resource size does not match the "
                           "attribute's element size (" +
                           Twine(blob->getData().size()) + " vs " +
                           Twine(value.getNumElements()) + ")"));
    }
    // DenseResourceElementsAttrs support HasAlignedBytesInterface, so fall
    // through.
  }

  if (auto hasAlignedBytesAttr = dyn_cast<HasAlignedBytesInterface>(value)) {
    // The attribute has an alignment or default alignment we can use.
    return hasAlignedBytesAttr.getAlignedBytesType().getAlign();
  }

  if (auto cacheConstant = dyn_cast<Cache::ConstantHashAttr>(value)) {
    // The attribute encodes its own (optional) alignment.
    return cacheConstant.getOptAlign();
  }

  return Error("cannot supply an alignment from type annotation for "
               "this attribute value.");
}

/// Parses the elements attribute, but also captures any alignment constraint.
static ParseResult parseElementsAttrWithAlign(AsmParser &p,
                                              ElementsAttr &aValue,
                                              IntegerAttr &bRawAlign) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseAttribute(aValue))
    return failure();

  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(aValue);
  if (const char *err = errOrOptAlign.getError())
    return p.emitError(loc) << StringRef(err);

  Type ui64 = IntegerType::get(p.getContext(), 64, IntegerType::Unsigned);
  APInt apint(ui64.getIntOrFloatBitWidth(), optSizeToRawSize(*errOrOptAlign),
              ui64.isSignedInteger());
  bRawAlign = IntegerAttr::get(ui64, apint);

  return success();
}

/// Prints the elements attribute. The alignment attribute is not printed.
static void printElementsAttrWithAlign(AsmPrinter &p,
                                       const CpuBufferConstantOp &op,
                                       ElementsAttr aValue,
                                       IntegerAttr bRawAlign) {
  p << aValue;
}

/// Get the raw alignment for elements when building a buffer constant.
static uint64_t getRawAlign(ElementsAttr value) {
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (errOrOptAlign.isError())
    return 0;

  // If error we'll report it when the result is validated.
  return optSizeToRawSize(*errOrOptAlign);
}

void CpuBufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                ElementsAttr value) {
  build(odsBuilder, odsState, value, getRawAlign(value));
}

void CpuBufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                Type type, ElementsAttr value) {
  build(odsBuilder, odsState, type, value, getRawAlign(value));
}

OpFoldResult CpuBufferConstantOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  assert(operands.empty() && "constant has no operands");
  return getAValue();
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult CpuBufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getAValue();
  ShapedType shapedType = cast<ShapedType>(value.getType());
  if (value.size() != shapedType.getNumElements())
    return emitOpError() << "does not support splat attributes.";
  if (!shapedType.hasRank() || shapedType.getShape().size() != 1)
    return emitOpError() << "only supports rank-1 attributes";
  Type elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitOpError() << "only supports attributes of uint8_t elements"
                         << " (have " << elemType << ").";

  // The resulting constant's type must be sized, and the size must match
  // the attribute's size.
  CpuBufferType bufferType = getResult().getType();
  if (!bufferType.getOptSize())
    return emitOpError() << "expected the buffer type have a fixed size";
  if (*bufferType.getOptSize() != static_cast<uint64_t>(value.getNumElements()))
    return emitOpError() << "expected the buffer type have a size matching the "
                            "attribute size"
                         << " (" << *bufferType.getOptSize() << " vs "
                         << value.getNumElements() << ").";

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint. Check they are valid.
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (auto *error = errOrOptAlign.getError())
    return emitOpError() << StringRef(error);

  if (!isValidOptAlign(*errOrOptAlign))
    return emitOpError()
           << "the attribute alignment must be a power of two (have "
           << **errOrOptAlign << ").";

  // The implied alignment must match the actual attribute alignment.
  if (getOptAlign().has_value() != errOrOptAlign->has_value())
    return emitOpError() << "expected alignment operand if and only if value "
                            "operand has alignment";
  if (getOptAlign().has_value() && *getOptAlign() != **errOrOptAlign)
    return emitOpError()
           << "expected operand alignment to match attribute's alignment"
           << " (" << *getOptAlign() << " vs " << **errOrOptAlign << ").";

  return success();
}

/// Infers the buffer type result from the attribute argument.
LogicalResult CpuBufferConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.empty() && "mismatched operand arity");
  auto elementsAttr = dyn_cast_if_present<ElementsAttr>(
      attributes.get(getAValueAttrName(name)));
  assert(elementsAttr && "mismatched operand types");
  int64_t size = elementsAttr.size();
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, size));
  return success();
}

std::optional<uint64_t> CpuBufferConstantOp::getOptAlign() {
  return rawSizeToOptSize(getBRawAlign());
}

//===----------------------------------------------------------------------===//
// CpuBufferCachedOp
//===----------------------------------------------------------------------===//

/// Parses a !mgp.cpu.buffer type where the (optional) buffer size is
/// represented in 'raw form'.
static ParseResult parseRawSizeAsBufferType(AsmParser &p,
                                            IntegerAttr &rawSize) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto bufferType = dyn_cast<CpuBufferType>(type);
  if (bufferType == nullptr)
    return p.emitError(loc) << "expected a !mgp.cpu.buffer type.";
  rawSize = bufferType.getRawSize();
  return success();
}

/// Prints a !mgp.cpu.buffer type, but represented in 'raw form'.
static void printRawSizeAsBufferType(MLIRContext *context, AsmPrinter &p,
                                     IntegerAttr rawSize) {
  p << CpuBufferType::get(context, rawSize);
}

static void printRawSizeAsBufferType(AsmPrinter &p,
                                     const CpuBufferGetCachedOp &op,
                                     IntegerAttr rawSize) {
  printRawSizeAsBufferType(op->getContext(), p, rawSize);
}

/// Infers the buffer type result from the size argument.
LogicalResult CpuBufferGetCachedOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawSizeAttr = dyn_cast_if_present<IntegerAttr>(
      attributes.get(getBRawSizeAttrName(name)));
  assert(rawSizeAttr && "mismatched operand types");
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, rawSizeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuBufferAlloc* Utils
//===----------------------------------------------------------------------===//

/// Verify any of the 6 alloc primitives.
static LogicalResult verifyAnyAllocOp(
    MLIRContext *context, llvm::function_ref<InFlightDiagnostic()> emitOpError,
    std::optional<uint64_t> optSize, uint64_t rawAlign,
    IndexType optRuntimeSizeType, CpuTensorSpecType optRuntimeSpecType,
    CpuBufferType resultBufferType) {
  // First validate our use of the util.
  assert((optRuntimeSizeType != nullptr ? 1 : 0) +
                 (optRuntimeSpecType != nullptr ? 1 : 0) +
                 (optSize.has_value() ? 1 : 0) ==
             1 &&
         "expecting exactly one of static size, runtime size, or runtime spec");

  // Check the size and align static values.
  if (!isValidOptSize(optSize))
    return emitOpError() << "expected size to be positive.";
  if (!isValidRawAlign(rawAlign))
    return emitOpError() << "expected align to be a power of two.";

  // Check the result buffer type is consistent.
  CpuBufferType impliedBufferType = CpuBufferType::get(context, optSize);
  if (resultBufferType != impliedBufferType)
    return emitOpError() << "expected result buffer type to match type "
                            "implied by operands ("
                         << resultBufferType << " vs " << impliedBufferType
                         << ").";

  // Check the tensor spec operand type is consistent.
  if (optRuntimeSpecType != nullptr) {
    if (!optRuntimeSpecType.hasRank())
      return emitOpError()
             << "expected tensor spec to have statically known rank.";

    std::optional<uint64_t> bytes = optRuntimeSpecType.getSizeInBytes();
    if (bytes)
      return emitOpError() << "expected tensor spec to not have statically "
                              "known size (have "
                           << optRuntimeSpecType << ").";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocStaticOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferAllocStaticOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/getASize(), /*rawAlign=*/getBRawAlign(),
      /*optRuntimeSizeType=*/{},
      /*optRuntimeSpecType=*/{}, getResult().getType());
}

/// Constructs the buffer type result using statically known size.
LogicalResult CpuBufferAllocStaticOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_if_present<mlir::IntegerAttr>(
      attributes.get(getASizeAttrName(name)));
  assert(sizeAttr != nullptr && "mismatched operand attributes");
  uint64_t size = sizeAttr.getUInt();
  // If the attribute happens to be invalid we'll still construct a result
  // type, but the verify will reject the overall op.
  std::optional<uint64_t> optSize =
      isValidPlainSize(size) ? size : std::optional<uint64_t>();
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, optSize));
  return success();
}

void CpuBufferAllocStaticOp::setInChain(Value chain) { setOperand(chain); }

//===----------------------------------------------------------------------===//
// CpuBufferAllocDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferAllocDynamicOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/{}, /*rawAlign=*/getRawAlign(),
      /*optRuntimeSizeType=*/getRuntimeSize().getType(),
      /*optRuntimeSpecType=*/{}, getResult().getType());
}

LogicalResult CpuBufferAllocDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.emplace_back(
      CpuBufferType::get(context, std::optional<uint64_t>()));
  return success();
}

void CpuBufferAllocDynamicOp::setInChain(Value chain) { setOperand(0, chain); }

//===----------------------------------------------------------------------===//
// CpuBufferAllocSpecOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferAllocSpecOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/{}, /*rawAlign=*/getRawAlign(),
      /*optRuntimeSizeType=*/{},
      /*optRuntimeSpecType=*/getRuntimeSpec().getType(), getResult().getType());
}

LogicalResult CpuBufferAllocSpecOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.emplace_back(
      CpuBufferType::get(context, std::optional<uint64_t>()));
  return success();
}

void CpuBufferAllocSpecOp::setInChain(Value chain) { setOperand(0, chain); }

//===----------------------------------------------------------------------===//
// CpuBufferReallocStaticOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferReallocStaticOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/getCSize(), /*rawAlign=*/getDRawAlign(),
      /*optRuntimeSizeType=*/{},
      /*optRuntimeSpecType=*/{}, getResult().getType());
}

/// Constructs the buffer type result from the statically known size.
LogicalResult CpuBufferReallocStaticOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr = dyn_cast_if_present<mlir::IntegerAttr>(
      attributes.get(getCSizeAttrName(name)));
  assert(sizeAttr != nullptr && "mismatched operand attributes");
  std::optional<uint64_t> size = sizeAttr.getUInt();
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, size));
  return success();
}

void CpuBufferReallocStaticOp::setInChain(Value chain) { setOperand(0, chain); }

//===----------------------------------------------------------------------===//
// CpuBufferReallocDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferReallocDynamicOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/{}, /*rawAlign=*/getCRawAlign(),
      /*optRuntimeSizeType=*/getRuntimeSize().getType(),
      /*optRuntimeSpecType=*/{}, getResult().getType());
}

LogicalResult CpuBufferReallocDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.emplace_back(
      CpuBufferType::get(context, std::optional<uint64_t>()));
  return success();
}

void CpuBufferReallocDynamicOp::setInChain(Value chain) {
  setOperand(0, chain);
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocSpecOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferReallocSpecOp::verify() {
  return verifyAnyAllocOp(
      getContext(), [this]() { return emitOpError(); },
      /*optSize=*/{}, /*rawAlign=*/getCRawAlign(),
      /*optRuntimeSizeType=*/{},
      /*optRuntimeSpecType=*/getRuntimeSpec().getType(), getResult().getType());
}

LogicalResult CpuBufferReallocSpecOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.emplace_back(
      CpuBufferType::get(context, std::optional<uint64_t>()));
  return success();
}

void CpuBufferReallocSpecOp::setInChain(Value chain) { setOperand(0, chain); }

//===----------------------------------------------------------------------===//
// CpuBufferConcatOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferConcatOp::verify() {
  if (!getOutBuffer().getType().getOptSize())
    // Nothing to check.
    return success();
  uint64_t inTotalSize = 0;
  for (const auto &inBuffer : getInBuffers()) {
    auto inBufferType = dyn_cast<CpuBufferType>(inBuffer.getType());
    std::optional<uint64_t> optSize = inBufferType.getOptSize();
    if (!optSize)
      // Not all input buffers have a statically known size.
      return success();
    inTotalSize += *optSize;
  }
  CpuBufferType expectedOutType =
      CpuBufferType::get(getContext(), std::optional<uint64_t>(inTotalSize));
  if (getOutBuffer().getType() != expectedOutType)
    return emitOpError() << "expected output buffer type does not match actual "
                            "output buffer type ("
                         << expectedOutType << " vs "
                         << getOutBuffer().getType() << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// CpuBufferSectionBorrowStaticOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferSectionBorrowStaticOp::verify() {
  if (!getInBuffer().getType().getOptSize().has_value()) {
    return emitOpError() << "expected input buffer of type '"
                         << getInBuffer().getType()
                         << "' to have a statically known size.";
  }
  uint64_t inSize = *getInBuffer().getType().getOptSize();

  if (getAOffset() > inSize) {
    return emitOpError() << "requesting section offset " << getAOffset()
                         << " which is beyond end of input buffer of size "
                         << inSize << ".";
  }

  if (getAOffset() + getBSize() > inSize) {
    return emitOpError() << "requesting section end "
                         << getAOffset() + getBSize()
                         << " which is beyond end of input buffer of size "
                         << inSize << ".";
  }

  if (!getOutBuffer().getType().getOptSize().has_value()) {
    return emitOpError() << "expected output buffer of type '"
                         << getOutBuffer().getType()
                         << "' to have a statically known size.";
  }
  uint64_t outSize = *getOutBuffer().getType().getOptSize();

  if (getBSize() != outSize) {
    return emitOpError() << "expected output buffer of type '"
                         << getOutBuffer().getType() << "' to have size "
                         << getBSize() << ".";
  }

  return success();
}

LogicalResult CpuBufferSectionBorrowStaticOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto sizeAttr =
      dyn_cast_if_present<IntegerAttr>(attributes.get(getBSizeAttrName(name)));
  if (sizeAttr)
    inferredReturnTypes.emplace_back(CpuBufferType::get(context, sizeAttr));
  else
    // Invalid, but will fail in verify.
    inferredReturnTypes.emplace_back(CpuBufferType::get(context, std::nullopt));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuBufferSectionBorrowDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult CpuBufferSectionBorrowDynamicOp::verify() {
  if (getInBuffer().getType().getOptSize().has_value()) {
    return emitOpError() << "expected input buffer of type '"
                         << getInBuffer().getType()
                         << "' to have a statically unknown size.";
  }
  if (getOutBuffer().getType().getOptSize().has_value()) {
    return emitOpError() << "expected output buffer of type '"
                         << getOutBuffer().getType()
                         << "' to have a statically unknown size.";
  }
  return success();
}

LogicalResult CpuBufferSectionBorrowDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, std::nullopt));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuBufferSplitBorrowOp
//===----------------------------------------------------------------------===//

/// Parses a list of mgp.cpu.buffer types, but also extracts the raw sizes
/// of each type.
static ParseResult
parseBufferTypesAndRawSizes(AsmParser &p, SmallVectorImpl<Type> &bufferTypes,
                            IntArrayElementsAttr &rawSizes) {
  SmallVector<APInt> rawDims;
  if (p.parseArrowTypeList(bufferTypes))
    return failure();
  rawDims.reserve(bufferTypes.size());
  for (const auto &type : bufferTypes) {
    auto bufferType = dyn_cast<CpuBufferType>(type);
    rawDims.emplace_back(bufferType.getRawSize().getValue());
  }
  Type ui64 = IntegerType::get(p.getContext(), 64, IntegerType::Unsigned);
  rawSizes = IntArrayElementsAttr::get(
      ArrayType::get(static_cast<int64_t>(rawDims.size()), ui64), rawDims);
  return success();
}

/// Prints a list of mgp.cpu.buffer types, ignoring the raw sizes.
static void
printBufferTypesAndRawSizes(AsmPrinter &p, const CpuBufferSplitBorrowOp &op,
                            mlir::ValueTypeRange<mlir::ResultRange> bufferTypes,
                            IntArrayElementsAttr rawSizes) {
  p.printArrowTypeList(bufferTypes);
}

// Verifies the output buffer sizes are consistent with the input buffer size,
// and that we have a runtime size operand for every output buffer of unknown
// size.
LogicalResult CpuBufferSplitBorrowOp::verify() {
  size_t numDynamic = llvm::count_if(getOutBuffers(), [](const Value &v) {
    return !dyn_cast<CpuBufferType>(v.getType()).getOptSize().has_value();
  });
  if (getRuntimeSizes().size() != numDynamic)
    return emitOpError() << "expected a runtime size operand for every output "
                            "buffer without a statically known size (given "
                         << getRuntimeSizes().size() << " but need "
                         << numDynamic << ").";

  if (!getInBuffer().getType().getOptSize())
    // Nothing more to check
    return success();
  uint64_t outTotalSize = 0;
  for (const auto &outBuffer : getOutBuffers()) {
    auto outBufferType = dyn_cast<CpuBufferType>(outBuffer.getType());
    std::optional<uint64_t> optSize = outBufferType.getOptSize();
    if (!optSize)
      // Not all output buffers have a statically known size.
      return success();
    outTotalSize += *optSize;
  }
  CpuBufferType expectedInType =
      CpuBufferType::get(getContext(), std::optional<uint64_t>(outTotalSize));
  if (getInBuffer().getType() != expectedInType)
    return emitOpError() << "expected input buffer type does not match "
                            "actual input buffer type ("
                         << expectedInType << " vs " << getInBuffer().getType()
                         << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateOp
//===----------------------------------------------------------------------===//

void CpuTensorSpecCreateOp::build(OpBuilder &odsBuilder,
                                  OperationState &odsState,
                                  CpuTensorSpecType spec,
                                  ValueRange runtimeDims) {
  assert(spec.hasRank() && "can only construct an mgp.cpu.tensor_spec.create "
                           "for tensor specs of statically known rank");
  CpuTensorSpecCreateOp::build(odsBuilder, odsState,
                               /*runtimeDims=*/runtimeDims,
                               /*aRawDims=*/spec.getRawDims(),
                               /*bRawDType=*/spec.getRawDType());
}

/// Parses a !mgp.cpu.tensor_spec type where the element type must be known,
/// but represented in 'raw form'.
static ParseResult parseRawDimsAndDType(AsmParser &p,
                                        IntArrayElementsAttr &rawDims,
                                        IntegerAttr &rawDType) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<CpuTensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mgp.cpu.tensor_spec type.";
  if (!tensorSpec.hasRank())
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type to have a "
                               "statically known rank.";
  if (!tensorSpec.getOptDType())
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type to have a "
                               "known element type.";

  rawDims = tensorSpec.getRawDims();
  rawDType = tensorSpec.getRawDType();

  return success();
}

/// Prints a !mgp.cpu.tensor_spec type, but represented in 'raw form'.
static void printRawDimsAndDType(AsmPrinter &p, const CpuTensorSpecCreateOp &op,
                                 IntArrayElementsAttr rawDims,
                                 IntegerAttr rawDType) {
  p << CpuTensorSpecType::get(rawDims, rawDType);
}

LogicalResult CpuTensorSpecCreateOp::verify() {
  auto impliedSpecType = CpuTensorSpecType::getChecked(
      [this]() { return emitOpError(); }, getContext(), getARawDims(),
      getBRawDType());
  if (impliedSpecType == nullptr)
    return failure();
  assert(impliedSpecType.hasRank());
  CpuTensorSpecType actualSpecType = getResult().getType();
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

LogicalResult CpuTensorSpecCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawDimsAttr = dyn_cast_if_present<IntArrayElementsAttr>(
      attributes.get(getARawDimsAttrName(name)));
  auto rawDTypeAttr = dyn_cast_if_present<IntegerAttr>(
      attributes.get(getBRawDTypeAttrName(name)));
  assert(rawDimsAttr && rawDTypeAttr && "mismatched operand type");
  inferredReturnTypes.emplace_back(
      CpuTensorSpecType::get(rawDimsAttr, rawDTypeAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateDynamicOp
//===----------------------------------------------------------------------===//

/// Parses a !mgp.cpu.tensor_spec type where the element type is unknown, but
/// represented in 'raw form'.
static ParseResult parseRawDims(AsmParser &p, IntArrayElementsAttr &rawDims) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto tensorSpec = dyn_cast<CpuTensorSpecType>(type);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mgp.cpu.tensor_spec type.";
  if (!tensorSpec.hasRank())
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type to have a "
                               "statically known rank.";
  if (tensorSpec.getOptDType())
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type to have an "
                               "unknown element type.";

  rawDims = tensorSpec.getRawDims();

  return success();
}

/// Prints a !mgp.cpu.tensor_spec type, but represented in 'raw form'.
static void printRawDims(AsmPrinter &p, const CpuTensorSpecCreateDynamicOp &op,
                         IntArrayElementsAttr rawDims) {
  p << CpuTensorSpecType::get(rawDims, /*rawDTypeOrNull=*/{});
}

LogicalResult CpuTensorSpecCreateDynamicOp::verify() {
  auto impliedSpecType = CpuTensorSpecType::getChecked(
      [this]() { return emitOpError(); }, getContext(), getRawDims(),
      /*rawDType=*/{});
  if (impliedSpecType == nullptr)
    return failure();
  CpuTensorSpecType actualSpecType = getResult().getType();
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

LogicalResult CpuTensorSpecCreateDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawDimsAttr = dyn_cast_if_present<IntArrayElementsAttr>(
      attributes.get(getRawDimsAttrName(name)));
  assert(rawDimsAttr && "mismatched operand type");
  inferredReturnTypes.emplace_back(
      CpuTensorSpecType::get(rawDimsAttr, /*rawDTypeOrNull=*/{}));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecWithDTypeOp
//===----------------------------------------------------------------------===//

/// Parses a !mgp.cpu.tensor_spec type where the element type is extracted
/// into the rawDType attribute.
static ParseResult parseRawDType(AsmParser &p, Type &tensorSpec,
                                 IntegerAttr &rawDType) {
  Type type;
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(type))
    return failure();
  auto parsedTensorSpec = dyn_cast<CpuTensorSpecType>(type);
  if (parsedTensorSpec == nullptr)
    return p.emitError(loc) << "expected a !mgp.cpu.tensor_spec type.";
  if (!parsedTensorSpec.getOptDType())
    return p.emitError(loc)
           << "expected an !mgp.cpu.tensor_spec type with known dtype.";
  // The parsed type gives us the requested dtype, which we capture in the
  // rawDType.
  Type ui8 = IntegerType::get(p.getContext(), 8, IntegerType::Unsigned);
  APInt apint(ui8.getIntOrFloatBitWidth(),
              DTypeType::asRawDType(parsedTensorSpec.getOptDType()),
              ui8.isSignedInteger());
  rawDType = IntegerAttr::get(ui8, apint);
  // The tensor spec operand, however, must have an unknown dtype.
  tensorSpec = CpuTensorSpecType::get(
      p.getContext(), parsedTensorSpec.getOptDims(), std::nullopt);
  return success();
}

/// Prints a !mgp.cpu.tensor_spec type.
static void printRawDType(AsmPrinter &p, const CpuTensorSpecWithDTypeOp op,
                          CpuTensorSpecType tensorSpec, IntegerAttr rawDType) {
  // Print as the tensor spec operand type, but with dtype taken from rawDType.
  p.getStream() << CpuTensorSpecType::get(
      op->getContext(), tensorSpec.getOptDims(),
      DTypeType::fromRawDType(rawDType.getUInt()));
}

LogicalResult CpuTensorSpecWithDTypeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorSpec = dyn_cast<CpuTensorSpecType>(operands[0].getType());
  assert(tensorSpec && "mismatched operand type");
  auto rawDTypeAttr = dyn_cast_if_present<IntegerAttr>(
      attributes.get(getRawDTypeAttrName(name)));
  assert(rawDTypeAttr && "mismatched operand type");
  // Result type is the tensor spec operand type but with rawDType as its dtype.
  inferredReturnTypes.emplace_back(
      CpuTensorSpecType::get(context, tensorSpec.getOptDims(),
                             DTypeType::fromRawDType(rawDTypeAttr.getUInt())));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDimsDynamicOp
//===----------------------------------------------------------------------===//

/// Parses a !mgp.cpu.tensor_spec type, however also sets the rawDims to
/// capture it's dimensions in 'raw form'.
static ParseResult
parseTensorSpecTypeWithRawDims(AsmParser &p, Type &specType,
                               IntArrayElementsAttr &rawDims) {
  llvm::SMLoc loc;
  if (p.getCurrentLocation(&loc) || p.parseType(specType))
    return failure();
  auto tensorSpec = dyn_cast<CpuTensorSpecType>(specType);
  if (tensorSpec == nullptr)
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type.";
  if (!tensorSpec.hasRank())
    return p.emitError(loc) << "expected !mgp.cpu.tensor_spec type to have a "
                               "statically known rank.";
  rawDims = tensorSpec.getRawDims();
  return success();
}

static void printTensorSpecTypeWithRawDims(AsmPrinter &p,
                                           const CpuTensorSpecDimsDynamicOp &op,
                                           Type specType,
                                           IntArrayElementsAttr rawDims) {
  p << specType;
}

LogicalResult CpuTensorSpecDimsDynamicOp::verify() {
  CpuTensorSpecType actualSpecType = getSpec().getType();
  CpuTensorSpecType impliedSpecType = CpuTensorSpecType::get(
      getContext(), getRawDims(), actualSpecType.getRawDType().getUInt());
  assert(impliedSpecType.hasRank());
  if (actualSpecType != impliedSpecType)
    return emitOpError() << "expected tensor spec type is " << actualSpecType
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

LogicalResult CpuTensorSpecDimsDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto specType = dyn_cast<CpuTensorSpecType>(operands[0].getType());
  assert(specType && "mismatched operand type");
  for (size_t i = 0, e = specType.getNumDynamicDims(); i < e; ++i)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorCreateOp
//===----------------------------------------------------------------------===//

/// Parses the expected result tensor type, but also derives the appropriate
/// buffer operand's type based on an buffer type annotation.
static ParseResult parseMayBeDynamicBuffer(AsmParser &parser,
                                           Type &bufferRawType,
                                           Type &resultRawType) {
  mlir::SMLoc loc;
  if (parser.getCurrentLocation(&loc))
    return failure();
  OptionalParseResult parseResult = parser.parseOptionalType(bufferRawType);
  bool hasBufferTypeAnnotation = parseResult.has_value();
  if (hasBufferTypeAnnotation) {
    if (failed(parseResult.value()))
      return failure();
    if (!isa<CpuBufferType>(bufferRawType))
      return parser.emitError(loc) << "expecting !mgp.cpu.buffer type";
  }

  if (parser.parseColon() || parser.parseType(resultRawType))
    return failure();
  auto resultTensorType = dyn_cast<CpuTensorType>(resultRawType);
  if (resultTensorType == nullptr)
    return parser.emitError(loc) << "expecting !mgp.cpu.tensor type";
  if (!hasBufferTypeAnnotation) {
    // No buffer type annotation -- buffer operand's type is uniquely determined
    // from the result tensor type.
    bufferRawType = resultTensorType.getTensorSpec().getExpectedBufferType();
  }
  return success();
}

/// Prints the actual result tensor type, but also prints an optional buffer
/// type annotation if the actual buffer type suggests it.
static void printMayBeDynamicBuffer(AsmPrinter &printer,
                                    const CpuTensorCreateOp &op,
                                    CpuBufferType bufferType,
                                    CpuTensorType resultType) {
  if (resultType.getTensorSpec().getExpectedBufferType() != bufferType)
    printer.printType(bufferType);
  printer.getStream() << " : ";
  printer.printType(resultType);
}

/// Checks the tensor spec, buffer type and result tensor types are
/// consistent.
LogicalResult CpuTensorCreateOp::verify() {
  CpuTensorSpecType tensorSpecType = getTensorSpec().getType();
  CpuBufferType actualBufferType = getBuffer().getType();
  CpuBufferType expectedBufferType = tensorSpecType.getExpectedBufferType();
  CpuTensorType tensorType = getResult().getType();

  if (tensorSpecType != tensorType.getTensorSpec())
    return emitOpError() << "expected spec to match result tensor spec"
                         << " (" << tensorSpecType << " vs "
                         << tensorType.getTensorSpec() << ")";

  bool actualBufferIsStatic = actualBufferType.getOptSize().has_value();
  bool expectedBufferIsStatic = expectedBufferType.getOptSize().has_value();
  if (actualBufferIsStatic != expectedBufferIsStatic) {
    // This is the 'dynamic' case, where either the input or the output buffer
    // has unknown size. This can happen when lowering reshapes involving
    // dynamic dimensions, eg:
    //    reshape %x, %y : (tensor<?x?xf32>, tensor<2xui32>) -> tensor<8x16xf32>
    //    reshape %x, %y : (tensor<8x4xf32>, tensor<2xui32>) -> tensor<?x?xf32>
    // Here the buffer backing the tensor with dynamic shape will have unknown
    // size, and in the first case, we wish to assert the input buffer has size
    // 512 bytes at runtime when constructing the result tensor.
    // no-op
  } else if (actualBufferType != expectedBufferType) {
    return emitOpError()
           << "expected buffer type to match type implied by tensor spec ("
           << actualBufferType << " vs " << expectedBufferType << ").";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorExtractOp
//===----------------------------------------------------------------------===//

LogicalResult CpuTensorExtractOp::verify() {
  CpuTensorType tensorType = getTensor().getType();
  if (tensorType.getTensorSpec() != getTensorSpecResult().getType())
    return emitOpError() << "expected the result spec to match tensor's spec"
                         << " (" << getTensorSpecResult().getType() << " vs "
                         << tensorType.getTensorSpec() << ")";
  CpuBufferType impliedBufferType =
      tensorType.getTensorSpec().getExpectedBufferType();
  if (impliedBufferType != getBufferResult().getType())
    return emitOpError() << "expected the result buffer to match the type "
                            "implied by the tensor type ("
                         << getBufferResult().getType() << " vs "
                         << impliedBufferType << ").";
  return success();
}

LogicalResult CpuTensorExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorType = dyn_cast<CpuTensorType>(operands[0].getType());
  assert(tensorType && "mismatched operand type");
  inferredReturnTypes.emplace_back(
      tensorType.getTensorSpec().getExpectedBufferType());
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec());
  return success();
}

//===----------------------------------------------------------------------===//
// CpuTensorExtractShapeOp
//===----------------------------------------------------------------------===//

LogicalResult CpuTensorExtractShapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  assert(operands.size() == 1 && "mismatched operand arity");
  auto tensorType = dyn_cast<CpuTensorType>(operands[0].getType());
  assert(tensorType && "mismatched operand type");
  // Result type is tensor operand's tensor spec type but with unknown dtype.
  inferredReturnTypes.emplace_back(CpuTensorSpecType::get(
      context, tensorType.getTensorSpec().getOptDims(), std::nullopt));
  return success();
}

//===----------------------------------------------------------------------===//
// CASLookupOp
//===----------------------------------------------------------------------===//

/// Parse string from base64 encoding.
static ParseResult parseBase64(AsmParser &parser, StringAttr &stringAttr) {
  std::vector<char> outBytes;
  outBytes.reserve(32);
  if (parser.parseBase64Bytes(&outBytes))
    return failure();
  std::string str(outBytes.data(), outBytes.size());
  stringAttr = StringAttr::get(parser.getContext(), str);
  return success();
}

/// Print string in base64 encoding.
static void printBase64(AsmPrinter &printer, const CASLookupOp &op,
                        StringAttr stringAttr) {
  printer.printKeywordOrString(llvm::encodeBase64(stringAttr.getValue()));
}

/// Prints a !mgp.cpu.buffer type, but represented in 'raw form'.
static void printRawSizeAsBufferType(AsmPrinter &p, const CASLookupOp &op,
                                     IntegerAttr rawSize) {
  printRawSizeAsBufferType(op->getContext(), p, rawSize);
}

/// Infers the buffer type result from the size argument.
LogicalResult CASLookupOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto rawSizeAttr = dyn_cast_if_present<IntegerAttr>(
      attributes.get(getCRawSizeAttrName(name)));
  assert(rawSizeAttr && "mismatched operand types");
  inferredReturnTypes.emplace_back(CpuBufferType::get(context, rawSizeAttr));
  return success();
}

/// Verifies optional alignment is valid.
LogicalResult CASLookupOp::verify() {
  if (!isValidRawAlign(getDRawAlign()))
    emitOpError() << "the alignment must be a power of two (have "
                  << getDRawAlign() << ").";
  return success();
}

//===----------------------------------------------------------------------===//
// CASStoreOp
//===----------------------------------------------------------------------===//

/// Print string in base64 encoding.
static void printBase64(AsmPrinter &printer, const CASStoreOp &op,
                        StringAttr stringAttr) {
  printer.printKeywordOrString(llvm::encodeBase64(stringAttr.getValue()));
}

//===----------------------------------------------------------------------===//
// CpuKernelExecuteOp
//===----------------------------------------------------------------------===//

static ParseResult
parseBufferSignature(OpAsmParser &p,
                     SmallVectorImpl<OpAsmParser::UnresolvedOperand> &buffers,
                     SmallVectorImpl<Type> &bufferTypes,
                     InOutSignatureAttr &inOutSignatureAttr) {
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics;
  auto parseOperandFn = [&]() -> ParseResult {
    llvm::SMLoc loc;

    // Parse the in/out/mut keyword.
    StringRef inOutMut;
    if (p.getCurrentLocation(&loc) || p.parseKeyword(&inOutMut))
      return failure();
    if (inOutMut == "in") {
      semantics.emplace_back(InOutSignatureAttr::kIn);
    } else if (inOutMut == "out") {
      semantics.emplace_back(InOutSignatureAttr::kOut);
    } else if (inOutMut == "mut") {
      semantics.emplace_back(InOutSignatureAttr::kMut);
    } else {
      return p.emitError(loc) << "expecting 'in', 'out' or 'mut' keyword";
    }

    // Parse the operand proper.
    OpAsmParser::UnresolvedOperand &unresolvedOperand = buffers.emplace_back();
    if (p.parseOperand(unresolvedOperand))
      return failure();

    // Parse the type annotation.
    Type &bufferType = bufferTypes.emplace_back();
    if (p.parseColon() || p.parseType(bufferType))
      return failure();

    return success();
  };

  if (p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseOperandFn,
                                "in buffer operand list"))
    return failure();

  inOutSignatureAttr = InOutSignatureAttr::get(p.getContext(), semantics);

  return success();
}

static void printBufferSignature(OpAsmPrinter &p, const CpuKernelExecuteOp &op,
                                 ValueRange buffers, TypeRange bufferTypes,
                                 InOutSignatureAttr inOutSignatureAttr) {
  size_t arity = inOutSignatureAttr.size();
  assert(buffers.size() == arity);
  assert(bufferTypes.size() == arity);
  p << "(";
  for (size_t i = 0; i < arity; ++i) {
    if (i > 0)
      p << ", ";
    switch (inOutSignatureAttr[i]) {
    case InOutSignatureAttr::kNone:
      llvm::llvm_unreachable_internal(
          "unexpected kNone buffer semantics in signature");
      break;
    case InOutSignatureAttr::kIn:
      p << "in ";
      break;
    case InOutSignatureAttr::kOut:
      p << "out ";
      break;
    case InOutSignatureAttr::kMut:
      p << "mut ";
      break;
    }
    p << buffers[i];
    p << " : ";
    p << bufferTypes[i];
  }
  p << ")";
}

LogicalResult CpuKernelExecuteOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  uint64_t numOutDimensions = 0;
  if (auto attr = attributes.get(getENumOutDimensionsAttrName(name))) {
    auto dNumOutDimensionsAttr = dyn_cast<IntegerAttr>(attr);
    assert(dNumOutDimensionsAttr && "mismatched operand types");
    numOutDimensions = dNumOutDimensionsAttr.getUInt();
  }
  inferredReturnTypes.emplace_back(MEF::ChainType::get(context));
  while (numOutDimensions-- > 0)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

LogicalResult CpuKernelExecuteOp::verify() {
  if (getCInOutSignature().size() != getBuffers().size())
    return emitOpError() << "expected in/out signature attribute to have arity "
                         << getBuffers().size() << " but given "
                         << getCInOutSignature().size() << ".";
  if (getENumOutDimensions() != getOutDimensions().size())
    return emitOpError()
           << "expected number of output dimensions attribute to be "
           << getOutDimensions().size() << " but given "
           << getENumOutDimensions() << ".";
  return success();
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

/// Returns the implicit signature of the 'init' entry point.
static FunctionType implicitInitSignature(Builder &builder) {
  return builder.getFunctionType({}, {builder.getType<MEF::ChainType>(),
                                      builder.getType<MGP::ContextType>()});
}

/// Returns the implicit signature of the 'execute' entry point, before
/// any explicit inputs and results have been included.
static FunctionType implicitExecuteSignature(Builder &builder) {
  return builder.getFunctionType(
      {builder.getType<MEF::ChainType>(), builder.getType<MGP::ContextType>()},
      {builder.getType<MEF::ChainType>()});
}

/// Returns true if requiredTypes is a prefix of actualTypes, in which case
/// the prefix is removed from actualTypes.
static bool checkAndRemovePrefix(ArrayRef<Type> requiredTypes,
                                 SmallVectorImpl<Type> &actualTypes) {
  auto [mismatchRequired, mismatchActual] = std::mismatch(
      requiredTypes.begin(), requiredTypes.end(), actualTypes.begin());
  if (mismatchRequired != requiredTypes.end())
    return false;
  actualTypes.erase(actualTypes.begin(),
                    actualTypes.begin() + requiredTypes.size());
  return true;
}

/// Parse the frameworkName attribute as a keyword.
static ParseResult parseClassName(OpAsmParser &p, StringAttr &frameworkName) {
  StringRef keyword;
  if (p.parseKeyword(&keyword))
    return failure();
  frameworkName = StringAttr::get(p.getContext(), keyword);
  return success();
}

/// Prints the frameworkName attribute as a keyword.
static void printClassName(OpAsmPrinter &p, const ModelOp &op,
                           StringAttr frameworkName) {
  p << frameworkName.getValue();
}

/// Parses the 'execute' body, and extracts from it the implicit signature.
static ParseResult parseExecuteRegion(OpAsmParser &p,
                                      TypeAttr &inputOutputTypes,
                                      Region &executeRegion) {
  llvm::SMLoc argLoc;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> outputTypes;
  if (p.getCurrentLocation(&argLoc) ||
      p.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                          /*allowType=*/true) ||
      p.parseArrowTypeList(outputTypes) ||
      p.parseRegion(executeRegion, args, /*enableNameShadowing=*/true))
    return failure();

  // Check the execute region's actual signature, and save its implicit
  // signature into inputOutputTypes.
  FunctionType implicitExecSig = implicitExecuteSignature(p.getBuilder());
  SmallVector<Type> inputTypes(executeRegion.getArgumentTypes());
  if (!checkAndRemovePrefix(implicitExecSig.getInputs(), inputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region arguments to begin with types "
           << implicitExecSig.getInputs() << ".";
  if (!checkAndRemovePrefix(implicitExecSig.getResults(), outputTypes))
    return p.emitError(argLoc)
           << "expected the 'execute' region outputs to begin with types "
           << implicitExecSig.getResults() << ".";

  inputOutputTypes =
      TypeAttr::get(p.getBuilder().getFunctionType(inputTypes, outputTypes));
  return success();
}

/// Prints the 'execute' body. The inputOutputTypes are ignored.
static void printExecuteRegion(OpAsmPrinter &p, const ModelOp &op,
                               TypeAttr inputOutputTypes,
                               Region &executeRegion) {
  Block &executeBlock = executeRegion.front();
  p << "(";
  llvm::interleaveComma(executeBlock.getArguments(), p.getStream(),
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ")";
  p.printArrowTypeList(executeBlock.getTerminator()->getOperandTypes());
  p << " ";
  p.printRegion(executeRegion, /*printEntryBlockArgs=*/false);
}

void ModelOp::build(OpBuilder &builder, OperationState &state,
                    StringAttr sym_name, StringRef frameworkName,
                    ArrayRef<Type> inputTypes, ArrayRef<Type> outputTypes,
                    ArrayRef<Location> inputArgLocs,
                    ArrayRef<NamedAttribute> extraAttrs) {
  DictionaryAttr extraAttrsDict =
      extraAttrs.empty() ? nullptr
                         : DictionaryAttr::get(state.getContext(), extraAttrs);
  build(builder, state, StringAttr::get(state.getContext(), frameworkName),
        TypeAttr::get(builder.getFunctionType(inputTypes, outputTypes)),
        sym_name, extraAttrsDict);

  // The 'init' entry point has no arguments.
  assert(implicitInitSignature(builder).getInputs().empty());
  Region *initRegion = state.regions[0].get();
  initRegion->emplaceBlock();

  // The 'execute' entry point has both implicit and explicit arguments.
  Region *executeRegion = state.regions[1].get();
  executeRegion->emplaceBlock();
  for (auto type : implicitExecuteSignature(builder).getInputs())
    executeRegion->addArgument(type, state.location);
  for (auto [type, loc] : llvm::zip(inputTypes, inputArgLocs))
    executeRegion->addArgument(type, loc);
}

/// Verifies the init and execute regions of the model.
LogicalResult ModelOp::verifyRegions() {
  OpBuilder builder(getContext());

  // The type of the init region is
  //  () -> (!mef.chain, !mgp.context)
  Block *initBlock = getInitBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(initBlock->getTerminator())) {
    FunctionType implicitInitSig = implicitInitSignature(builder);
    assert(implicitInitSig.getInputs().empty());
    if (!initBlock->getArgumentTypes().empty())
      return emitOpError()
             << "expected no arguments to the 'init' body, but got "
             << initBlock->getArgumentTypes() << ".";

    if (failed(checkOperandTypes(output, implicitInitSig.getResults())))
      return failure();
  }

  // The type of the execute region is:
  //   (!mef.chain, !mgp.context, arg_1, ... arg_n) ->
  //   (!mef.chain, res_1, ... res_m)
  // where the ModelOp's inputOutputTypes are:
  //   (arg_1, ... arg_n) -> (res_1, ... res_m)
  Block *execBlock = getExecuteBlock();
  if (auto output = dyn_cast<MEF::OutputOp>(execBlock->getTerminator())) {
    FunctionType implicitExecSig = implicitExecuteSignature(builder);
    ArrayRef<Type> modelArgTypes = getInputOutputTypes().getInputs();

    SmallVector<Type> expectedArgTypes(implicitExecSig.getInputs());
    expectedArgTypes.append(modelArgTypes.begin(), modelArgTypes.end());
    if (execBlock->getArgumentTypes() != expectedArgTypes)
      return emitOpError() << "expected the arguments of the "
                              "'execute' body to be "
                           << expectedArgTypes << ", but got "
                           << execBlock->getArgumentTypes() << ".";

    ArrayRef<Type> modelResultTypes = getInputOutputTypes().getResults();
    SmallVector<Type> expectedResultTypes(implicitExecSig.getResults());
    expectedResultTypes.append(modelResultTypes.begin(),
                               modelResultTypes.end());

    if (failed(checkOperandTypes(output, expectedResultTypes)))
      return failure();
  }
  return success();
}

/// Verifies the input/output type signature.
LogicalResult ModelOp::verify() {
  // Ensure that ModelOp's arguments and results are RankedTensorTypes.
  // TODO(#4753): We don't want to require args to be tensors, but don't
  // yet have a protocol for tying the M::Model subclass input and output
  // handling with the types declared here.
  FunctionType modelType = getInputOutputTypes();
  for (size_t i = 0, e = modelType.getNumInputs(); i < e; ++i) {
    Type type = modelType.getInput(i);
    if (!type.isa<ShapedType>())
      return emitOpError()
             << "expected the 'execute' entry point to take arguments "
                "of ShapedType, but argument "
             << i << " is of type " << type << ".";
  }
  for (size_t i = 0, e = modelType.getNumResults(); i < e; ++i) {
    Type type = modelType.getResult(i);
    if (!type.isa<ShapedType>())
      return emitOpError()
             << "expected the 'execute' entry point to return results "
                "of ShapedType, but result "
             << i << " is of type " << type << ".";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MGPDialect op support
//===----------------------------------------------------------------------===//

void MGPDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "GenericML/MGPDialect/MGP.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GenericML/MGPDialect/MGP.cpp.inc"
