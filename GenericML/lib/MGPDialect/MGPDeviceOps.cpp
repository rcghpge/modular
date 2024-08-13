//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/Support/Diagnostic.h"
#include "Support/AssertStream.h"
#include "Support/Compiler/ParserUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/ErrorOr.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Base64.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Assigns to device from attribute with name in attrs. Assigns <"cpu", 0>
/// if no such attribute.
static LogicalResult getDeviceAttrOrDefault(Location location,
                                            DictionaryAttr attributes,
                                            StringAttr name,
                                            DeviceRefAttr &device) {
  if (!attributes.contains(name)) {
    device = DeviceRefAttr::getDefault(attributes.getContext());
  } else {
    device = dyn_cast<DeviceRefAttr>(attributes.get(name));
    if (!device) {
      return mlir::emitError(location,
                             "expected an #M.device_ref for attribute '")
             << name << "'";
    }
  }
  return success();
}

/// Assigns to optSize from attribute with name in attrs.
static LogicalResult getSizeAttr(Location location, DictionaryAttr attributes,
                                 StringAttr name,
                                 std::optional<uint64_t> &optSize) {
  auto sizeAttr = dyn_cast_if_present<IntegerAttr>(attributes.get(name));
  if (!sizeAttr) {
    return mlir::emitError(location, "expected a uint64_t for attribute '")
           << name << "'";
  }
  optSize = std::optional<uint64_t>(sizeAttr.getUInt());
  return success();
}

/// Assigns to optSize from attribute with name in attrs.
static LogicalResult getRawSizeAttr(Location location,
                                    DictionaryAttr attributes, StringAttr name,
                                    std::optional<uint64_t> &optSize) {
  auto rawSizeAttr = dyn_cast_if_present<IntegerAttr>(attributes.get(name));
  if (!rawSizeAttr) {
    return mlir::emitError(location, "expected a uint64_t for attribute '")
           << name << "'";
  }
  optSize = rawSizeToOptSize(rawSizeAttr.getUInt());
  return success();
}

/// Appends to optSizes from array attribute with name in attrs.
static LogicalResult
getRawSizeArrayAttr(Location location, DictionaryAttr attributes,
                    StringAttr name,
                    SmallVectorImpl<std::optional<uint64_t>> &optSizes) {
  auto arrayAttr =
      dyn_cast_if_present<IntArrayElementsAttr>(attributes.get(name));
  if (!arrayAttr) {
    return mlir::emitError(location, "expected an array for attribute '")
           << name << "'";
  }
  for (auto elem : arrayAttr.getValues())
    optSizes.emplace_back(rawSizeToOptSize(elem.getZExtValue()));
  return success();
}

/// Assigns to num from attribute with name in attrs. Assigns 0 if no such
/// attribute.
static LogicalResult getNumAttrOrDefault(Location location,
                                         DictionaryAttr attributes,
                                         StringAttr name, uint64_t &num) {
  if (!attributes.contains(name)) {
    num = 0;
  } else {
    auto numAttr = dyn_cast<IntegerAttr>(attributes.get(name));
    if (!numAttr) {
      return mlir::emitError(location, "expected a uint64_t for attribute '")
             << name << "'";
    }
    num = numAttr.getUInt();
  }
  return success();
}

/// Assigns to elements from attribute with name in attrs.
static LogicalResult getElementsAttr(Location location,
                                     DictionaryAttr attributes, StringAttr name,
                                     ElementsAttr &elements) {
  elements = dyn_cast_if_present<ElementsAttr>(attributes.get(name));
  if (!elements) {
    return mlir::emitError(location, "expecting ElementsAttr for attribute '")
           << name << "'";
  }
  return success();
}

/// Assigns to flag from attribute with name in attrs.
static LogicalResult getBoolAttr(Location location, DictionaryAttr attributes,
                                 StringAttr name, bool &flag) {
  auto boolAttr = dyn_cast_if_present<BoolAttr>(attributes.get(name));
  if (!boolAttr) {
    return mlir::emitError(location, "expected a bool for attribute '")
           << name << "'";
  }
  flag = boolAttr.getValue();
  return success();
}

/// Returns a description of optSize.
static std::string optSizeToString(std::optional<uint64_t> optSize) {
  if (optSize)
    return std::to_string(*optSize);
  else
    return "unsized";
}

/// Returns the raw alignment for an elements attribute.
static uint64_t getRawAlign(ElementsAttr value) {
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (errOrOptAlign.isError())
    return 0;

  // If error we'll report it when the result is validated.
  return optSizeToRawSize(*errOrOptAlign);
}

//===----------------------------------------------------------------------===//
// RuntimeCreateOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DeviceContextCreateOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextCreateOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// DeviceContextExtractOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextExtractOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// DeviceContextProfileStartOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextProfileStartOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// DeviceContextProfileEndOp
//===----------------------------------------------------------------------===//

LogicalResult DeviceContextProfileEndOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleLoadOp::verify() {
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

void BufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             Value inChain, Value theContext,
                             uint64_t aRuntimeSlot, ElementsAttr bValue,
                             DeviceRefAttr dDevice, bool eIsTrivial) {
  return build(odsBuilder, odsState, inChain, theContext, aRuntimeSlot, bValue,
               getRawAlign(bValue), dDevice, eIsTrivial);
}

/// Verifies the attribute and result types are consistent, and the attribute
/// is supported.
LogicalResult BufferConstantOp::verify() {
  // We can only construct from attributes which are dense arrays (ie rank-1)
  // of uint8_ts.
  ElementsAttr value = getBValue();
  ShapedType shapedType = cast<ShapedType>(value.getType());
  if (value.size() != shapedType.getNumElements())
    return emitOpError("does not support splat attributes.");
  if (!shapedType.hasRank() || shapedType.getShape().size() != 1)
    return emitOpError("only supports rank-1 attributes.");
  Type elemType = value.getElementType();
  if (!elemType.isUnsignedInteger(8))
    return emitOpError("only supports attributes of uint8_t elements")
           << " (have " << elemType << ").";

  // The resulting constant's type must be sized, and the size must match
  // the attribute's size.
  uint64_t actualSize = static_cast<uint64_t>(value.getNumElements());
  auto expectedBufferType =
      DeviceBufferType::get(getDDevice(), std::optional<uint64_t>(actualSize));
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }

  // Only BufferAttr and DenseResourceElementsAttr can convey an align
  // constraint. Check they are valid.
  ErrorOr<std::optional<uint64_t>> errOrOptAlign = getImpliedAlignment(value);
  if (auto *error = errOrOptAlign.getError())
    return emitOpError(error);

  if (!isValidOptAlign(*errOrOptAlign))
    return emitOpError("invalid alignment ") << **errOrOptAlign << ".";

  // The implied alignment must match the actual attribute alignment.
  if (getOptAlign().has_value() != errOrOptAlign->has_value())
    return emitOpError("expected alignment operand if and only if value "
                       "operand has alignment.");
  if (getOptAlign().has_value() && *getOptAlign() != **errOrOptAlign)
    return emitOpError(
               "expected operand alignment to match attribute's alignment (")
           << *getOptAlign() << " vs " << **errOrOptAlign << ").";

  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

LogicalResult BufferConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  ElementsAttr elementsAttr;
  if (failed(getElementsAttr(location.value_or(operands[0].getLoc()),
                             attributes, getBValueAttrName(name),
                             elementsAttr)))
    return failure();
  std::optional<uint64_t> optSize = elementsAttr.size();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getDDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, device, optSize));
  return success();
}

std::optional<uint64_t> BufferConstantOp::getOptAlign() {
  return rawSizeToOptSize(getCRawAlign());
}

//===----------------------------------------------------------------------===//
// BufferConstantLoadOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConstantLoadOp::verify() {
  if (!isValidPlainSize(getDSize()))
    return emitOpError("invalid static size ") << getDSize() << ".";
  if (!getOutBuffer().getType().getOptSize())
    return emitOpError("expected output buffer to be statically sized.");
  auto expectedBufferType =
      DeviceBufferType::get(getEDevice(), std::optional<uint64_t>(getDSize()));
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

LogicalResult BufferConstantLoadOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  std::optional<uint64_t> optSize;
  if (failed(getSizeAttr(location.value_or(operands[0].getLoc()), attributes,
                         getDSizeAttrName(name), optSize)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getEDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferAllocStaticOp
//===----------------------------------------------------------------------===//

LogicalResult BufferAllocStaticOp::verify() {
  if (!isValidPlainSize(getBSize()))
    return emitOpError("invalid static size ") << getBSize() << ".";
  auto expectedBufferType =
      DeviceBufferType::get(getDDevice(), std::optional<uint64_t>(getBSize()));
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  if (!isValidRawAlign(getCRawAlign()))
    return emitOpError("invalid alignment ") << getCRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

LogicalResult BufferAllocStaticOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  std::optional<uint64_t> optSize;
  if (failed(getSizeAttr(location.value_or(operands[0].getLoc()), attributes,
                         getBSizeAttrName(name), optSize)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getDDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

void BufferAllocStaticOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

uint64_t BufferAllocStaticOp::getRawAlign() { return getCRawAlign(); }

uint64_t BufferAllocStaticOp::getSize() { return getBSize(); }

//===----------------------------------------------------------------------===//
// BufferAllocDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult BufferAllocDynamicOp::verify() {
  auto expectedBufferType =
      DeviceBufferType::get(getCDevice(), std::optional<uint64_t>());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  if (!isValidRawAlign(getBRawAlign()))
    return emitOpError("invalid alignment ") << getBRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

LogicalResult BufferAllocDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getCDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(device, std::optional<uint64_t>()));
  return success();
}

void BufferAllocDynamicOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

uint64_t BufferAllocDynamicOp::getRawAlign() { return getBRawAlign(); }

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

LogicalResult BufferCacheOp::verify() {
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

LogicalResult BufferGetCachedOp::verify() {
  std::optional<uint64_t> optSize = rawSizeToOptSize(getCRawSize());
  auto expectedBufferType = DeviceBufferType::get(getDDevice(), optSize);
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("output buffer type does not match expected (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

LogicalResult BufferGetCachedOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  std::optional<uint64_t> optSize;
  if (failed(getRawSizeAttr(location.value_or(operands[0].getLoc()), attributes,
                            getCRawSizeAttrName(name), optSize)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getDDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

//===----------------------------------------------------------------------===//
// BufferHostToDeviceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferHostToDeviceOp::verify() {
  if (getHostBuffer().getType().getDevice() != getCHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getCHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getDOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getDOtherDevice() << ".";
  if (getHostBuffer().getType().getRawSize() !=
      getOtherBuffer().getType().getRawSize())
    return emitOpError("host and other buffer sizes do not match (")
           << optSizeToString(getHostBuffer().getType().getOptSize()) << " vs "
           << optSizeToString(getOtherBuffer().getType().getOptSize()) << ").";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferHostToDeviceOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getOtherBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToHostOp
//===----------------------------------------------------------------------===//

LogicalResult BufferDeviceToHostOp::verify() {
  if (getHostBuffer().getType().getDevice() != getDHostDevice())
    return emitOpError("expected host buffer to be on 'host' device ")
           << getDHostDevice() << ".";
  if (getOtherBuffer().getType().getDevice() != getCOtherDevice())
    return emitOpError("expected other buffer to be on 'other' device ")
           << getCOtherDevice() << ".";
  if (getHostBuffer().getType().getRawSize() !=
      getOtherBuffer().getType().getRawSize())
    return emitOpError("host and other buffer sizes do not match (")
           << optSizeToString(getHostBuffer().getType().getOptSize()) << " vs "
           << optSizeToString(getOtherBuffer().getType().getOptSize()) << ").";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToHostOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getHostBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToDeviceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferDeviceToDeviceOp::verify() {
  if (getSrcBuffer().getType().getRawSize() !=
      getDstBuffer().getType().getRawSize())
    return emitOpError("src and dst buffer sizes do not match (")
           << optSizeToString(getSrcBuffer().getType().getOptSize()) << " vs "
           << optSizeToString(getDstBuffer().getType().getOptSize()) << ").";
  if (getCSrcDevice().getLabel() != getDDstDevice().getLabel()) {
    return emitOpError("expecting source and destination devices to have the "
                       "same device label.");
  }
  if (getSrcBuffer().getType().getDevice() != getCSrcDevice() ||
      getDstBuffer().getType().getDevice() != getDDstDevice()) {
    return emitOpError(
               "expecting input chain and source buffer to be on device '")
           << getCSrcDevice()
           << "', and output chain and destination buffer to be on device '"
           << getDDstDevice() << "'.";
  }
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToDeviceOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getDstBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferUnsizeOp
//===----------------------------------------------------------------------===//

LogicalResult BufferUnsizeOp::verify() {
  if (!getInBuffer().getType().getOptSize())
    return emitOpError("expected input buffer to be statically sized.");
  if (getOutBuffer().getType().getOptSize())
    return emitOpError("expected output buffer to be dynamically sized.");
  return verifyConsistentDevices(getOperation(),
                                 {getInBuffer().getType().getDevice()});
}

LogicalResult BufferUnsizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  if (operands.size() != 1) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected one operand.");
  }
  auto bufferType = dyn_cast<DeviceBufferType>(operands[0].getType());
  if (!bufferType) {
    return mlir::emitError(operands[0].getLoc(),
                           "expected !mgp.buffer operand.");
  }
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(bufferType.getDevice(), std::optional<uint64_t>()));
  return success();
}

bool BufferUnsizeOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return true;
}

//===----------------------------------------------------------------------===//
// BufferResizeOp
//===----------------------------------------------------------------------===//

LogicalResult BufferResizeOp::verify() {
  if (getInBuffer().getType().getOptSize())
    return emitOpError("expected input buffer to be dynamically sized.");
  if (!getOutBuffer().getType().getOptSize())
    return emitOpError("expected output buffer to be statically sized.");
  auto expectedBufferType =
      DeviceBufferType::get(getBDevice(), std::optional<uint64_t>(getASize()));
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected output buffer type does not match actual "
                       "output buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }

  return verifyConsistentDevices(getOperation(),
                                 {getInBuffer().getType().getDevice()});
}

LogicalResult BufferResizeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  std::optional<uint64_t> optSize;
  if (failed(getSizeAttr(location.value_or(operands[0].getLoc()), attributes,
                         getASizeAttrName(name), optSize)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getBDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

bool BufferResizeOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return true;
}

//===----------------------------------------------------------------------===//
// BufferConcatOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConcatOp::verify() {
  if (failed(verifyConsistentDevices(getOperation(), {getBDevice()})))
    return failure();

  // If possible, verify sum of input buffer sizes equals the output
  // buffer size.
  if (!getOutBuffer().getType().getOptSize())
    // Output does not have a statically known size.
    return success();

  uint64_t inTotalSize = 0;
  for (const auto &inBuffer : getInBuffers()) {
    auto inBufferType = dyn_cast<DeviceBufferType>(inBuffer.getType());
    std::optional<uint64_t> optSize = inBufferType.getOptSize();
    if (!optSize)
      // Not all input buffers have a statically known size.
      return success();
    inTotalSize += *optSize;
  }
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getBDevice(), std::optional<uint64_t>(inTotalSize));
  if (getOutBuffer().getType() != expectedBufferType) {
    return emitOpError("expected output buffer type does not match actual "
                       "output buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferConcatOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getOutBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferSplitBorrowOp
//===----------------------------------------------------------------------===//

void BufferSplitBorrowOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<MEF::ChainType> inChain, TypedValue<ContextType> theContext,
    TypedValue<DeviceBufferType> inBuffer,
    ArrayRef<TypedValue<IndexType>> runtimeSizes, uint64_t aRuntimeSlot,
    llvm::ArrayRef<std::optional<uint64_t>> bOutBufferOptSizes,
    M::DeviceRefAttr cDevice) {
  SmallVector<Type> resultBufferTypes;
  SmallVector<uint64_t> outBufferRawSizes;
  for (auto optSize : bOutBufferOptSizes) {
    resultBufferTypes.emplace_back(DeviceBufferType::get(cDevice, optSize));
    outBufferRawSizes.emplace_back(optSizeToRawSize(optSize));
  }
  SmallVector<Value> runtimeSizesAnyValue;
  llvm::append_range(runtimeSizesAnyValue, runtimeSizes);
  return build(odsBuilder, odsState, resultBufferTypes, inChain, theContext,
               inBuffer, runtimeSizesAnyValue, aRuntimeSlot, outBufferRawSizes,
               cDevice);
}

LogicalResult BufferSplitBorrowOp::verify() {
  size_t numDynamic = llvm::count_if(getOutBuffers(), [](const Value &v) {
    return !dyn_cast<DeviceBufferType>(v.getType()).getOptSize();
  });
  if (getRuntimeSizes().size() != numDynamic) {
    return emitOpError("expected a runtime size operand for every output "
                       "buffer with a dynamic size (given ")
           << getRuntimeSizes().size() << " sizes, but need " << numDynamic
           << " sizes).";
  }

  if (failed(verifyConsistentDevices(getOperation(), {getCDevice()})))
    return failure();

  if (!getInBuffer().getType().getOptSize())
    // Nothing more to check if dynamically sized input buffer.
    return success();

  uint64_t outTotalSize = 0;
  for (const auto &outBuffer : getOutBuffers()) {
    auto outBufferType = dyn_cast<DeviceBufferType>(outBuffer.getType());
    std::optional<uint64_t> optSize = outBufferType.getOptSize();
    if (!optSize)
      // Nothing more to check if at least one dynamically sized output
      // buffer.
      return success();
    outTotalSize += *optSize;
  }
  auto expectedInBufferType = DeviceBufferType::get(
      getCDevice(), std::optional<uint64_t>(outTotalSize));
  if (getInBuffer().getType() != expectedInBufferType) {
    return emitOpError("expected input buffer type does not match "
                       "actual input buffer type (")
           << expectedInBufferType << " vs " << getInBuffer().getType() << ").";
  }
  return success();
}

LogicalResult BufferSplitBorrowOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  SmallVector<std::optional<uint64_t>> optSizes;
  if (failed(getRawSizeArrayAttr(
          location.value_or(operands[0].getLoc()), attributes,
          getBOutBufferRawSizesAttrName(name), optSizes)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getCDeviceAttrName(name),
                                    device)))
    return failure();
  for (auto optSize : optSizes)
    inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

bool BufferSplitBorrowOp::extendsLifetime(size_t operandIndex,
                                          size_t resultIndex) {
  return operandIndex == getInBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// BufferSectionBorrowStaticOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSectionBorrowStaticOp::verify() {
  if (!isValidPlainSize(getCSize()))
    return emitOpError("invalid size ") << getCSize() << ".";
  if (!isValidPlainSize(getBOffset()))
    return emitOpError("invalid offset ") << getBOffset() << ".";

  if (!getInBuffer().getType().getOptSize())
    return emitOpError("expected input buffer to be statically sized.");
  uint64_t inSize = *getInBuffer().getType().getOptSize();
  if (getBOffset() > inSize) {
    return emitOpError("requesting section offset ")
           << getBOffset() << " which is beyond end of input buffer of size "
           << inSize << ".";
  }
  if (getBOffset() + getCSize() > inSize) {
    return emitOpError("requesting section end ")
           << getBOffset() + getCSize()
           << " which is beyond end of input buffer of size " << inSize << ".";
  }

  auto expectedOutBufferType = DeviceBufferType::get(getDDevice(), getCSize());
  if (expectedOutBufferType != getOutBuffer().getType()) {
    return emitOpError("expected output buffer type does not match "
                       "actual output buffer type (")
           << expectedOutBufferType << " vs " << getOutBuffer().getType()
           << ").";
  }

  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

LogicalResult BufferSectionBorrowStaticOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  std::optional<uint64_t> optSize;
  if (failed(getSizeAttr(location.value_or(operands[0].getLoc()), attributes,
                         getCSizeAttrName(name), optSize)))
    return failure();
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getDDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(device, optSize));
  return success();
}

bool BufferSectionBorrowStaticOp::extendsLifetime(size_t operandIndex,
                                                  size_t resultIndex) {
  return operandIndex == getInBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// BufferSectionBorrowDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSectionBorrowDynamicOp::verify() {
  if (!isValidPlainSize(getBUnscaledOffset()))
    return emitOpError("invalid offset ") << getBUnscaledOffset() << ".";
  if (!isValidPlainSize(getCUnscaledSize()))
    return emitOpError("invalid size ") << getCUnscaledSize() << ".";
  if (getInBuffer().getType().getOptSize())
    return emitOpError("expected input buffer to be dynamically sized.");
  if (getOutBuffer().getType().getOptSize())
    return emitOpError("expected output buffer to be dynamically sized.");
  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

LogicalResult BufferSectionBorrowDynamicOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getDDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(device, std::optional<uint64_t>()));
  return success();
}

bool BufferSectionBorrowDynamicOp::extendsLifetime(size_t operandIndex,
                                                   size_t resultIndex) {
  return operandIndex == getInBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSliceOp::verify() {
  if (getOutBuffer().getType().getOptSize())
    return emitOpError("expected output buffer to be dynamically sized.");
  return success();
}

LogicalResult BufferSliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto inBufferTy = dyn_cast<DeviceBufferType>(operands[0].getType());
  if (!inBufferTy)
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(inBufferTy.getDevice(), std::optional<uint64_t>()));
  return success();
}

bool BufferSliceOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getInBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// BufferToBoolOp
//===----------------------------------------------------------------------===//

LogicalResult BufferToBoolOp::verify() {
  auto expectedBufferType =
      DeviceBufferType::get(getBDevice(), std::optional<uint64_t>(1));
  if (expectedBufferType != getBuffer().getType()) {
    return emitOpError("expected operand buffer type does not match actual "
                       "operand buffer type (")
           << expectedBufferType << " vs " << getBuffer().getType() << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//

LogicalResult BufferSetWithIndexOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

InOutSignatureAttr::InOutSemantics
BufferSetWithIndexOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  return index == getBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferPlanOp
//===----------------------------------------------------------------------===//
LogicalResult BufferPlanOp::verify() {

  if (!llvm::all_equal({getRuntimeSizes().size(), getOffsets().size(),
                        getAAlignments().size(), getBMinPre().size(),
                        getCMinPost().size(), getDMaxPre().size(),
                        getEMaxPost().size()}))
    return emitOpError("expected number of sizes, output offsets, alignments "
                       "and lifetime info attributes to "
                       "be equal.");
  return success();
}

LogicalResult BufferPlanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  // All outputs are index
  inferredReturnTypes.reserve(operands.size() + 1);
  for (size_t i = 0; i < operands.size() + 1; ++i)
    inferredReturnTypes.emplace_back(mlir::IndexType::get(context));

  return success();
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::verify() {
  auto expectedOutTensorType =
      DeviceTensorType::get(getBDevice(), getTensorSpec().getType());
  if (expectedOutTensorType != getOutTensor().getType()) {
    return emitOpError("expected output tensor type does not match "
                       "actual output tensor type (")
           << expectedOutTensorType << " vs " << getOutTensor().getType()
           << ").";
  }
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getOutTensor().getType());
  if (getBuffer().getType() != expectedBufferType) {
    return emitOpError(
               "expected buffer type to match type implied by tensor spec (")
           << getBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

LogicalResult TensorCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  if (operands.size() != 4) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected four operands.");
  }
  auto tensorSpecType = dyn_cast<TensorSpecType>(operands[3].getType());
  if (!tensorSpecType) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected tensor spec operand.");
  }
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getBDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceTensorType::get(device, tensorSpecType));
  return success();
}

bool TensorCreateOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getBufferMutable().getOperandNumber();
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

LogicalResult TensorExtractBufferOp::verify() {
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getTensor().getType());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected the result buffer to match the type "
                       "implied by the tensor type (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

LogicalResult TensorExtractBufferOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  if (operands.size() != 2) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected two operands.");
  }
  auto tensorType = dyn_cast<DeviceTensorType>(operands[1].getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected tensor operand.");
  }
  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getBDeviceAttrName(name),
                                    device)))
    return failure();
  inferredReturnTypes.emplace_back(DeviceBufferType::get(tensorType));
  return success();
}

bool TensorExtractBufferOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getTensorMutable().getOperandNumber();
}

OpFoldResult TensorExtractBufferOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getBuffer();

  return {};
}

//===----------------------------------------------------------------------===//
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//

LogicalResult TensorExtractTensorSpecOp::verify() {
  TensorSpecType expectedSpecType =
      getTensor().getType().getTensorSpec(getCIsShapeOnly());
  if (expectedSpecType != getOutSpec().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutSpec().getType() << " vs " << expectedSpecType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

LogicalResult TensorExtractTensorSpecOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  if (operands.size() != 2) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected two operands.");
  }
  auto tensorType = dyn_cast<DeviceTensorType>(operands[1].getType());
  if (!tensorType) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected tensor operand.");
  }
  bool isShapeOnly = false;
  if (failed(getBoolAttr(location.value_or(operands[0].getLoc()), attributes,
                         getCIsShapeOnlyAttrName(name), isShapeOnly)))
    return failure();
  inferredReturnTypes.emplace_back(tensorType.getTensorSpec(isShapeOnly));
  return success();
}

OpFoldResult TensorExtractTensorSpecOp::fold(FoldAdaptor adaptor) {
  Operation *tensorProducer = getTensor().getDefiningOp();
  if (auto create = dyn_cast_or_null<TensorCreateOp>(tensorProducer))
    return create.getTensorSpec();
  return {};
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

LogicalResult DebugTensorPrintOp::verify() {
  auto expectedBufferType =
      DeviceBufferType::get(getCDevice(), getSpec().getType());
  if (expectedBufferType != getBuffer().getType()) {
    return emitOpError("expected input buffer type does not match actual "
                       "input buffer type (")
           << expectedBufferType << " vs " << getBuffer().getType() << ").";
  }
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

//===----------------------------------------------------------------------===//
// KernelExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult KernelExecuteOp::verify() {
  if (getEInOutSignature().size() != getArgs().size())
    return emitOpError("kernel has ")
           << getArgs().size()
           << " arguments, but the in/out signature attribute has arity "
           << getEInOutSignature().size() << ".";

  for (auto [idx, arg] : llvm::enumerate(getArgs())) {
    if (!isa<DeviceBufferType>(arg.getType()) &&
        getEInOutSignature()[idx] != InOutSignatureAttr::kIn) {
      return emitOpError("argument ") << idx
                                      << " is not an !mgp.buffer type, so "
                                         "must be marked as read-only.";
    }
  }

  if (getINumOutDimensions() != getOutDimensions().size()) {
    return emitOpError()
           << "expected number of output dimensions attribute to be "
           << getOutDimensions().size() << ", but given "
           << getINumOutDimensions() << ".";
  }

  return verifyConsistentDevices(getOperation(),
                                 {getFHostDevice(), getGOtherDevice()});
}

LogicalResult KernelExecuteOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  DeviceRefAttr otherDevice;
  if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
                                    attributes, getGOtherDeviceAttrName(name),
                                    otherDevice)))
    return failure();

  uint64_t numOutDimensions = 0;
  if (failed(getNumAttrOrDefault(location.value_or(operands[0].getLoc()),
                                 attributes, getINumOutDimensionsAttrName(name),
                                 numOutDimensions)))
    return failure();

  inferredReturnTypes.emplace_back(MEF::ChainType::get(context));
  while (numOutDimensions-- > 0)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

InOutSignatureAttr::InOutSemantics
KernelExecuteOp::getInOutSemantics(size_t index) {
  assert(index < getNumOperands());
  size_t firstArgsIndex = getTheContextMutable().getOperandNumber() + 1;
  size_t endArgsIndex = firstArgsIndex + getArgs().size();
  if (index < firstArgsIndex || index >= endArgsIndex)
    return InOutSignatureAttr::kIn;
  return getEInOutSignature()[index - firstArgsIndex];
}
