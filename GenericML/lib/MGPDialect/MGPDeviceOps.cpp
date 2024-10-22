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
// BufferConstantExternalOp
//===----------------------------------------------------------------------===//

LogicalResult BufferConstantExternalOp::verify() {
  uint64_t align = getDAlign();
  if (!isValidOptAlign(align))
    return emitOpError("invalid alignment ") << align << ".";

  return success();
}

LogicalResult BufferConstantExternalOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Directly take the size and device of the result buffer from op attributes.
  uint64_t sizeBytes = cast<IntegerAttr>(attributes.get("cSize")).getUInt();

  DeviceRefAttr device;
  if (failed(getDeviceAttrOrDefault(
          location.value_or(operands[0].getLoc()), attributes,
          getEDeviceAttrName(/*name=*/{getOperationName(), context}), device)))
    return failure();
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, device, sizeBytes));
  return success();
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
// BufferAllocOp
//===----------------------------------------------------------------------===//

LogicalResult BufferAllocOp::verify() {
  if (getCDevice() != getOutBuffer().getType().getDevice()) {
    return emitOpError("expected result buffer device does not match actual "
                       "result buffer device (")
           << getCDevice() << " vs " << getOutBuffer().getType().getDevice()
           << ").";
  }
  if (!isValidRawAlign(getBRawAlign()))
    return emitOpError("invalid alignment ") << getBRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

// LogicalResult BufferAllocOp::inferReturnTypes(
//     MLIRContext *context, std::optional<Location> location, ValueRange
//     operands, DictionaryAttr attributes, mlir::OpaqueProperties properties,
//     mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
//   mlir::OperationName name(getOperationName(), context);
//   DeviceRefAttr device;
//   if (failed(getDeviceAttrOrDefault(location.value_or(operands[0].getLoc()),
//                                     attributes, getCDeviceAttrName(name),
//                                     device)))
//     return failure();
//   inferredReturnTypes.emplace_back(
//       DeviceBufferType::get(device, std::optional<uint64_t>()));
//   return success();
// }

void BufferAllocOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

LogicalResult BufferCacheOp::verify() {
  return verifyConsistentDevices(getOperation(), {getCDevice()});
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
  auto expectedOutTensorType = DeviceTensorType::get(
      getBuffer().getType().getDevice(), getTensorSpec().getType());
  if (expectedOutTensorType != getOutTensor().getType()) {
    return emitOpError("expected output tensor type does not match "
                       "actual output tensor type (")
           << expectedOutTensorType << " vs " << getOutTensor().getType()
           << ").";
  }
  return success();
}

LogicalResult TensorCreateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);
  auto tensorSpecType = dyn_cast<TensorSpecType>(operands[2].getType());
  if (!tensorSpecType) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected tensor spec operand.");
  }

  DeviceRefAttr device =
      cast<DeviceBufferType>(operands[1].getType()).getDevice();
  inferredReturnTypes.emplace_back(
      DeviceTensorType::get(device, tensorSpecType));
  return success();
}

bool TensorCreateOp::mayAlias(size_t operandIndex, size_t resultIndex) {
  return operandIndex == getBufferMutable().getOperandNumber();
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
