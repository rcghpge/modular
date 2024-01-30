//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/Support/Assert.h"
#include "GenericML/Support/Diagnostic.h"
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
#include "llvm/Support/Base64.h"

using namespace M;
using namespace MGP;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Verifies the types of all operands of op uses devices drawn from the
/// given devices.
static LogicalResult
verifyConsistentDevices(Operation *op,
                        std::initializer_list<DeviceRefAttr> expectedDevices) {
  llvm::SmallSetVector<DeviceRefAttr, 2> allDevices;
  for (auto device : expectedDevices)
    allDevices.insert(device);
  size_t expectedNumDevices = allDevices.size();
  std::string expectedStr;
  {
    llvm::raw_string_ostream os(expectedStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  for (OpOperand &opOperand : op->getOpOperands()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceInterface>(opOperand.get().getType()))
      allDevices.insert(hasDeviceType.getDevice());
  }
  for (OpResult opResult : op->getResults()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceInterface>(opResult.getType()))
      allDevices.insert(hasDeviceType.getDevice());
  }

  if (allDevices.size() == expectedNumDevices)
    return success();

  std::string actualStr;
  {
    llvm::raw_string_ostream os(actualStr);
    llvm::interleaveComma(allDevices, os,
                          [&os](DeviceRefAttr device) { os << device; });
  }

  return op->emitOpError("expected all operands, results, and attributes to "
                         "use the device(s) {")
         << expectedStr << "}, but they instead use the devices(s) {"
         << actualStr << "}.";
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

LogicalResult RuntimeCreateOp::verify() {
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleLoadOp::verify() {
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

//===----------------------------------------------------------------------===//
// ChainCreateHostOp
//===----------------------------------------------------------------------===//

LogicalResult ChainCreateHostOp::verify() {
  return verifyConsistentDevices(getOperation(), {getDevice()});
}

llvm::SmallVector<mlir::Value> ChainCreateHostOp::getInputChains() {
  return getGatingOperands();
}

void ChainCreateHostOp::setInputChains(::mlir::ValueRange gatingOperands) {
  getGatingOperandsMutable().assign(gatingOperands);
}

//===----------------------------------------------------------------------===//
// ChainCreateOtherOp
//===----------------------------------------------------------------------===//

LogicalResult ChainCreateOtherOp::verify() {
  if (getBHostDevice() == getCOtherDevice())
    return emitOpError("expected 'host' and 'other' devices to be different.");
  return verifyConsistentDevices(getOperation(),
                                 {getBHostDevice(), getCOtherDevice()});
}

llvm::SmallVector<mlir::Value> ChainCreateOtherOp::getInputChains() {
  return getGatingOperands();
}

void ChainCreateOtherOp::setInputChains(::mlir::ValueRange gatingOperands) {
  getGatingOperandsMutable().assign(gatingOperands);
}

//===----------------------------------------------------------------------===//
// ChainDeviceToHostOp
//===----------------------------------------------------------------------===//

LogicalResult ChainDeviceToHostOp::verify() {
  if (getInChain().getType().getDevice() != getCOtherDevice())
    return emitOpError("input chain must be on the 'other' device ")
           << getCOtherDevice() << ".";
  if (getOutChain().getType().getDevice() != getBHostDevice())
    return emitOpError("output chain must be on the 'host' device ")
           << getBHostDevice() << ".";
  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

void BufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             Value theContext, uint64_t aRuntimeSlot,
                             ElementsAttr bValue, DeviceRefAttr dDevice) {
  return build(odsBuilder, odsState, theContext, aRuntimeSlot, bValue,
               getRawAlign(bValue), dDevice);
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
  auto elementsAttr = dyn_cast_if_present<ElementsAttr>(
      attributes.get(getBValueAttrName(name)));
  if (!elementsAttr)
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expecting elements attribute.");
  std::optional<uint64_t> optSize = elementsAttr.size();
  auto device = dyn_cast_if_present<DeviceRefAttr>(
      attributes.get(getDDeviceAttrName(name)));
  if (!device) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected a device attribute.");
  }
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
  auto expectedBufferType = DeviceBufferType::get(getEDevice(), getDSize());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  return verifyConsistentDevices(getOperation(), {getEDevice()});
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

void BufferAllocDynamicOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// BufferReallocStaticOp
//===----------------------------------------------------------------------===//

LogicalResult BufferReallocStaticOp::verify() {
  if (!isValidPlainSize(getDSize()))
    return emitOpError("invalid size ") << getDSize() << ".";
  auto expectedBufferType =
      DeviceBufferType::get(getFDevice(), std::optional<uint64_t>(getDSize()));
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  if (!isValidRawAlign(getERawAlign()))
    return emitOpError("invalid alignment ") << getERawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getFDevice()});
}

void BufferReallocStaticOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// BufferReallocDynamicOp
//===----------------------------------------------------------------------===//

LogicalResult BufferReallocDynamicOp::verify() {
  auto expectedBufferType =
      DeviceBufferType::get(getEDevice(), std::optional<uint64_t>());
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("expected result buffer type does not match actual "
                       "result buffer type (")
           << expectedBufferType << " vs " << getOutBuffer().getType() << ").";
  }
  if (!isValidRawAlign(getDRawAlign()))
    return emitOpError("invalid alignment ") << getDRawAlign() << ".";
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

void BufferReallocDynamicOp::setInChain(Value chain) {
  getInChainMutable().assign(chain);
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

LogicalResult BufferCacheOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

LogicalResult BufferGetCachedOp::verify() {
  std::optional<uint64_t> optSize = rawSizeToOptSize(getBRawSize());
  auto expectedBufferType = DeviceBufferType::get(getCDevice(), optSize);
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("output buffer type does not match expected (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
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
           << getHostBuffer().getType().getRawSize().getUInt() << " vs "
           << getOtherBuffer().getType().getRawSize().getUInt() << ").";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferHostToDeviceOp::getInOutSemantics(size_t index) {
  GENERICML_ASSERT(index < getNumOperands());
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
           << getHostBuffer().getType().getRawSize().getUInt() << " vs "
           << getOtherBuffer().getType().getRawSize().getUInt() << ").";
  return success();
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToHostOp::getInOutSemantics(size_t index) {
  GENERICML_ASSERT(index < getNumOperands());
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
    return emitOpError("src and dst buffer sizes do not match ")
           << getSrcBuffer().getType().getRawSize().getUInt() << " vs "
           << getDstBuffer().getType().getRawSize().getUInt() << ").";
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

InOutSignatureAttr::InOutSemantics
BufferDeviceToDeviceOp::getInOutSemantics(size_t index) {
  GENERICML_ASSERT(index < getNumOperands());
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

//===----------------------------------------------------------------------===//
// BufferSplitBorrowOp
//===----------------------------------------------------------------------===//

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
      // Nothing more to check if at least one dynamically sized output buffer.
      return success();
    outTotalSize += *optSize;
  }
  auto expectedInBufferType = DeviceBufferType::get(
      getContext(), getCDevice(), std::optional<uint64_t>(outTotalSize));
  if (getInBuffer().getType() != expectedInBufferType) {
    return emitOpError("expected input buffer type does not match "
                       "actual input buffer type (")
           << expectedInBufferType << " vs " << getInBuffer().getType() << ").";
  }
  return success();
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
  auto sizeAttr =
      dyn_cast_if_present<IntegerAttr>(attributes.get(getCSizeAttrName(name)));
  if (!sizeAttr)
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected size attribute.");
  std::optional<uint64_t> optSize = sizeAttr.getUInt();
  auto device = dyn_cast_if_present<DeviceRefAttr>(
      attributes.get(getDDeviceAttrName(name)));
  if (!device)
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected a device attribute.");
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, device, optSize));
  return success();
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
  auto device = dyn_cast_if_present<DeviceRefAttr>(
      attributes.get(getDDeviceAttrName(name)));
  if (!device)
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected a device attribute");
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, device, std::optional<uint64_t>()));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

LogicalResult TensorCreateOp::verify() {
  if (getTensorSpec().getType() != getOutTensor().getType().getTensorSpec())
    return emitOpError("expected spec to match result tensor spec")
           << " (" << getTensorSpec().getType() << " vs "
           << getOutTensor().getType().getTensorSpec() << ").";
  DeviceBufferType expectedBufferType =
      DeviceBufferType::get(getOutTensor().getType());
  if (getBuffer().getType() != expectedBufferType) {
    return emitOpError(
               "expected buffer type to match type implied by tensor spec (")
           << getBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
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

//===----------------------------------------------------------------------===//
// TensorExtractShapeOp
//===----------------------------------------------------------------------===//

LogicalResult TensorExtractShapeOp::verify() {
  TensorSpecType expectedSpecType = TensorSpecType::get(
      getContext(), getTensor().getType().getTensorSpec().getOptDims(),
      std::optional<DType>());
  if (expectedSpecType != getOutShape().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutShape().getType() << " vs " << expectedSpecType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getBDevice()});
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
// KernelExecuteGenericOp
//===----------------------------------------------------------------------===//

LogicalResult KernelExecuteGenericOp::verify() {
  if (getInChain().getType().getDevice() != getGOtherDevice())
    return emitOpError("expected input chain to be on 'other' device ")
           << getGOtherDevice() << ".";
  if (getOutChain().getType().getDevice() != getGOtherDevice())
    return emitOpError("expected output chain to be on 'other' device ")
           << getGOtherDevice() << ".";

  if (getEInOutSignature().size() != getArgs().size())
    return emitOpError("kernel has ")
           << getArgs().size()
           << " arguments, but the in/out signature attribute has arity "
           << getEInOutSignature().size() << ".";

  for (auto [idx, arg] : llvm::enumerate(getArgs())) {
    if (!isa<DeviceBufferType>(arg.getType()) &&
        getEInOutSignature()[idx] != InOutSignatureAttr::kIn) {
      return emitOpError("argument ")
             << idx
             << " is not an !mgp.buffer type, so must be marked as read-only.";
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

LogicalResult KernelExecuteGenericOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, mlir::OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  mlir::OperationName name(getOperationName(), context);

  auto otherDevice = dyn_cast_if_present<DeviceRefAttr>(
      attributes.get(getGOtherDeviceAttrName(name)));
  if (!otherDevice) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected a device attribute for other device.");
  }

  uint64_t numOutDimensions = 0;
  if (auto attr = attributes.get(getINumOutDimensionsAttrName(name))) {
    auto numOutDimensionsAttr = dyn_cast<IntegerAttr>(attr);
    if (!numOutDimensionsAttr)
      return mlir::emitError(
          location.value_or(operands[0].getLoc()),
          "expecting integer attribute for number of output dimensions.");
    numOutDimensions = numOutDimensionsAttr.getUInt();
  }

  inferredReturnTypes.emplace_back(DeviceChainType::get(context, otherDevice));
  while (numOutDimensions-- > 0)
    inferredReturnTypes.emplace_back(IndexType::get(context));
  return success();
}

InOutSignatureAttr::InOutSemantics
KernelExecuteGenericOp::getInOutSemantics(size_t index) {
  GENERICML_ASSERT(index < getNumOperands());
  size_t firstArgsIndex = getTheContextMutable().getOperandNumber() + 1;
  size_t endArgsIndex = firstArgsIndex + getArgs().size();
  if (index < firstArgsIndex || index >= endArgsIndex)
    return InOutSignatureAttr::kIn;
  return getEInOutSignature()[index - firstArgsIndex];
}
