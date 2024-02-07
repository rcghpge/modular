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
            dyn_cast<MGPHasDeviceTypeInterface>(opOperand.get().getType()))
      allDevices.insert(hasDeviceType.getDevice());
  }
  for (OpResult opResult : op->getResults()) {
    if (auto hasDeviceType =
            dyn_cast<MGPHasDeviceTypeInterface>(opResult.getType()))
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
// ModuleLoadOp
//===----------------------------------------------------------------------===//

LogicalResult ModuleLoadOp::verify() {
  return verifyConsistentDevices(getOperation(), {getEDevice()});
}

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

void ChainCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          TypedValue<ContextType> theContext,
                          ValueRange gatingOperands, uint64_t aOtherRuntimeSlot,
                          DeviceRefAttr bHostDevice,
                          DeviceRefAttr cOtherDevice) {
  build(odsBuilder, odsState,
        DeviceChainType::get(odsBuilder.getContext(), cOtherDevice), theContext,
        gatingOperands, aOtherRuntimeSlot, bHostDevice, cOtherDevice);
}

LogicalResult ChainCreateOp::verify() {
  return verifyConsistentDevices(getOperation(),
                                 {getBHostDevice(), getCOtherDevice()});
}

llvm::SmallVector<mlir::Value> ChainCreateOp::getInputChains() {
  return getGatingOperands();
}

void ChainCreateOp::setInputChains(::mlir::ValueRange gatingOperands) {
  getGatingOperandsMutable().assign(gatingOperands);
}

//===----------------------------------------------------------------------===//
// ChainDeviceToHostOp
//===----------------------------------------------------------------------===//

LogicalResult ChainDeviceToHostOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

LogicalResult
ChainDeviceToHostOp::canonicalize(ChainDeviceToHostOp deviceToHostOp,
                                  PatternRewriter &rewriter) {
  // Replace device_to_host(host_to_device(x)) with x.
  Operation *input = deviceToHostOp.getInChain().getDefiningOp();
  if (!input)
    return failure();
  auto hostToDeviceOp = dyn_cast<ChainHostToDeviceOp>(input);
  if (!hostToDeviceOp)
    return failure();
  // devices must match
  if (hostToDeviceOp.getBDevice() != deviceToHostOp.getBDevice())
    return failure();
  // Can only replace from if this is the only user.
  auto users = hostToDeviceOp->getUsers();
  if (std::distance(users.begin(), users.end()) != 1)
    return failure();
  rewriter.replaceOp(deviceToHostOp, hostToDeviceOp.getInChain());
  rewriter.eraseOp(hostToDeviceOp);
  return success();
}

//===----------------------------------------------------------------------===//
// ChainHostToDeviceOp
//===----------------------------------------------------------------------===//

void ChainHostToDeviceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypedValue<MEF::ChainType> inChain,
                                TypedValue<ContextType> theContext,
                                uint64_t aRuntimeSlot,
                                M::DeviceRefAttr bDevice) {
  return build(odsBuilder, odsState,
               DeviceChainType::get(odsBuilder.getContext(), bDevice), inChain,
               theContext, aRuntimeSlot, bDevice);
}

LogicalResult ChainHostToDeviceOp::verify() {
  return verifyConsistentDevices(getOperation(), {getBDevice()});
}

LogicalResult
ChainHostToDeviceOp::canonicalize(ChainHostToDeviceOp hostToDeviceOp,
                                  PatternRewriter &rewriter) {
  // Replace host_to_device(device_to_host(x)) with x.
  Operation *input = hostToDeviceOp.getInChain().getDefiningOp();
  if (!input)
    return failure();
  auto deviceToHostOp = dyn_cast<ChainDeviceToHostOp>(input);
  if (!deviceToHostOp)
    return failure();
  // devices must match
  if (deviceToHostOp.getBDevice() != hostToDeviceOp.getBDevice())
    return failure();
  // Can only replace from if this is the only user.
  auto users = deviceToHostOp->getUsers();
  if (std::distance(users.begin(), users.end()) != 1)
    return failure();
  rewriter.replaceOp(hostToDeviceOp, deviceToHostOp.getInChain());
  rewriter.eraseOp(deviceToHostOp);
  return success();
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

void BufferConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                             Value theContext, uint64_t aRuntimeSlot,
                             ElementsAttr bValue, DeviceRefAttr dDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(
                   dDevice, std::optional<uint64_t>(static_cast<uint64_t>(
                                bValue.getNumElements()))),
               theContext, aRuntimeSlot, bValue, getRawAlign(bValue), dDevice);
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

void BufferConstantLoadOp::build(OpBuilder &odsBuilder,
                                 OperationState &odsState, Value theContext,
                                 uint64_t aRuntimeSlot, uint64_t bModuleIndex,
                                 uint64_t cConstantIndex, uint64_t dSize,
                                 DeviceRefAttr eDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(eDevice, std::optional<uint64_t>(dSize)),
               theContext, aRuntimeSlot, bModuleIndex, cConstantIndex, dSize,
               eDevice);
}

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

//===----------------------------------------------------------------------===//
// BufferAllocStaticOp
//===----------------------------------------------------------------------===//

void BufferAllocStaticOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                TypedValue<DeviceChainType> inChain,
                                TypedValue<ContextType> theContext,
                                uint64_t aRuntimeSlot, uint64_t bSize,
                                std::optional<uint64_t> cAlign,
                                M::DeviceRefAttr dDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(dDevice, std::optional<uint64_t>(bSize)),
               inChain, theContext, aRuntimeSlot, bSize,
               optSizeToRawSize(cAlign), dDevice);
}

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

void BufferAllocDynamicOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
    TypedValue<IndexType> runtimeSize, uint64_t aRuntimeSlot,
    std::optional<uint64_t> bAlign, M::DeviceRefAttr cDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(cDevice, std::optional<uint64_t>()),
               inChain, theContext, runtimeSize, aRuntimeSlot,
               optSizeToRawSize(bAlign), cDevice);
}

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

void BufferReallocStaticOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
    uint64_t aRuntimeSlot, uint64_t bBufferSlot, uint64_t cBufferIndex,
    uint64_t dSize, std::optional<uint64_t> eAlign, M::DeviceRefAttr fDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(fDevice, std::optional<uint64_t>(dSize)),
               inChain, theContext, aRuntimeSlot, bBufferSlot, cBufferIndex,
               dSize, optSizeToRawSize(eAlign), fDevice);
}

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

void BufferReallocDynamicOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
    TypedValue<IndexType> runtimeSize, uint64_t aRuntimeSlot,
    uint64_t bBufferSlot, uint64_t cBufferIndex, std::optional<uint64_t> dAlign,
    M::DeviceRefAttr eDevice) {
  return build(odsBuilder, odsState,
               DeviceBufferType::get(eDevice, std::optional<uint64_t>()),
               inChain, theContext, runtimeSize, aRuntimeSlot, bBufferSlot,
               cBufferIndex, optSizeToRawSize(dAlign), eDevice);
}

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
  return verifyConsistentDevices(getOperation(), {getCDevice()});
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

void BufferGetCachedOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                              TypedValue<MEF::ChainType> inChain,
                              TypedValue<ContextType> theContext,
                              uint64_t aRuntimeSlot, uint64_t bBufferSlot,
                              std::optional<uint64_t> cOptSize,
                              M::DeviceRefAttr dDevice) {
  return build(
      odsBuilder, odsState,
      DeviceBufferType::get(odsBuilder.getContext(), dDevice, cOptSize),
      inChain, theContext, aRuntimeSlot, bBufferSlot,
      optSizeToRawSize(cOptSize), dDevice);
}

LogicalResult BufferGetCachedOp::verify() {
  std::optional<uint64_t> optSize = rawSizeToOptSize(getCRawSize());
  auto expectedBufferType = DeviceBufferType::get(getDDevice(), optSize);
  if (expectedBufferType != getOutBuffer().getType()) {
    return emitOpError("output buffer type does not match expected (")
           << getOutBuffer().getType() << " vs " << expectedBufferType << ").";
  }
  return verifyConsistentDevices(getOperation(), {getDDevice()});
}

//===----------------------------------------------------------------------===//
// BufferHostToDeviceOp
//===----------------------------------------------------------------------===//

void BufferHostToDeviceOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
    TypedValue<DeviceBufferType> hostBuffer,
    TypedValue<DeviceBufferType> otherBuffer, uint64_t aHostRuntimeSlot,
    uint64_t bOtherRuntimeSlot, M::DeviceRefAttr cHostDevice,
    M::DeviceRefAttr dOtherDevice) {
  return build(odsBuilder, odsState,
               DeviceChainType::get(odsBuilder.getContext(), dOtherDevice),
               inChain, theContext, hostBuffer, otherBuffer, aHostRuntimeSlot,
               bOtherRuntimeSlot, cHostDevice, dOtherDevice);
}

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
  GENERICML_ASSERT(index < getNumOperands());
  return index == getOtherBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToHostOp
//===----------------------------------------------------------------------===//

void BufferDeviceToHostOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
    TypedValue<DeviceBufferType> otherBuffer,
    TypedValue<DeviceBufferType> hostBuffer, uint64_t aOtherRuntimeSlot,
    uint64_t bHostRuntimeSlot, M::DeviceRefAttr cOtherDevice,
    M::DeviceRefAttr dHostDevice) {
  return build(odsBuilder, odsState,
               DeviceChainType::get(odsBuilder.getContext(), dHostDevice),
               inChain, theContext, otherBuffer, hostBuffer, aOtherRuntimeSlot,
               bHostRuntimeSlot, cOtherDevice, dHostDevice);
}

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
  GENERICML_ASSERT(index < getNumOperands());
  return index == getHostBufferMutable().getOperandNumber()
             ? InOutSignatureAttr::kOut
             : InOutSignatureAttr::kIn;
}

//===----------------------------------------------------------------------===//
// BufferDeviceToDeviceOp
//===----------------------------------------------------------------------===//

void BufferDeviceToDeviceOp::build(OpBuilder &odsBuilder,
                                   OperationState &odsState,
                                   TypedValue<DeviceChainType> inChain,
                                   TypedValue<ContextType> theContext,
                                   TypedValue<DeviceBufferType> srcBuffer,
                                   TypedValue<DeviceBufferType> dstBuffer,
                                   uint64_t aRuntimeSlot,
                                   M::DeviceRefAttr bDevice) {
  return build(odsBuilder, odsState,
               DeviceChainType::get(odsBuilder.getContext(), bDevice), inChain,
               theContext, srcBuffer, dstBuffer, aRuntimeSlot, bDevice);
}

LogicalResult BufferDeviceToDeviceOp::verify() {
  if (getSrcBuffer().getType().getRawSize() !=
      getDstBuffer().getType().getRawSize())
    return emitOpError("src and dst buffer sizes do not match (")
           << optSizeToString(getSrcBuffer().getType().getOptSize()) << " vs "
           << optSizeToString(getDstBuffer().getType().getOptSize()) << ").";
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

void BufferSplitBorrowOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<DeviceChainType> inChain, TypedValue<ContextType> theContext,
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
  if (!device) {
    return mlir::emitError(location.value_or(operands[0].getLoc()),
                           "expected a device attribute");
  }
  inferredReturnTypes.emplace_back(
      DeviceBufferType::get(context, device, std::optional<uint64_t>()));
  return success();
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

void TensorCreateOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                           TypedValue<DeviceChainType> inChain,
                           TypedValue<ContextType> theContext,
                           TypedValue<DeviceBufferType> buffer,
                           TypedValue<TensorSpecType> tensorSpec,
                           uint64_t aRuntimeSlot, DeviceRefAttr bDevice) {
  build(odsBuilder, odsState,
        DeviceTensorType::get(bDevice, tensorSpec.getType()), inChain,
        theContext, buffer, tensorSpec, aRuntimeSlot, bDevice);
}

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

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

void TensorExtractBufferOp::build(OpBuilder &odsBuilder,
                                  OperationState &odsState,
                                  TypedValue<ContextType> theContext,
                                  TypedValue<DeviceTensorType> tensor,
                                  uint64_t aRuntimeSlot,
                                  DeviceRefAttr bDevice) {
  build(odsBuilder, odsState, DeviceBufferType::get(tensor.getType()),
        theContext, tensor, aRuntimeSlot, bDevice);
}

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
// TensorExtractTensorSpecOp
//===----------------------------------------------------------------------===//

void TensorExtractTensorSpecOp::build(
    OpBuilder &odsBuilder, OperationState &odsState,
    TypedValue<ContextType> theContext, TypedValue<DeviceTensorType> tensor,
    uint64_t aRuntimeSlot, DeviceRefAttr bDevice, bool cIsShapeOnly) {
  build(odsBuilder, odsState,
        TensorSpecType::get(
            odsBuilder.getContext(),
            /*optDims=*/tensor.getType().getTensorSpec().getDims(),
            /*optDType=*/
            cIsShapeOnly ? std::optional<DType>()
                         : tensor.getType().getTensorSpec().getOptDType()),
        theContext, tensor, aRuntimeSlot, bDevice);
}

LogicalResult TensorExtractTensorSpecOp::verify() {
  TensorSpecType expectedSpecType = TensorSpecType::get(
      getContext(),
      /*optDims=*/getTensor().getType().getTensorSpec().getOptDims(),
      /*optDType=*/
      getCIsShapeOnly() ? std::optional<DType>()
                        : getTensor().getType().getTensorSpec().getOptDType());
  if (expectedSpecType != getOutSpec().getType()) {
    return emitOpError("expected the result tensor spec to match the type "
                       "implied by the tensor type (")
           << getOutSpec().getType() << " vs " << expectedSpecType << ").";
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
