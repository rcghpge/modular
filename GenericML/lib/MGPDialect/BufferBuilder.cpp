//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPDialect/BufferBuilder.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPUtils.h"
#include "GraphRT/GRTDialect/GRTOps.h"
#include "mlir/IR/Builders.h"

using namespace M;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// TypeBuilder
//===----------------------------------------------------------------------===//

void TypeBuilder::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// BufferBuilder
//===----------------------------------------------------------------------===//

TypedValue<ContextType> BufferBuilder::getContextForOp(Operation *op) {
  return getContextForBlock(op->getBlock());
}

Value BufferBuilder::anyToNativeOrHostChain(OpBuilder &builder, Location loc,
                                            TypedValue<ContextType> context,
                                            BufferBuilder *chainBuilder,
                                            Value chain) {
  if (chainBuilder == this)
    return chain;
  else
    return chainBuilder->toHostChain(builder, loc, context, chain);
}

Value BufferBuilder::toNativeChain(OpBuilder &builder, Location loc,
                                   TypedValue<ContextType> context,
                                   Value chain) {
  if (isa<MEF::ChainType>(chain.getType()))
    return fromHostChain(builder, loc, context,
                         cast<TypedValue<MEF::ChainType>>(chain));
  else
    return chain;
}

void BufferBuilder::vtableAnchor() {}

//===----------------------------------------------------------------------===//
// MGPBufferBuilder
//===----------------------------------------------------------------------===//

TypedValue<MEF::ChainType> MGPBufferBuilder::setupForModel(
    ModelOp modelOp, MGPBufferBuilder *hostBufferBuilder, OpBuilder &builder,
    Location loc, TypedValue<MEF::ChainType> inChain) {
  this->hostBufferBuilder = hostBufferBuilder;

  // Allocate a runtime slot to hold the runtime.
  TypedValue<ContextType> context;
  modelOp.walk([&](ContextCreateOp contextCreateOp) {
    runtimeSlot = contextCreateOp.getNumSlots();
    contextCreateOp.setNumSlots(runtimeSlot + 1);
    context = contextCreateOp.getResult();
    return WalkResult::interrupt();
  });

  // Figure out the slot which holds the device map.
  uint64_t deviceMapSlot = 0;
  modelOp.walk([&](CheckRuntimeOp checkRuntimeOp) {
    deviceMapSlot = checkRuntimeOp.getASlot();
  });

  // Create a runtime for the device.
  return builder.create<MGP::RuntimeCreateOp>(
      loc,
      /*inChain=*/inChain,
      /*theContext=*/context,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDeviceSpecMapSlot=*/deviceMapSlot,
      /*cDevice=*/device);
}

void MGPBufferBuilder::recoverFromModel(ModelOp modelOp,
                                        MGPBufferBuilder *hostBufferBuilder) {
  this->hostBufferBuilder = hostBufferBuilder;

  // Recover the runtime slot index from the unique mgp.runtime.create op
  // for this device.
  modelOp.walk([&](RuntimeCreateOp runtimeCreateOp) {
    if (runtimeCreateOp.getCDevice() != device)
      return WalkResult::skip();
    runtimeSlot = runtimeCreateOp.getARuntimeSlot();
    return WalkResult::interrupt();
  });
  // If there's no mgp.runtime.create, leave the slot index as 0.
  // (Helps unit tests.)

  // Recover the function and constant from all mgp.module.load ops for
  // this device.
  size_t moduleIndex = 0;
  modelOp.walk([&](ModuleLoadOp moduleLoadOp) {
    for (auto [functionIndex, name] :
         llvm::enumerate(moduleLoadOp.getCFunctionSymbols())) {
      functionToIndexPair[name] = std::make_pair(moduleIndex, functionIndex);
    }
    for (auto [constantIndex, name] :
         llvm::enumerate(moduleLoadOp.getDConstantSymbols())) {
      constantToIndexPair[name] = std::make_pair(moduleIndex, constantIndex);
    }
    ++moduleIndex;
  });
}

TypedValue<ContextType> MGPBufferBuilder::getContextForOp(Operation *op) {
  if (auto hasSlotOp = dyn_cast<MGPHasRuntimeSlotOpInterface>(op))
    return hasSlotOp.getTheContext();
  return BufferBuilder::getContextForOp(op);
}

DeviceRefAttr MGPBufferBuilder::getDevice(OpBuilder &builder) { return device; }

Value MGPBufferBuilder::bufferConstant(OpBuilder &builder, Location loc,
                                       TypedValue<MEF::ChainType> inChain,
                                       TypedValue<ContextType> context,
                                       ElementsAttr value) {
  bool isOnHost = this == hostBufferBuilder;
  return builder.create<BufferConstantOp>(loc,
                                          /*inChain*/ inChain,
                                          /*theContext=*/context,
                                          /*aRuntimeSlot=*/runtimeSlot,
                                          /*bValue=*/value,
                                          /*dDevice=*/device,
                                          /*eIsTrivial=*/isOnHost);
}

Value MGPBufferBuilder::tensorExtractBuffer(OpBuilder &builder, Location loc,
                                            TypedValue<ContextType> context,
                                            Value tensor) {
  return builder.create<TensorExtractBufferOp>(
      loc,
      /*theContext=*/context,
      /*tensor=*/cast<TypedValue<DeviceTensorType>>(tensor),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}

TypedValue<TensorSpecType>
MGPBufferBuilder::tensorExtractTensorSpec(OpBuilder &builder, Location loc,
                                          TypedValue<ContextType> context,
                                          Value tensor, bool isShapeOnly) {
  return builder.create<TensorExtractTensorSpecOp>(
      loc,
      /*theContext=*/context,
      /*tensor=*/cast<TypedValue<DeviceTensorType>>(tensor),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device,
      /*cIsShapeOnly=*/isShapeOnly);
}

Value MGPBufferBuilder::tensorCreate(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inChain, Value buffer,
                                     TypedValue<TensorSpecType> tensorSpec) {
  // The buffer type and tensor spec types don't need to agree on whether they
  // are static. This can arise when lowering value-semantic reshapes
  // into buffer-semantic form. For example:
  //   %z = reshape %x, %y : (tensor<?x?xf32>, tensor<2xui32>) ->
  //                         tensor<8x16xf32>
  // Here %z will be created from the same buffer as %x, but will now have
  // a known size. Conversely, in:
  //   %z = reshape %x, %y : (tensor<8x4xf32>, tensor<2xui32>) ->
  //                         tensor<?x?xf32>
  // the buffer for %z will now have an unknown size.
  // We may need to insert mgp.buffer.{unsize, resize} ops to compensate.
  auto deviceBuffer = cast<TypedValue<DeviceBufferType>>(buffer);
  std::optional<uint64_t> specOptSize = tensorSpec.getType().getSizeInBytes();
  std::optional<uint64_t> bufferOptSize = deviceBuffer.getType().getOptSize();
  if (specOptSize.has_value() && !bufferOptSize.has_value()) {
    deviceBuffer = cast<TypedValue<DeviceBufferType>>(
        bufferResize(builder, loc, context, deviceBuffer, *specOptSize));
  } else if (!specOptSize.has_value() && bufferOptSize.has_value()) {
    deviceBuffer = cast<TypedValue<DeviceBufferType>>(
        bufferUnsize(builder, loc, context, deviceBuffer));
  }
  return builder.create<TensorCreateOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*buffer=*/deviceBuffer,
      /*tensorSpec=*/tensorSpec,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::bufferAllocStatic(OpBuilder &builder, Location loc,
                                          TypedValue<ContextType> context,
                                          Value inChain, uint64_t size,
                                          uint64_t alignment) {
  return builder.create<BufferAllocStaticOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bSize=*/size,
      /*cAlign=*/alignment,
      /*dDevice=*/device);
}

Value MGPBufferBuilder::bufferAllocDynamic(OpBuilder &builder, Location loc,
                                           TypedValue<ContextType> context,
                                           Value inChain, Value size,
                                           uint64_t alignment) {
  return builder.create<BufferAllocDynamicOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*runtimeSize=*/cast<TypedValue<IndexType>>(size),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bAlign=*/alignment,
      /*cDevice=*/device);
}

Value MGPBufferBuilder::bufferTransferFromHost(OpBuilder &builder, Location loc,
                                               TypedValue<ContextType> context,
                                               Value inChain,
                                               BufferBuilder *srcBuilder,
                                               Value srcBuffer,
                                               Value dstBuffer) {
  return builder.create<BufferHostToDeviceOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          srcBuilder->toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*hostBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
      /*otherBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
      /*aHostRuntimeSlot=*/hostBufferBuilder->runtimeSlot,
      /*bOtherRuntimeSlot=*/runtimeSlot,
      /*cHostDevice=*/hostBufferBuilder->device,
      /*dOtherDevice=*/device);
}

Value MGPBufferBuilder::bufferTransferToHost(OpBuilder &builder, Location loc,
                                             TypedValue<ContextType> context,
                                             Value inChain, Value srcBuffer,
                                             BufferBuilder *dstBuilder,
                                             Value dstBuffer) {
  return builder.create<BufferDeviceToHostOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*otherBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
      /*hostBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
      /*aOtherRuntimeSlot=*/runtimeSlot,
      /*bHostRuntimeSlot=*/hostBufferBuilder->runtimeSlot,
      /*cOtherDevice=*/device,
      /*dHostDevice=*/hostBufferBuilder->device);
}

Value MGPBufferBuilder::bufferTransferDevice(OpBuilder &builder, Location loc,
                                             TypedValue<ContextType> context,
                                             Value inChain, Value srcBuffer,
                                             BufferBuilder *dstBufferBuilder,
                                             Value dstBuffer) {
  auto dstMGPBufferBuilder = static_cast<MGPBufferBuilder *>(dstBufferBuilder);
  return builder.create<BufferDeviceToDeviceOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*srcBuffer=*/cast<TypedValue<DeviceBufferType>>(srcBuffer),
      /*dstBuffer=*/cast<TypedValue<DeviceBufferType>>(dstBuffer),
      /*aSrcRuntimeSlot=*/runtimeSlot,
      /*bDstRuntimeSlot=*/dstMGPBufferBuilder->runtimeSlot,
      /*cSrcDevice=*/device,
      /*dDstDevice=*/dstMGPBufferBuilder->device);
}

Value MGPBufferBuilder::bufferUnsize(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inBuffer) {
  return builder.create<BufferUnsizeOp>(
      loc,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer));
}

Value MGPBufferBuilder::bufferResize(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inBuffer, uint64_t size) {
  return builder.create<BufferResizeOp>(
      loc,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*aSize=*/size,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::debugTensorPrint(OpBuilder &builder, Location loc,
                                         TypedValue<ContextType> context,
                                         Value inChain, Value buffer,
                                         TypedValue<TensorSpecType> spec,
                                         StringAttr label) {
  return builder.create<DebugTensorPrintOp>(
      loc,
      /*inChain=*/toHostChain(builder, loc, context, inChain),
      /*theContext=*/context,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*spec=*/spec,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bLabel=*/label,
      /*cDevice=*/device);
}

/// Returns arrayAttr, which should be an array of StringAttrs, encoded as
/// a MultiLineStringAttr. Also adds each string to the given map.
static MultiLineStringAttr
toMultiline(OpBuilder &builder, uint64_t moduleIndex, ArrayAttr arrayAttr,
            llvm::StringMap<std::pair<uint64_t, uint64_t>> &nameToIndexPair) {
  SmallVector<StringAttr> stringAttrs;
  if (arrayAttr) {
    uint64_t nameIndex = 0;
    for (StringRef symbol : arrayAttr.getAsValueRange<mlir::StringAttr>()) {
      stringAttrs.emplace_back(builder.getStringAttr(symbol));
      nameToIndexPair.insert({symbol, std::make_pair(moduleIndex, nameIndex)});
      ++nameIndex;
    }
  }
  return MultiLineStringAttr::get(builder.getContext(), stringAttrs);
}

TypedValue<MEF::ChainType> MGPBufferBuilder::moduleLoad(
    OpBuilder &builder, Location loc, TypedValue<ContextType> context,
    TypedValue<MEF::ChainType> inChain, ElementsAttr image,
    ArrayAttr functionSymbols, ArrayAttr constantSymbols) {
  auto functionSymbolsAttr = toMultiline(builder, nextModuleIndex,
                                         functionSymbols, functionToIndexPair);
  auto constantSymbolsAttr = toMultiline(builder, nextModuleIndex,
                                         constantSymbols, constantToIndexPair);
  ++nextModuleIndex;
  return builder.create<ModuleLoadOp>(loc,
                                      /*inChain=*/
                                      cast<TypedValue<MEF::ChainType>>(inChain),
                                      /*theContext=*/context,
                                      /*aRuntimeSlot=*/runtimeSlot,
                                      /*bImage=*/image,
                                      /*cFunctionSymbols=*/functionSymbolsAttr,
                                      /*dConstantSymbols=*/constantSymbolsAttr,
                                      /*eDevice=*/device);
}

Value MGPBufferBuilder::kernelExecute(
    OpBuilder &builder, Location loc, TypedValue<ContextType> context,
    StringRef kernelName, Value inChain, ArrayRef<Value> args,
    ArrayRef<InOutSignatureAttr::InOutSemantics> inOutSemantics,
    ArrayRef<Value> inDimensions, uint64_t numOutDimensions,
    SmallVectorImpl<Value> &outDimensions, StringRef summary, bool isAsync) {
  // Use the kernel name to determine the (module index, function index) pair
  // on the host.
  auto [moduleIndex, functionIndex] =
      hostBufferBuilder->getFunction(kernelName);

  auto inOutSignatureAttr =
      InOutSignatureAttr::get(builder.getContext(), inOutSemantics);
  std::string augmentedSummary =
      this == hostBufferBuilder ? summary.str()
                                : (Twine(summary) + "[" + device.getLabel() +
                                   ":" + Twine(device.getId()) + "]")
                                      .str();

  auto executeOp = builder.create<KernelExecuteOp>(
      loc,
      /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*args=*/args,
      /*inDimensions=*/inDimensions,
      /*aHostRuntimeSlot=*/hostBufferBuilder->runtimeSlot,
      /*bOtherRuntimeSlot=*/runtimeSlot,
      /*cHostModuleIndex=*/moduleIndex,
      /*dHostFunctionIndex=*/functionIndex,
      /*eInOutSignature=*/inOutSignatureAttr,
      /*fHostDevice=*/hostBufferBuilder->device,
      /*gOtherDevice=*/device,
      /*hSummary=*/augmentedSummary,
      /*iNumOutDimensions=*/numOutDimensions,
      /*jIsAsync=*/isAsync);

  for (Value outDimension : executeOp.getOutDimensions())
    outDimensions.emplace_back(outDimension);

  return executeOp.getOutChain();
}

Value MGPBufferBuilder::chainCreate(OpBuilder &builder, Location loc,
                                    TypedValue<ContextType> context,
                                    ArrayRef<Value> inChains) {
  return builder.create<ChainCreateOp>(
      loc,
      /*theContext=*/context,
      /*gatingOperands=*/inChains,
      /*aOtherRuntimeSlot=*/runtimeSlot,
      /*bHostDevice=*/hostBufferBuilder->device,
      /*cOtherDevice=*/device);
}

std::pair<TypedValue<MEF::ChainType>, uint64_t>
MGPBufferBuilder::cache(OpBuilder &initBuilder, Location loc,
                        TypedValue<ContextType> context,
                        TypedValue<MEF::ChainType> inChain,
                        ContextCreateOp contextCreateOp, Value buffer) {
  uint64_t bufferSlot = contextCreateOp.getNumSlots();
  contextCreateOp.setNumSlots(bufferSlot + 1);
  auto cacheOp = initBuilder.create<BufferCacheOp>(
      loc, /*inChain=*/inChain, /*theContext=*/contextCreateOp,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bBufferSlot=*/bufferSlot, /*cDevice=*/device);
  return {cacheOp.getOutChain(), bufferSlot};
}

Value MGPBufferBuilder::getCached(OpBuilder &builder, Location loc,
                                  TypedValue<ContextType> context,
                                  TypedValue<MEF::ChainType> inChain,
                                  uint64_t bufferSlot,
                                  Type expectedBufferType) {
  return builder.create<BufferGetCachedOp>(
      loc, /*inChain=*/inChain, /*theContext=*/context,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bBufferSlot=*/bufferSlot,
      /*cRawSize=*/
      cast<DeviceBufferType>(expectedBufferType).getRawSize().getUInt(),
      /*dDevice=*/device);
}

TypedValue<MEF::ChainType>
MGPBufferBuilder::toHostChain(OpBuilder &builder, Location loc,
                              TypedValue<ContextType> context, Value inChain) {
  if (isa<MEF::ChainType>(inChain.getType()))
    return cast<TypedValue<MEF::ChainType>>(inChain);
  return builder.create<ChainDeviceToHostOp>(
      loc,
      /*inChain=*/cast<TypedValue<DeviceChainType>>(inChain),
      /*theContext=*/context,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::fromHostChain(OpBuilder &builder, Location loc,
                                      TypedValue<ContextType> context,
                                      TypedValue<MEF::ChainType> inChain) {
  return builder.create<ChainHostToDeviceOp>(
      loc, /*inChain=*/inChain, /*theContext=*/context,
      /*aRuntimeSlot=*/runtimeSlot, /*bDevice=*/device);
}

Value MGPBufferBuilder::bufferConcat(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inChain, Value outBuffer,
                                     ArrayRef<Value> inBuffers) {
  return builder.create<BufferConcatOp>(
      loc, /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*outBuffer=*/cast<TypedValue<DeviceBufferType>>(outBuffer),
      /*inBuffers=*/inBuffers,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}

SmallVector<Value> MGPBufferBuilder::splitBorrow(
    OpBuilder &builder, Location loc, TypedValue<ContextType> context,
    Value inChain, Value inBuffer, ArrayRef<Value> runtimeSizes,
    ArrayRef<std::optional<uint64_t>> outBufferSizes) {
  SmallVector<uint64_t> rawSizes;
  for (auto optSize : outBufferSizes)
    rawSizes.emplace_back(optSizeToRawSize(optSize));
  return builder
      .create<BufferSplitBorrowOp>(
          loc,
          /*inChain=*/
          cast<TypedValue<DeviceChainType>>(
              toNativeChain(builder, loc, context, inChain)),
          /*theContext=*/context,
          /*inBuffer=*/inBuffer,
          /*runtimeSizes=*/runtimeSizes,
          /*aRuntimeSlot=*/runtimeSlot,
          /*bOutBufferRawSizes=*/rawSizes,
          /*cDevice=*/device)
      .getOutBuffers();
}

Value MGPBufferBuilder::borrowSectionStatic(OpBuilder &builder, Location loc,
                                            TypedValue<ContextType> context,
                                            Value inBuffer, uint64_t offset,
                                            uint64_t size) {
  return builder.create<BufferSectionBorrowStaticOp>(
      loc,
      /*theContext=*/context,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bOffset=*/offset,
      /*cSize=*/size,
      /*dDevice=*/device);
}

Value MGPBufferBuilder::borrowSectionDynamic(OpBuilder &builder, Location loc,
                                             TypedValue<ContextType> context,
                                             Value inBuffer, Value scale,
                                             uint64_t unscaledOffset,
                                             uint64_t unscaledSize) {
  return builder.create<BufferSectionBorrowDynamicOp>(
      loc,
      /*theContext=*/context,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*scale=*/cast<TypedValue<IndexType>>(scale),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bUnscaledOffset=*/unscaledOffset,
      /*cUnscaledSize=*/unscaledSize,
      /*dDevice=*/device);
}

Value MGPBufferBuilder::bufferToBool(OpBuilder &builder, Location loc,
                                     TypedValue<ContextType> context,
                                     Value inChain, Value inBuffer) {
  return builder.create<BufferToBoolOp>(
      loc, /*inChain=*/
      cast<TypedValue<DeviceChainType>>(
          toNativeChain(builder, loc, context, inChain)),
      /*theContext=*/context,
      /*inBuffer=*/cast<TypedValue<DeviceBufferType>>(inBuffer),
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}

Value MGPBufferBuilder::bufferSetWithIndex(OpBuilder &builder, Location loc,
                                           TypedValue<ContextType> context,
                                           Value buffer,
                                           ArrayRef<Value> values) {
  return builder.create<BufferSetWithIndexOp>(
      loc, /*theContext=*/context,
      /*buffer=*/cast<TypedValue<DeviceBufferType>>(buffer),
      /*values=*/values,
      /*aRuntimeSlot=*/runtimeSlot,
      /*bDevice=*/device);
}
