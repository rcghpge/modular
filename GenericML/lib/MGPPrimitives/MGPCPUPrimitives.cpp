//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/StateContext.h"
#include "ArraySupport/Tensor.h"
#include "ArraySupport/TensorBufferRef.h"
#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/DebugPrint.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "LLCL/Support/Chain.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::LLCL;
using namespace M::GML;
using namespace M::MGP;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static std::optional<size_t> getRuntimeSize(uint64_t rawSize) {
  return hasRawSize(rawSize)
             ? std::optional<size_t>(rawSizeToRuntimeSize(rawSize))
             : std::nullopt;
}

static std::optional<size_t> getRuntimeAlign(uint64_t rawAlign) {
  return getRuntimeSize(rawAlign);
}

TensorBufferRef createBorrowedSlice(TensorBufferRef &borroweBufferRef,
                                    size_t offset, size_t size) {
  const void *newBuffer = borroweBufferRef.getBuffer<uint8_t>() + offset;

  return TensorBufferRef::create(const_cast<void *>(newBuffer), size,
                                 borroweBufferRef.getMemStorageHandle(),
                                 std::optional<size_t>());
}

//===----------------------------------------------------------------------===//
// mgp.runtime.create.cpu
//===----------------------------------------------------------------------===//

void mgp_runtime_create_cpu(Chain inChain, StateContext &theContext,
                            Result<Chain> outChain,
                            MEFAttr<uint64_t> aRuntimeSlot,
                            MEFAttr<uint64_t> bDeviceSpecMapSlot,
                            MEFAttr<uint64_t> cDeviceId, Runtime &runtime) {
  // Make sure the runtime actually has the "cpu" device we are trying to use.
  const auto &deviceSpecMap =
      theContext.getStateSlotPayload<RuntimeDeviceSpecMap>(
          static_cast<size_t>(*bDeviceSpecMapSlot));
  DeviceRef device(kCPULabel, *cDeviceId);
  auto itr = deviceSpecMap.map.find(device);
  if (itr == deviceSpecMap.map.end())
    return std::move(outChain).setToError(Twine("no device matching '") +
                                          device.toString() + "' is available");

  // Establish an execution engine, to be shared amongst all loaded models
  // across all models using this runtime.
  //
  // TODO(#30647): Currently there's no garbage collection of dylibs loaded into
  //               this engine.
  ErrorOr<KGEN::ExecutionEngine *> engineOr =
      runtime.context->createIfMissing<KGEN::ExecutionEngine>(
          [&]() -> ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> {
            // Initialize the host target.
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmParser();
            llvm::InitializeNativeTargetAsmPrinter();
            auto targetMachineOr = getTargetMachineForHost();
            if (targetMachineOr.isError())
              return targetMachineOr.takeError();

            // Create the execution engine.
            KGEN::ExecutionEngineOptions options;
            options.registerPerfPlugins =
                runtime.getProfilerDebuginfo() ==
                RuntimeOptions::ProfilerDebuginfo::kPerfProfiler;
            return KGEN::ExecutionEngine::createWithStandardLayers(
                std::move(options), **targetMachineOr);
          });

  if (engineOr.isError())
    return std::move(outChain).setToError(engineOr.takeError());

  theContext.emplaceStateSlot<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot),
      /*numBuffers=*/0, static_cast<size_t>(*cDeviceId), *engineOr);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.module.load.cpu
//===----------------------------------------------------------------------===//

static void mgp_module_load_cpu(Chain inChain, StateContext &theContext,
                                Result<Chain> outChain,
                                MEFAttr<uint64_t> aRuntimeSlot,
                                MEFDynamicAttr bImage,
                                MEFAttr<StringRef> cFunctionSymbols,
                                MEFAttr<StringRef> dConstantSymbols,
                                MEFAttr<uint64_t> eDeviceId, Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*eDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  if (!dConstantSymbols->empty() && !(dConstantSymbols->size() == 0) &&
      !(dConstantSymbols->size() == 1 && *dConstantSymbols->begin() == '\0')) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu does not support constants");
  }

  // Create a kernels context.
  CompiledKernelsContext &module =
      *cpuRuntime.modules.emplace_back(RCRef<CompiledKernelsContext>::create(
          cpuRuntime.engine,
          runtime.getProfilerDebuginfo() ==
              RuntimeOptions::ProfilerDebuginfo::kSOProfiler));

  // Decode the image.
  ArrayRef<uint8_t> image;
  if (bImage.decode(&image).isError()) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu requires image to be a byte array");
  }
  if (image.empty()) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu requires a non-empty image to load");
  }

  // Split the combined function symbols into a list of individual kernel names.
  // Remember that M.multiline strings are explicitly null-terminated when
  // encoded into MEF.
  SmallVector<StringRef> functionSymbols;
  StringRef combined = *cFunctionSymbols;
  size_t start = 0;
  size_t end = combined.size();
  while (start < end) {
    size_t next = combined.find('\n', start);
    if (next == std::string::npos) {
      ASSERT_STREAM(start + 1 == end && combined[start] == '\0');
      break;
    } else {
      size_t len = next - start;
      if (len)
        functionSymbols.emplace_back(combined.data() + start, len);
      start = next + 1;
    }
  }
  if (functionSymbols.empty()) {
    return std::move(outChain).setToError(
        "expecting at least one function symbol to load");
  }

  // Load the binary.
  ErrorOrSuccess errOr = module.load(image, functionSymbols);
  if (errOr)
    return std::move(outChain).setToError(errOr.takeError());

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.chain.create.cpu
//===----------------------------------------------------------------------===//

static void mgp_chain_create_cpu(StateContext &theContext,
                                 RemainingArguments gatingOperands,
                                 Result<Chain> outChain,
                                 MEFAttr<uint64_t> aOtherRuntimeSlot,
                                 MEFAttr<uint64_t> bHostDeviceId,
                                 MEFAttr<uint64_t> cOtherDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aOtherRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bHostDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cOtherDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.chain.{device_to_host, host_to_device}.cpu
//===----------------------------------------------------------------------===//

static void mgp_chain_noop_cpu(Chain inChain, StateContext &theContext,
                               Result<Chain> outChain,
                               MEFAttr<uint64_t> aRuntimeSlot,
                               MEFAttr<uint64_t> bDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.constant.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_constant_cpu(Chain inChain, StateContext &theContext,
                                    Result<TensorBufferRef> outBuffer,
                                    MEFAttr<uint64_t> aRuntimeSlot,
                                    MEFAttr<ArrayRef<uint8_t>> bValue,
                                    MEFAttr<uint64_t> cRawAlign,
                                    MEFAttr<uint64_t> dDeviceId,
                                    MEFAttr<bool> eIsTrivial) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*dDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  std::optional<size_t> optAlign;
  if (hasRawSize(*cRawAlign))
    optAlign = rawSizeToRuntimeSize(*cRawAlign);

  void *buffer =
      reinterpret_cast<void *>(const_cast<uint8_t *>(bValue->data()));
  std::move(outBuffer).emplace(TensorBufferRef::createWithNonTrackedMemory(
      buffer, bValue->size(), optAlign));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.constant.load.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_constant_load_cpu(StateContext &theContext,
                                         Result<TensorBufferRef> outBuffer,
                                         MEFAttr<uint64_t> aRuntimeSlot,
                                         MEFAttr<uint64_t> bModuleIndex,
                                         MEFAttr<uint64_t> cConstantIndex,
                                         MEFAttr<uint64_t> dSize,
                                         MEFAttr<uint64_t> eDeviceId) {
  return std::move(outBuffer).setToError(
      "mgp.buffer.constant.load.cpu is not supported");
}

//===----------------------------------------------------------------------===//
// mgp.buffer.alloc.static.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_alloc_static_cpu(Chain inChain, StateContext &theContext,
                                        Result<TensorBufferRef> outBuffer,
                                        MEFAttr<uint64_t> aRuntimeSlot,
                                        MEFAttr<uint64_t> bSize,
                                        MEFAttr<uint64_t> cRawAlign,
                                        MEFAttr<uint64_t> dDeviceId,
                                        Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*dDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  if (!isValidPlainSize(*bSize))
    return std::move(outBuffer).setToError("invalid size attribute");
  if (!isValidRawAlign(*cRawAlign))
    return std::move(outBuffer).setToError("invalid align attribute");
  size_t runtimeSize = plainSizeToRuntimeSize(*bSize);
  std::move(outBuffer).emplace(TensorBufferRef::createOwned(
      runtime, runtimeSize, getRuntimeAlign(*cRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.alloc.dynamic.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_alloc_dynamic_cpu(
    Chain inChain, StateContext &theContext, size_t runtimeSize,
    Result<TensorBufferRef> outBuffer, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bRawAlign, MEFAttr<uint64_t> cDeviceId,
    Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  if (!isValidRawAlign(*bRawAlign))
    return std::move(outBuffer).setToError("invalid align attribute");
  std::move(outBuffer).emplace(TensorBufferRef::createOwned(
      runtime, runtimeSize, getRuntimeAlign(*bRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.realloc.static.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_realloc_static_cpu(
    Chain inChain, StateContext &theContext, Result<TensorBufferRef> outBuffer,
    MEFAttr<uint64_t> aRuntimeSlot, MEFAttr<uint64_t> bBufferSlot,
    MEFAttr<uint64_t> cBufferIndex, MEFAttr<uint64_t> dSize,
    MEFAttr<uint64_t> eRawAlign, MEFAttr<uint64_t> fDeviceId,
    Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*fDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  if (!isValidPlainSize(*dSize))
    return std::move(outBuffer).setToError("invalid size attribute");
  if (!isValidRawAlign(*eRawAlign))
    return std::move(outBuffer).setToError("invalid align attribute");
  size_t runtimeSize = plainSizeToRuntimeSize(*dSize);
  StateSlot &stateSlot =
      theContext.getStateSlot(static_cast<size_t>(*bBufferSlot));
  std::move(outBuffer).emplace(stateSlot.getOrCreateBuffer(
      *cBufferIndex, runtime, runtimeSize, getRuntimeAlign(*eRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.realloc.dynamic.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_realloc_dynamic_cpu(
    Chain inChain, StateContext &theContext, size_t runtimeSize,
    Result<TensorBufferRef> outBuffer, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bBufferSlot, MEFAttr<uint64_t> cBufferIndex,
    MEFAttr<uint64_t> dRawAlign, MEFAttr<uint64_t> eDeviceId,
    Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*eDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  if (!isValidRawAlign(*dRawAlign))
    return std::move(outBuffer).setToError("invalid align attribute");
  StateSlot &stateSlot =
      theContext.getStateSlot(static_cast<size_t>(*bBufferSlot));
  std::move(outBuffer).emplace(stateSlot.getOrCreateBuffer(
      *cBufferIndex, runtime, runtimeSize, getRuntimeAlign(*dRawAlign)));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.alloc_pinned.{static,dynamic}
//===----------------------------------------------------------------------===//

static void mgp_buffer_alloc_pinned_static_cpu(
    Chain inChain, StateContext &theContext, Result<TensorBufferRef> outBuffer,
    MEFAttr<uint64_t> aRuntimeSlot, MEFAttr<uint64_t> bSize,
    MEFAttr<uint64_t> cRawAlign, MEFAttr<uint64_t> dHostDeviceId,
    MEFAttr<uint64_t> eDeviceId) {
  return std::move(outBuffer).setToError(
      "mgp.buffer.alloc_pinned.static.cpu is not implemented");
}

static void mgp_buffer_alloc_pinned_dynamic_cpu(
    Chain inChain, StateContext &theContext, size_t runtimeSize,
    Result<TensorBufferRef> outBuffer, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bMinPinnedSize, MEFAttr<uint64_t> cRawAlign,
    MEFAttr<uint64_t> dHostDeviceId, MEFAttr<uint64_t> eDeviceId) {
  return std::move(outBuffer).setToError(
      "mgp.buffer.alloc_pinned.dynamic.cpu is not implemented");
}

//===----------------------------------------------------------------------===//
// mgp.buffer.cache.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_cache_cpu(Chain inChain, StateContext &theContext,
                                 AsyncValueRef<TensorBufferRef> buffer,
                                 Result<Chain> outChain,
                                 MEFAttr<uint64_t> aRuntimeSlot,
                                 MEFAttr<uint64_t> bBufferSlot,
                                 MEFAttr<uint64_t> cDeviceId,
                                 Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  theContext.emplaceStateSlot<DefaultStateSlot>(
      static_cast<size_t>(*bBufferSlot), /*numBuffers=*/1);
  auto &defaultStateSlot = theContext.getStateSlot(*bBufferSlot);

  // Always borrow the buffer.
  ErrorOrSuccess errOr = defaultStateSlot.setBuffer(
      /*bufferIndex=*/0, buffer.get().borrow());

  if (errOr.isError())
    return std::move(outChain).setToError(errOr.takeError());
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.get_cached.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_get_cached_cpu(Chain inChain, StateContext &theContext,
                                      Result<TensorBufferRef> outBuffer,
                                      MEFAttr<uint64_t> aRuntimeSlot,
                                      MEFAttr<uint64_t> bBufferSlot,
                                      MEFAttr<uint64_t> cRawSize,
                                      MEFAttr<uint64_t> dDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*dDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  auto &stateSlot = theContext.getStateSlot(*bBufferSlot);
  ErrorOr<TensorBufferRef> errOr = stateSlot.getBuffer(/*bufferIndex=*/0);
  if (errOr)
    return std::move(outBuffer).setToError(errOr.takeError());
  std::optional<size_t> optExpectedSize = getRuntimeSize(*cRawSize);
  if (optExpectedSize && errOr->getSize() != *optExpectedSize)
    return std::move(outBuffer).setToError(
        Twine("buffer cached at slot ") + Twine(*bBufferSlot) +
        " has different size than expected (" + Twine(errOr->getSize()) +
        " vs " + Twine(*optExpectedSize) + ")");
  std::move(outBuffer).emplace(std::move(*errOr));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.{host_to_device, device_to_host}.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_no_transfers_cpu(
    Chain inChain, StateContext &theContext, TensorBufferRef &hostBuffer,
    TensorBufferRef &otherBuffer, Result<Chain> outChain,
    MEFAttr<uint64_t> aHostRuntimeSlot, MEFAttr<uint64_t> bOtherRuntimeSlot,
    MEFAttr<uint64_t> cHostDeviceId, MEFAttr<uint64_t> dOtherDeviceId) {
  std::move(outChain).setToError(
      "mgp.buffer.{host_to_device, device_to_host}.cpu not supported");
}

//===----------------------------------------------------------------------===//
// mgp.buffer.device_to_device.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_device_to_device_cpu(
    Chain inChain, StateContext &theContext, TensorBufferRef &srcBuffer,
    TensorBufferRef &dstBuffer, Result<Chain> outChain,
    MEFAttr<uint64_t> aSrcRuntimeSlot, MEFAttr<uint64_t> bDstRuntimeSlot,
    MEFAttr<uint64_t> cSrcDeviceId, MEFAttr<uint64_t> dDstDeviceId) {
  if (*cSrcDeviceId != *dDstDeviceId) {
    return std::move(outChain).setToError(
        "mgp.buffer.device_to_device.cpu does not support cross device "
        "transfers");
  }
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aSrcRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cSrcDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  if (srcBuffer.getSize() != dstBuffer.getSize())
    return std::move(outChain).setToError(
        Twine("attempting to copy from buffer of size ") +
        Twine(srcBuffer.getSize()) + " to buffer of size " +
        Twine(dstBuffer.getSize()));
  profiledMemcpy(dstBuffer.getBuffer(), srcBuffer.getBuffer(),
                 srcBuffer.getSize());
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.resize.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_resize_cpu(AsyncValueRef<TensorBufferRef> inBuffer,
                                  RemainingResults outBuffer,
                                  MEFAttr<uint64_t> aSize,
                                  MEFAttr<uint64_t> bDeviceId,
                                  EncodedLocation loc) {
  if (inBuffer->getSize() != static_cast<size_t>(*aSize)) {
    return std::move(outBuffer).setAllToError(
        Twine("attempting to resize from buffer of size ") +
            Twine(inBuffer->getSize()) + " to buffer of size " + Twine(*aSize),
        loc);
  }
  *outBuffer.begin() = std::move(inBuffer);
}

//===----------------------------------------------------------------------===//
// mgp.buffer.concat.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_concat_cpu(Chain inChain, StateContext &theContext,
                                  AsyncValueRef<TensorBufferRef> outBuffer,
                                  RemainingArguments inBuffers,
                                  Result<Chain> outChain,
                                  MEFAttr<uint64_t> aRuntimeSlot,
                                  MEFAttr<uint64_t> bDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outChain).setToError("mismatched device id");

  uint8_t *const out = outBuffer->getBuffer<uint8_t>();
  const size_t outSize = outBuffer->getSize();
  size_t totalSize = 0; // total size of all input buffers processed so far
  for (const auto &inBuffer : inBuffers) {
    const TensorBufferRef &ref = inBuffer.get<TensorBufferRef>();
    const size_t thisSize = ref.getSize();
    if (totalSize + thisSize <= outSize) {
      // Just don't copy if it turns out the input sizes overflow the output.
      // We'll report failure below.
      memcpy(out + totalSize, ref.getBuffer(), thisSize);
    }
    totalSize += thisSize;
  }
  if (totalSize != outBuffer->getSize()) {
    return std::move(outChain).setToError(
        Twine("output buffer size must be sum of input buffer sizes (") +
        Twine(outBuffer->getSize()) + " vs " + Twine(totalSize));
  }
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.split.borrow.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_split_borrow_cpu(
    Chain inChain, StateContext &theContext,
    AsyncValueRef<TensorBufferRef> inBuffer, RemainingArguments runtimeSizes,
    RemainingResults outBuffers, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<ArrayRef<uint64_t>> bOutBufferRawSizes, MEFAttr<uint64_t> cDeviceId,
    Runtime &runtime, EncodedLocation loc) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cDeviceId))
    return std::move(outBuffers).setAllToError("mismatched device id", loc);
  const size_t inSize = inBuffer->getSize();
  size_t totalSize = 0; // total size of all output buffers processed so far
  size_t runtimeSizeIndex = 0; // index of next runtimeSizes operand to use
  SmallVector<TensorBufferRef> newBuffers;
  // We'll have an entry in bOutBufferRawSizes for each output buffer, which
  // will be either the output buffer's compile time size or the distinguished
  // 'no value' value.
  newBuffers.reserve(bOutBufferRawSizes.size());
  for (uint64_t outBufferRawSize : *bOutBufferRawSizes) {
    size_t outBufferRuntimeSize = rawSizeToRuntimeSize(outBufferRawSize);
    size_t thisSize;
    if (hasRuntimeSize(outBufferRuntimeSize)) {
      // The size of this output buffer is known at compile time.
      thisSize = outBufferRuntimeSize;
    } else {
      // The size of this output buffer in unknown at compile time, so
      // must be taken from a runtimeSize operand value.
      if (runtimeSizeIndex > runtimeSizes.size())
        return std::move(outBuffers)
            .setAllToError("mismatched runtimeSizes arity", loc);
      thisSize = runtimeSizes[runtimeSizeIndex].get<size_t>();
      runtimeSizeIndex++;
    }

    // Just stop creating results if it turns out the sizes don't line up, and
    // we'll report failure below. Otherwise, create a 'safe' borrow from
    // inBuffer, with the same mutability. Each borrow will take a reference to
    // inBuffer, thus guaranteeing its lifetime will exceed each borrow.
    if (totalSize + thisSize <= inSize)
      newBuffers.emplace_back(
          createBorrowedSlice(inBuffer.get(), totalSize, thisSize));
    totalSize += thisSize;
  }

  if (totalSize != inBuffer->getSize()) {
    return std::move(outBuffers)
        .setAllToError(Twine("mgp.buffer.split.borrow.cpu input buffer "
                             "size must be sum of output buffer sizes (") +
                           Twine(inBuffer->getSize()) + " vs " +
                           Twine(totalSize) + ")",
                       loc);
  }

  for (auto [outBuffer, newBuffer] : llvm::zip(outBuffers.get(), newBuffers))
    outBuffer = AnyAsyncValueRef::createReady<TensorBufferRef>(
        runtime, std::move(newBuffer));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.section.borrow.static.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_section_borrow_static_cpu(
    Chain inChain, StateContext &theContext,
    AsyncValueRef<TensorBufferRef> inBuffer, Result<TensorBufferRef> outBuffer,
    MEFAttr<uint64_t> aRuntimeSlot, MEFAttr<uint64_t> bOffset,
    MEFAttr<uint64_t> cSize, MEFAttr<uint64_t> dDeviceId, Runtime &runtime,
    EncodedLocation loc) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*dDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  size_t outOffset = static_cast<size_t>(*bOffset);
  size_t outSize = static_cast<size_t>(*cSize);
  size_t outEndOffset = outOffset + outSize; // could overflow
  if (outEndOffset < outOffset || outEndOffset > inBuffer->getSize()) {
    return std::move(outBuffer).setToError(
        Twine("mgp.buffer.section.borrow.static.cpu is "
              "requesting a section of ") +
        Twine(outSize) + " bytes, starting at offset " + Twine(outOffset) +
        ", which is outside the range of the input buffer of " +
        Twine(inBuffer->getSize()) + " bytes");
  }
  std::move(outBuffer).emplace(
      createBorrowedSlice(inBuffer.get(), outOffset, outSize));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.section.borrow.dynamic.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_section_borrow_dynamic_cpu(
    Chain inChain, StateContext &theContext,
    AsyncValueRef<TensorBufferRef> inBuffer, size_t scale,
    Result<TensorBufferRef> outBuffer, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bUnscaledOffset, MEFAttr<uint64_t> cUnscaledSize,
    MEFAttr<uint64_t> dDeviceId, EncodedLocation loc) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*dDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  // Any of the following could overflow.
  const size_t outOffset = static_cast<size_t>(*bUnscaledOffset) * scale;
  const size_t outSize = static_cast<size_t>(*cUnscaledSize) * scale;
  const size_t outEndOffset = outOffset + outSize;
  if (outEndOffset < outOffset || outEndOffset > inBuffer->getSize()) {
    return std::move(outBuffer).setToError(
        Twine("mgp.buffer.section.borrow.dynamic.cpu is "
              "requesting a section of ") +
        Twine(outSize) + " bytes, starting at offset " + Twine(outOffset) +
        ", which is outside the range of the input buffer of " +
        Twine(inBuffer->getSize()) + " bytes");
  }
  std::move(outBuffer).emplace(
      createBorrowedSlice(inBuffer.get(), outOffset, outSize));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.to_bool.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_to_bool_cpu(Chain inChain, StateContext &theContext,
                                   const TensorBufferRef &buffer,
                                   Result<bool> result,
                                   MEFAttr<uint64_t> aRuntimeSlot,
                                   MEFAttr<uint64_t> bDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(result).setToError("mismatched device id");
  if (buffer.getSize() != 1)
    return std::move(result).setToError(
        Twine("expecting buffer to have size 1 (actually is ") +
        Twine(buffer.getSize()) + ")");
  std::move(result).emplace(*buffer.getBuffer<uint8_t>() != 0);
}

//===----------------------------------------------------------------------===//
// mgp.buffer.set_with_index.cpu
//===----------------------------------------------------------------------===//

/// Helper to fill the buffer with the given index values.
template <typename T>
static void fillBuffer(TensorBufferRef &buffer, RemainingArguments idxValues) {
  T *target = buffer.getBuffer<T>();
  for (const auto &asyncIdx : idxValues)
    *(target++) = static_cast<T>(asyncIdx.get<size_t>());
}

static void mgp_buffer_set_with_index_cpu(StateContext &theContext,
                                          TensorBufferRef &buffer,
                                          RemainingArguments values,
                                          Result<Chain> outChain,
                                          MEFAttr<uint64_t> aRuntimeSlot,
                                          MEFAttr<uint64_t> bDeviceId) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outChain).setToError("mismatched device id");

  size_t bufSize = buffer.getSize();
  size_t numArgs = values.size();
  if (bufSize % numArgs != 0) {
    return std::move(outChain).setToError(
        Twine("buffer size (") + Twine(bufSize) +
        ") not divisible by number of index arguments (" + Twine(numArgs) +
        ")");
  }
  size_t elSize = bufSize / numArgs;
  switch (elSize) {
  case 4:
    fillBuffer<int32_t>(buffer, values);
    break;
  case 8:
    fillBuffer<int64_t>(buffer, values);
    break;
  default:
    return std::move(outChain).setToError(
        Twine("unsupported element size for index buffer: ") + Twine(elSize));
  }
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.tensor.create.cpu
//===----------------------------------------------------------------------===//

static void mgp_tensor_create_cpu(Chain inChain, StateContext &theContext,
                                  AsyncValueRef<TensorBufferRef> buffer,
                                  TensorSpec tensorSpec,
                                  Result<Tensor> outTensor,
                                  MEFAttr<uint64_t> aRuntimeSlot,
                                  MEFAttr<uint64_t> bDeviceId, Runtime &runtime,
                                  EncodedLocation loc) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outTensor).setToError("mismatched device id");
  size_t expectedSize = tensorSpec.getSizeInBytes();
  size_t actualSize = buffer->getSize();
  if (expectedSize != actualSize)
    return std::move(outTensor).setToError(
        Twine("buffer size should match spec size (") + Twine(actualSize) +
        " vs " + Twine(expectedSize) + ")");
  std::move(outTensor).emplace(buffer->borrow(), std::move(tensorSpec));
}

//===----------------------------------------------------------------------===//
// mgp.tensor.extract.buffer.cpu
//===----------------------------------------------------------------------===//

static void mgp_tensor_extract_buffer_cpu(StateContext &theContext,
                                          AsyncValueRef<Tensor> tensor,
                                          Result<TensorBufferRef> outBuffer,
                                          MEFAttr<uint64_t> aRuntimeSlot,
                                          MEFAttr<uint64_t> bDeviceId,
                                          Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outBuffer).setToError("mismatched device id");
  std::move(outBuffer).emplace(tensor.get().getBufferRef().borrow());
}

//===----------------------------------------------------------------------===//
// mgp.tensor.extract.tensor_spec.cpu
//===----------------------------------------------------------------------===//

static void mgp_tensor_extract_tensor_spec_cpu(
    StateContext &theContext, AsyncValueRef<Tensor> tensor,
    Result<TensorSpec> outSpec, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bDeviceId, MEFAttr<bool> cIsShapeOnly, Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*bDeviceId))
    return std::move(outSpec).setToError("mismatched device id");
  if (*cIsShapeOnly)
    std::move(outSpec).emplace(tensor->getSpec().getDimsCopy(), DType());
  else
    std::move(outSpec).emplace(tensor->getSpec());
}

//===----------------------------------------------------------------------===//
// mgp.tensor.buffer.print.cpu
//===----------------------------------------------------------------------===//

static void mgp_debug_tensor_print_cpu(
    Chain inChain, StateContext &theContext, const TensorBufferRef &buffer,
    const TensorSpec &spec, Result<Chain> outChain,
    MEFAttr<uint64_t> aRuntimeSlot, MEFAttr<StringRef> bLabel,
    MEFAttr<uint64_t> cDeviceId, Runtime &runtime) {
  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*cDeviceId))
    return std::move(outChain).setToError("mismatched device id");
  if (bLabel->empty()) {
    // Metadata only.
    llvm::outs() << "Tensor(" << spec << ", " << buffer << ")\n";
    llvm::outs().flush();
  } else {
    // Buffer contents, using tensor spec for shape.
    DebugTensorPrintOptions defaultOptions;
    DebugTensorPrintOptions *options =
        runtime.context->get<DebugTensorPrintOptions>();
    if (!options)
      options = &defaultOptions;
    std::string labelWithDevice = (Twine(*bLabel) + "_" + kCPULabel).str();
    if (auto errOr =
            options->printTensor(buffer.getBuffer(), spec, labelWithDevice))
      return std::move(outChain).setToError(errOr.takeError());
  }
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.kernel.execute.cpu
//===----------------------------------------------------------------------===//

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

static void mgp_kernel_execute_cpu(
    Chain inChain, StateContext &theContext, RemainingArguments operands,
    Result<Chain> outChain, RemainingResults outDimensions,
    MEFAttr<uint64_t> aHostRuntimeSlot, MEFAttr<uint64_t> bOtherRuntimeSlot,
    MEFAttr<uint64_t> cHostModuleIndex, MEFAttr<uint64_t> dHostKernelIndex,
    MEFAttr<StringRef> eInOutSignature, MEFAttr<uint64_t> fHostDeviceId,
    MEFAttr<uint64_t> gOtherDeviceId, MEFAttr<StringRef> hSummary,
    MEFAttr<uint64_t> iNumOutDimensions, MEFAttr<bool> jIsAsync,
    Runtime &runtime, EncodedLocation loc) {
  if (*aHostRuntimeSlot != *bOtherRuntimeSlot ||
      *fHostDeviceId != *gOtherDeviceId) {
    return std::move(outChain).setToError(
        "mgp.kernel.execute.cpu requires 'host' and 'other' devices to be the "
        "same");
  }

  CPURuntime &cpuRuntime = theContext.getStateSlotPayload<CPURuntime>(
      static_cast<size_t>(*aHostRuntimeSlot));
  if (cpuRuntime.virtualDeviceId != static_cast<size_t>(*fHostDeviceId))
    return std::move(outChain).setToError("mismatched device id");

  CompiledKernelFn *kernelPtr =
      cpuRuntime.getKernel(static_cast<size_t>(*cHostModuleIndex),
                           static_cast<size_t>(*dHostKernelIndex));

  // The operands are:
  //  [input_1, ..., input_numInputs,
  //   inDimension_1, ..., inDimension_numInDimension]
  StringRef inOutSignature = *eInOutSignature;
  size_t numInputs = inOutSignature.size();
#if MODULAR_DEBUG
  ASSERT_STREAM(numInputs <= operands.size())
      << "invalid inputs signature arity";
#endif
  size_t numInDimensions = operands.size() - numInputs;

  // The outDimensions are:
  //  [outDimension_1, ..., outDimension_numOutDimensions]
  size_t numOutDimensions = static_cast<size_t>(*iNumOutDimensions);
#if MODULAR_DEBUG
  ASSERT_STREAM(numOutDimensions == outDimensions.size())
      << "invalid out dimensions count attribute";
#endif

#if MODULAR_PARANOID
  /// All 'uses' of 'resources' needed by this call which should be considered
  /// active while the call is in flight. This can be used to capture which
  /// of the refs below are for BufferRefs which are being read, written or
  /// modified while the call is active, and thus detect use-after-free and
  /// data races over BufferRefs.
  SmallVector<LLCL::ResourceUse> uses;
#endif

  // Hold on to ABI intermediate datastructures until the kernel returns.
  // Asynchronous kernels are responsible for unmarshalling what they need
  // before returning.
  CompiledKernelABI::CompiledKernelCall call(numInputs);

  // Put the kernel arguments into the ABI void* array. This will mix the
  // input/output buffers, and other types of arguments.
  for (size_t i = 0; i < numInputs; ++i) {
    auto errOr =
        call.pushObject(operands[i].copy(), inOutSignature[i], *hSummary, loc
#if MODULAR_PARANOID
                        ,
                        uses
#endif
        );
    if (errOr) {
      Error error(errOr.takeError());
      std::move(outChain).setToError(error.copy());
      std::move(outDimensions).setAllToError(std::move(error), loc);
      return;
    }
  }

  // Put the runtime dimensions, corresponding to each kernel 'parameter',
  // into the ABI ssize_t array. This will mix the input and output
  // dimensions.
  SmallVector<ssize_t, 16> dimensions;
  dimensions.resize(numInDimensions + numOutDimensions);
  for (size_t i = 0; i < numInDimensions; ++i)
    dimensions[i] = operands[numInputs + i].get<size_t>();
#if MODULAR_PARANOID
  for (size_t i = 0; i < numOutDimensions; ++i)
    dimensions[numInDimensions + i] = -1;
#endif

  // Start profiling the kernel, and mark it as the 'current' event so that
  // it can be picked up as the parent for Mojo trace entries which request it.
  auto profilerEntry = MGPProfilerEntry::create(*hSummary);
  profilerEntry.setAsCurrentId();

  // Execute the kernel.
  if (*jIsAsync) {
    //
    // Asynchronous kernel call.
    //
    // For asynchronous kernels we'll allow the kernel to either set the
    // outChain to an error before suspending, or emplace it / set it to an
    // error when it fully completes. We must make sure the argument
    // MojoCallContext and operands are all held alive until the chain is ready.
    //
#if MODULAR_DEBUG
    ASSERT_STREAM(numOutDimensions == 0)
        << "async kernels cannot return output dimensions";
#endif

    auto launchKernel =
        [operands = operands.copyAsVector(),
         profilerEntry = std::move(profilerEntry), call = std::move(call),
         dimensions = std::move(dimensions), kernelPtr,
         outChain = outChain.copy(), loc = outChain.copyLocation()
#if MODULAR_PARANOID
                                         ,
         uses = std::move(uses)
#endif
    ]() mutable {
          KGEN::MojoCallContext callContext(outChain.copy(), std::move(loc));

          kernelPtr(call.inputPointers.data(), dimensions.data(), &callContext);

          // Stop profiling.
          std::move(profilerEntry).record();

#if MODULAR_PARANOID
          // Release uses of the operands before the operands themselves go out
          // of scope.
          uses.clear();
#endif

          // callContext and operands can now be released.
          operands.clear();
          // Mark the chain as ready.
          if (!outChain.isError())
            std::move(outChain).emplace();
        };

    runtime.getWorkQueue()->addTask(std::move(launchKernel));

#if MODULAR_PARANOID
    // If the kernel is asynchronous but accidentally depends on the stack
    // at the time sub-tasks were launched, deliberately clobber that.
    // Pick a modest size to clobber to avoid stack overflow.
    const size_t s = 256;
    void *p = alloca(s);
    memset(p, 0xef, s);
#endif

  } else {
    //
    // Synchronous kernel call.
    //
    // For synchronous kernels we'll check if the kernel has set the outChain
    // to an error on return, otherwise we'll emplace the outChain to relieve
    // kernels of that responsibility.
    //
    KGEN::MojoCallContext callContext(outChain.copy(), outChain.copyLocation());
    kernelPtr(call.inputPointers.data(), dimensions.data(), &callContext);
    // Stop profiling.
    std::move(profilerEntry).record();
#if MODULAR_PARANOID
    // Release uses to the operands before the operands themselves go out
    // of scope.
    uses.clear();
#endif
    if (outChain.copy().isError()) {
      // The kernel failed while trying to launch.
      std::move(outDimensions)
          .setAllToError(outChain.copy().getDiagnostic().getMessage().copy(),
                         loc);
      return;
    }

    // Any output dimensions can be copied out now.
    for (size_t i = 0; i < numOutDimensions; ++i) {
      outDimensions[i] = AsyncValueRef<size_t>::createReady(
          runtime, dimensions[numInDimensions + i]);
    }

    // We can mark the chain as ready.
#if MODULAR_DEBUG
    ASSERT_STREAM(!outChain.copy().isReady())
        << "synchronous kernels should not emplace the output chain";
#endif
    std::move(outChain).emplace();
  }

  // Make sure our profiling entry (or any which the Mojo kernel may have
  // designated as 'current') does not escape the scope of the Mojo kernel
  // call, either synchronous or up to first await point for asynchronous.
  MGPProfilerEntry::clearCurrentId();
}

//===----------------------------------------------------------------------===//
// mgp.buffer.unsize.cpu
//===----------------------------------------------------------------------===//

// TODO(#30906): Move into MGPPrimitives once mgp.buffer.unsize removed.
static void mgp_buffer_unsize(AnyAsyncValueRef inBuffer,
                              RemainingResults outputs) {
  *outputs.begin() = std::move(inBuffer);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void M::registerMGPCPUPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.runtime.create.cpu", mgp_runtime_create_cpu);
  ADD_PRIMITIVE(registry, "mgp.module.load.cpu", mgp_module_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.chain.create.cpu", mgp_chain_create_cpu);
  ADD_PRIMITIVE(registry, "mgp.chain.device_to_host.cpu", mgp_chain_noop_cpu);
  ADD_PRIMITIVE(registry, "mgp.chain.host_to_device.cpu", mgp_chain_noop_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.cpu", mgp_buffer_constant_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.load.cpu",
                mgp_buffer_constant_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.alloc.static.cpu",
                mgp_buffer_alloc_static_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.alloc.dynamic.cpu",
                mgp_buffer_alloc_dynamic_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.realloc.static.cpu",
                mgp_buffer_realloc_static_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.realloc.dynamic.cpu",
                mgp_buffer_realloc_dynamic_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.alloc_pinned.static.cpu",
                mgp_buffer_alloc_pinned_static_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.alloc_pinned.dynamic.cpu",
                mgp_buffer_alloc_pinned_dynamic_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.cache.cpu", mgp_buffer_cache_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.get_cached.cpu",
                mgp_buffer_get_cached_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.host_to_device.cpu",
                mgp_buffer_no_transfers_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.device_to_host.cpu",
                mgp_buffer_no_transfers_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.device_to_device.cpu",
                mgp_buffer_device_to_device_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.resize.cpu", mgp_buffer_resize_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.concat.cpu", mgp_buffer_concat_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.split.borrow.cpu",
                mgp_buffer_split_borrow_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.section.borrow.static.cpu",
                mgp_buffer_section_borrow_static_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.section.borrow.dynamic.cpu",
                mgp_buffer_section_borrow_dynamic_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.to_bool.cpu", mgp_buffer_to_bool_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.set_with_index.cpu",
                mgp_buffer_set_with_index_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.create.cpu", mgp_tensor_create_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.extract.buffer.cpu",
                mgp_tensor_extract_buffer_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.extract.tensor_spec.cpu",
                mgp_tensor_extract_tensor_spec_cpu);
  ADD_PRIMITIVE(registry, "mgp.debug.tensor.print.cpu",
                mgp_debug_tensor_print_cpu);
  ADD_PRIMITIVE(registry, "mgp.kernel.execute.cpu", mgp_kernel_execute_cpu);

  // TODO(#30906): Move into MGPPrimitives once mgp.buffer.unsize removed.
  ADD_PRIMITIVE(registry, "mgp.buffer.unsize", mgp_buffer_unsize);
}
