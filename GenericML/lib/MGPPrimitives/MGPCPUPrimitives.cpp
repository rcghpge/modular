//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/Chain.h"
#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/DebugPrint.h"
#include "GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Runtime/Tensor/StateContext.h"
#include "Runtime/Tensor/Tensor.h"
#include "Runtime/Tensor/TensorBufferRef.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorSpec.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mgp"

using namespace M;
using namespace M::AsyncRT;
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
                            MEFAttr<uint64_t> bDeviceId, Runtime &runtime) {
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

  theContext.emplaceStateSlot<KernelExecutionState>(
      static_cast<size_t>(*aRuntimeSlot),
      /*numBuffers=*/0, *engineOr);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.module.load.cpu
//===----------------------------------------------------------------------===//

static void
mgp_module_load_cpu(Chain inChain, StateContext &theContext,
                    Result<Chain> outChain, MEFAttr<uint64_t> aRuntimeSlot,
                    MEFDynamicAttr bImage, MEFAttr<StringRef> cFunctionSymbols,
                    MEFAttr<StringRef> dConstantSymbols,
                    MEFAttr<uint64_t> eDeviceId, MEFAttr<uint64_t> fModuleIndex,
                    MEFAttr<bool> debuginfo, Runtime &runtime) {
  KernelExecutionState &kernelExecutionState =
      theContext.getStateSlotPayload<KernelExecutionState>(
          static_cast<size_t>(*aRuntimeSlot));
  if (!dConstantSymbols->empty() && !(dConstantSymbols->size() == 0) &&
      !(dConstantSymbols->size() == 1 && *dConstantSymbols->begin() == '\0')) {
    return std::move(outChain).setToError(
        "mgp.module.load.cpu does not support constants");
  }

  // Create a kernels context.
  if (fModuleIndex.get() >= kernelExecutionState.modules.size())
    kernelExecutionState.modules.resize(fModuleIndex.get() + 1);
  kernelExecutionState.modules[fModuleIndex.get()] =
      RCRef<CompiledKernelsContext>::create(
          kernelExecutionState.engine,
          debuginfo.get() ||
              runtime.getProfilerDebuginfo() ==
                  RuntimeOptions::ProfilerDebuginfo::kSOProfiler);
  CompiledKernelsContext &module =
      *kernelExecutionState.modules[fModuleIndex.get()];

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
      assert(start + 1 == end && combined[start] == '\0');
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
// mgp.buffer.constant.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_constant_cpu(Chain inChain, StateContext &theContext,
                                    Result<TensorBufferRef> outBuffer,
                                    MEFAttr<uint64_t> aRuntimeSlot,
                                    MEFAttr<ArrayRef<uint8_t>> bValue,
                                    MEFAttr<uint64_t> cRawAlign,
                                    MEFAttr<uint64_t> dDeviceId,
                                    MEFAttr<bool> eIsTrivial) {
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
// mgp.buffer.cache.cpu
//===----------------------------------------------------------------------===//

static void mgp_buffer_cache_cpu(Chain inChain, StateContext &theContext,
                                 AsyncValueRef<TensorBufferRef> buffer,
                                 Result<Chain> outChain,
                                 MEFAttr<uint64_t> aRuntimeSlot,
                                 MEFAttr<uint64_t> bBufferSlot,
                                 MEFAttr<uint64_t> cDeviceId,
                                 Runtime &runtime) {
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
// mgp.tensor.create.cpu
//===----------------------------------------------------------------------===//

static void mgp_tensor_create_cpu(Chain inChain, StateContext &theContext,
                                  AsyncValueRef<TensorBufferRef> buffer,
                                  TensorSpec tensorSpec,
                                  Result<Tensor> outTensor,
                                  MEFAttr<uint64_t> aRuntimeSlot,
                                  MEFAttr<uint64_t> bDeviceId, Runtime &runtime,
                                  EncodedLocation loc) {
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
  std::move(outBuffer).emplace(tensor.get().getBufferRef().borrow());
}

//===----------------------------------------------------------------------===//
// mgp.tensor.extract.tensor_spec.cpu
//===----------------------------------------------------------------------===//

static void mgp_tensor_extract_tensor_spec_cpu(
    StateContext &theContext, AsyncValueRef<Tensor> tensor,
    Result<TensorSpec> outSpec, MEFAttr<uint64_t> aRuntimeSlot,
    MEFAttr<uint64_t> bDeviceId, MEFAttr<bool> cIsShapeOnly, Runtime &runtime) {
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

/// Profiler entry for launched CPU kernels.
using MGPProfilerEntry =
    ProfilerEntry<Trace::EnableTrace(Trace::kOther, 1), Trace::kOther>;

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
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.cpu", mgp_buffer_constant_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.load.cpu",
                mgp_buffer_constant_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.cache.cpu", mgp_buffer_cache_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.get_cached.cpu",
                mgp_buffer_get_cached_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.resize.cpu", mgp_buffer_resize_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.concat.cpu", mgp_buffer_concat_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.split.borrow.cpu",
                mgp_buffer_split_borrow_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.create.cpu", mgp_tensor_create_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.extract.buffer.cpu",
                mgp_tensor_extract_buffer_cpu);
  ADD_PRIMITIVE(registry, "mgp.tensor.extract.tensor_spec.cpu",
                mgp_tensor_extract_tensor_spec_cpu);
  ADD_PRIMITIVE(registry, "mgp.debug.tensor.print.cpu",
                mgp_debug_tensor_print_cpu);

  // TODO(#30906): Move into MGPPrimitives once mgp.buffer.unsize removed.
  ADD_PRIMITIVE(registry, "mgp.buffer.unsize", mgp_buffer_unsize);
}
