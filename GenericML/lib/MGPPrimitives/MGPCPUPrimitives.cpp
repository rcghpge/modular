//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "MLRT/AsyncRT/Support/Chain.h"
#include "MLRT/Core/MojoValue.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorSpec.h"
#include "mojo_string_c_api.h"
#include "weights_registry_c_api.h"
#include "llvm/ADT/bit.h"
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

TensorBufferRef createBorrowedSlice(TensorBufferRef &borroweBufferRef,
                                    size_t offset, size_t size) {
  const void *newBuffer = borroweBufferRef.getBuffer<uint8_t>() + offset;
  return TensorBufferRef::create(const_cast<void *>(newBuffer), size,
                                 borroweBufferRef.getMemStorageHandle());
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
// mgp.buffer.constant.external.cpu
//===----------------------------------------------------------------------===//

// Note: This primitive is temporarily implemented in C++ to enable subgraph
// inlining after JitCompileKernels. GEX-2312 tracks the future work regarding
// this primitive.
static void mgp_buffer_constant_external_cpu(
    Chain inChain, AsyncValueRef<MojoValue> weights,
    Result<TensorBufferRef> outBuffer, MEFAttr<StringRef> bName,
    MEFAttr<uint64_t> cSize, MEFAttr<uint64_t> dAlign,
    MEFAttr<uint64_t> eDeviceId, MEFAttr<bool> gHasAlias,
    MEFAttr<bool> hIsPlaceholder) {

  if (dAlign.get() == 0) {
    return std::move(outBuffer).setToError(
        "align must be a positive integer value");
  }

  // Look up the weight by name in the registry using the C API directly
  void *data;
  static constexpr int kResultSuccess = 0;

  // Copy into SmallString to guarantee null termination
  SmallString<128> nameStr(bName.get());

  if (weights_registry_lookup(&data, weights->getData(),
                              const_cast<char *>(nameStr.c_str())) !=
      kResultSuccess) {
    return std::move(outBuffer).setToError(Twine("unable to look up weight: ") +
                                           bName.get());
  }

  const std::byte *weightPtr = llvm::bit_cast<const std::byte *>(data);

  // Validate alignment
  if ((reinterpret_cast<uintptr_t>(weightPtr) % dAlign.get()) != 0) {
    return std::move(outBuffer).setToError(
        Twine("invalid alignment for address ") +
        Twine(reinterpret_cast<uintptr_t>(weightPtr)) + " and align " +
        Twine(dAlign.get()));
  }

  // Create a TensorBufferRef from the weight data
  // Note: The weights registry owns the data, so we create a non-tracked buffer
  std::move(outBuffer).emplace(TensorBufferRef::createWithNonTrackedMemory(
      const_cast<void *>(reinterpret_cast<const void *>(weightPtr)),
      cSize.get(),
      /*alignment=*/dAlign.get()));
}

//===----------------------------------------------------------------------===//
// mgp.string.create
//===----------------------------------------------------------------------===//

static void mgp_string_create(Result<MojoValue> outString,
                              MEFAttr<StringRef> value) {
  StringRef strValue = value.get();

  // Create a Mojo String via the C API
  void *mojoStringPtr =
      create_mojo_string(const_cast<char *>(strValue.data()),
                         static_cast<ssize_t>(strValue.size()));

  // Create MojoValue that owns the Mojo UnsafePointer[String]
  std::move(outString).emplace(MojoValue::createOwningRef(
      mojoStringPtr,
      reinterpret_cast<MojoValue::DestructorFn>(destroy_mojo_string)));
}

//===----------------------------------------------------------------------===//
// mgp.string.concat
//===----------------------------------------------------------------------===//

static void mgp_string_concat(AsyncValueRef<MojoValue> prefix,
                              AsyncValueRef<MojoValue> end,
                              Result<MojoValue> outString) {
  // Concatenate the two Mojo Strings via the C API
  void *mojoStringPtr = concat_mojo_strings(prefix->getData(), end->getData());

  // Create MojoValue that owns the new Mojo UnsafePointer[String]
  std::move(outString).emplace(MojoValue::createOwningRef(
      mojoStringPtr,
      reinterpret_cast<MojoValue::DestructorFn>(destroy_mojo_string)));
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void M::registerMGPCPUPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.module.load.cpu", mgp_module_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.external.cpu",
                mgp_buffer_constant_external_cpu);
  ADD_PRIMITIVE(registry, "mgp.buffer.constant.external.gpu",
                mgp_buffer_constant_external_cpu);
  ADD_PRIMITIVE(registry, "mgp.string.create", mgp_string_create);
  ADD_PRIMITIVE(registry, "mgp.string.concat", mgp_string_concat);
}
