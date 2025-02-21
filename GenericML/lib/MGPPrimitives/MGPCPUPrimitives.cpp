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
// mgp.tensor.buffer.print.cpu
//===----------------------------------------------------------------------===//

static void
mgp_debug_tensor_print_cpu(Chain inChain, const TensorBufferRef &buffer,
                           const TensorSpec &spec, Result<Chain> outChain,
                           MEFAttr<StringRef> bLabel,
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
    if (auto errOr = options->printTensor(buffer.getBuffer(), spec, *bLabel))
      return std::move(outChain).setToError(errOr.takeError());
  }
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void M::registerMGPCPUPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.module.load.cpu", mgp_module_load_cpu);
  ADD_PRIMITIVE(registry, "mgp.debug.tensor.print.cpu",
                mgp_debug_tensor_print_cpu);
}
