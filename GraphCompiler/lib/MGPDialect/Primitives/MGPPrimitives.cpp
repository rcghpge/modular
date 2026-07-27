//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValue.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "GraphCompiler/MGPDialect/Primitives/KernelLogger.h"
#include "GraphCompiler/Support/CompiledKernelsContext.h"
#include "GraphCompiler/Support/List.h"
#include "Init/Init.h"
#include "KGEN/Support/Configuration.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/StateSlot.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/AssertStream.h"
#include "Support/Context.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "Support/Profiling/internal/Tracy.h"
#include "Support/Threading/SpinWaiter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "mgp"

// For convenience this can be set here or in the BUILD.bazel file.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmacro-redefined"
// Change the value below to 1 to log each MGP kernel launch and completion to
// stderr.
#define ENABLE_MGP_KERNEL_LOGGING 0
#pragma clang diagnostic pop

#if ENABLE_MGP_KERNEL_LOGGING
#define MAYBE_LOG_ASYNC_KERNEL(summary, operands, results)                     \
  MGP::KernelAsyncLogger<decltype(operands)> asyncLogger(summary, operands,    \
                                                         results)
#define MAYBE_LOG_SYNC_KERNEL(summary, operands, results)                      \
  MGP::KernelSyncLogger<RemainingArguments> logger(summary, operands, results)
#else
#define MAYBE_LOG_ASYNC_KERNEL(summary, operands, results) ((void)(summary))
#define MAYBE_LOG_SYNC_KERNEL(summary, operands, results) ((void)(summary))
#endif // ENABLE_MGP_KERNEL_LOGGING

using namespace M;
using namespace M::MLRT;
using namespace M::AsyncRT;
using namespace M::GML;
using namespace M::MGP;

/// Indicates that a work item has no device preference for execution.
static constexpr int kNoDevicePreference = -1;

//===----------------------------------------------------------------------===//
// mgp.context.create
//===----------------------------------------------------------------------===//

/// Fulfills the result with a StateContext initialized with `numPrimContexts`
/// primitive contexts.
static void mgp_context_create(Result<StateContext> result,
                               MEFAttr<uint64_t> numSlots) {
  std::move(result).emplace(*numSlots);
}

//===----------------------------------------------------------------------===//
// mgp.generic.execute
//===----------------------------------------------------------------------===//

/// The kernel expects the inputs to be a list of pointers to async values. This
/// function just packs the input / output asyncs into one array.
template <typename OperandTy, typename ResultTy>
llvm::SmallVector<void *> getKernelInputs(OperandTy &operands,
                                          ResultTy &results,
                                          MEFListAttr constantResources) {
  TRACY_ZONE_SCOPED_NC("getKernelInputs", TRACY_COLOR_TEAL);

  llvm::SmallVector<void *> kernelInputs;
  kernelInputs.reserve(results.size() + operands.size() +
                       constantResources.size());

  auto appendInput = [&kernelInputs](const AnyAsyncValueRef &ref) {
    kernelInputs.push_back(
        const_cast<void *>(reinterpret_cast<const void *>(&ref)));
  };

  for (const AnyAsyncValueRef &res : operands)
    appendInput(res);
  for (const AnyAsyncValueRef &res : results)
    appendInput(res);

  for (MEFDynamicAttr attr : constantResources) {
    // Append read-only pointer to attributes.
    kernelInputs.push_back(
        const_cast<void *>(reinterpret_cast<const void *>(attr.get())));

    size_t byteCount = 1;
    if (attr.getAttrKind() == MEF::AttributeKind::kString) {
      byteCount = M::decodeMEFArrayLength(attr.get());
    } else if (((uint8_t)attr.getAttrKind()) &
               ((uint8_t)MEF::AttributeKind::mArrayType)) {
      byteCount = M::decodeMEFArrayLength(attr.get()) *
                  M::MEF::getWidthInBits(attr.getAttrKind()) / 8;
    } else {
      ASSERT_STREAM(attr.getAttrKind() != MEF::AttributeKind::kBool &&
                        attr.getAttrKind() != MEF::AttributeKind::kDictionary &&
                        attr.getAttrKind() != MEF::AttributeKind::kList &&
                        attr.getAttrKind() != MEF::AttributeKind::kType &&
                        attr.getAttrKind() != MEF::AttributeKind::kSymbolRef,
                    "unhandled type of resources.");
    }
    kernelInputs.push_back(reinterpret_cast<void *>(byteCount));
  }

  return kernelInputs;
}

/// Extract device hint from DeviceContext if present in operands.
/// Returns kNoDevicePreference if no single accelerator device is found.
/// Also returns the DeviceContextRef for pinning if exactly one accelerator
/// context is found.
static std::pair<int, M::Driver::DeviceContextRef>
deviceHintAndContextFor(const RemainingArguments &operands) {
  TRACY_ZONE_SCOPED_NC("deviceHintAndContextFor", TRACY_COLOR_TEAL);

  llvm::SmallVector<M::Driver::DeviceContextRef, 8> acceleratorContexts;

  for (const AnyAsyncValueRef &operand : operands) {
    // Check for DeviceContextRef.
    if (operand.isType<M::Driver::DeviceContextRef>()) {
      // getPointerToData() returns a pointer to the payload (DeviceContextRef).
      const auto &deviceCtxRef =
          *static_cast<const M::Driver::DeviceContextRef *>(
              operand.getPointerToData());
      if (deviceCtxRef->device().api() != M::Driver::Device::cpuAPI) {
        // Track only accelerator contexts.
        acceleratorContexts.push_back(
            M::Driver::DeviceContextRef::copy(deviceCtxRef.getPointer()));
      }
    }
  }

  // Return valid hint and context only if there's exactly one accelerator
  // DeviceContextRef.
  if (acceleratorContexts.size() == 1) {
    return {acceleratorContexts[0]->device().id(),
            std::move(acceleratorContexts[0])};
  }
  // Multiple contexts or CPU context -> no preference.
  return {kNoDevicePreference, M::Driver::DeviceContextRef{}};
}

static void
mgp_generic_execute(StateContext &ctx, RemainingArguments operands,
                    RemainingResults outResults, MEFAttr<uint64_t> aSlot,
                    MEFAttr<uint64_t> bModuleIndex,
                    MEFAttr<uint64_t> cFunctionIndex,
                    MEFAttr<StringRef> dSummary, MEFAttr<bool> eIsAsync,
                    MEFListAttr fConstantResources, CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NCT("mgp_generic_execute", TRACY_COLOR_TEAL,
                        "(Summary: " + dSummary.get().str() + ")");

  M::MGP::KernelExecutionState &kernelExecutionState =
      ctx.getStateSlotPayload<M::MGP::KernelExecutionState>(
          static_cast<size_t>(*aSlot));
  CompiledKernelAsyncABIFn *kernelPtr = kernelExecutionState.getKernelAsyncABI(
      static_cast<size_t>(*bModuleIndex), static_cast<size_t>(*cFunctionIndex));

  // Extract device hint and context from DeviceContext if present in operands.
  // deviceCtx is only valid (non-null) when deviceHint != kNoDevicePreference.
  auto [deviceHint, deviceCtx] = deviceHintAndContextFor(operands);

  // Compute taskId for worker affinity via the shared DeviceAffinity logic.
  AsyncRT::WorkQueue *workQueue = cpuDevice.getWorkQueue();
  int taskId =
      (deviceHint == kNoDevicePreference)
          ? kDefaultTaskId
          : MLRT::taskIdForDevice(deviceHint, workQueue->getParallelismLevel());

  // Launch the kernel on a new thread when:
  // - isAsync = true: always run out of line.
  // - Otherwise, sync kernel:
  //   - Kernel has no task affinity => run inline.
  //   - Otherwise, kernel has task affinity:
  //     - Run out of line if we're on the wrong worker for this taskId.
  //     - Otherwise, already on preferred worker: run inline.
  if (eIsAsync.get() || !workQueue->shouldRunInlineForTask(taskId)) {
    TRACY_ZONE_SCOPED_NCT("-> mgp_generic_execute (async)", TRACY_COLOR_TEAL,
                          "(Summary: " + dSummary.get().str() + ")");

    // Allocate indirect results as GraphRT is expecting result async values to
    // be allocated. The kernel will set them to ready when it is finished. We
    // only need to do this for async primitives as they outlive this function.
    outResults.allocateIndirectResults();

    // Run the kernel and extend the lifetime of the input and output async
    // values until it completes by capturing them.
    // Also capture the device context for pinning optimization when valid.
    auto runKernelAndExtendLifetimes =
        [outResults = outResults.copyAsVector(),
         takenOperands = operands.takeAsVector(),
         constantResources = std::move(fConstantResources), kernelPtr, dSummary,
         deviceCtx = std::move(deviceCtx)]() mutable {
          TRACY_ZONE_SCOPED_NCT("runKernelAndExtendLifetimes", TRACY_COLOR_TEAL,
                                "(Summary: " + dSummary.get().str() + ")");

          // The kernel writes its results into these closure-local slots; they
          // are published into `outResults` as the task's last action.
          // This is to ensure no downstream work starts before the full
          // completion of `kernelPtr`.
          llvm::SmallVector<AnyAsyncValueRef> localResults(outResults.size());

          llvm::SmallVector<void *> kernelInputs =
              getKernelInputs(takenOperands, localResults, constantResources);
          MAYBE_LOG_ASYNC_KERNEL(*dSummary, takenOperands, localResults);

          // Bind the device context to this thread for the duration of kernel
          // execution.
          if (deviceCtx.getPointer() != nullptr)
            (void)deviceCtx->bindToCurrentThread();

          {
            TRACY_ZONE_SCOPED_NCT("Invoke Mojo function", TRACY_COLOR_ORANGE,
                                  "(Summary: " + dSummary.get().str() + ")");
            kernelPtr(reinterpret_cast<void **>(kernelInputs.data()));
          }

          if (deviceCtx.getPointer() != nullptr)
            M::Driver::DeviceContext::unbindFromCurrentThread();

          // Must have all been released/reset by the compiled kernel.
          assert(llvm::all_of(takenOperands, [](const AnyAsyncValueRef &ref) {
            return !ref.getPointer();
          }));

          // Release everything context-reachable BEFORE the driver can observe
          // completion: operand slots (already reset by the kernel) and the
          // pinned device context reference.
          takenOperands.clear();
          deviceCtx = M::Driver::DeviceContextRef();

          // Publish the kernel's results by resolving the pre-allocated
          // indirect result values.
          for (const auto &[local, output] :
               llvm::zip_equal(localResults, outResults)) {
            assert(local && "compiled kernel did not fulfill a result slot");
            std::move(output).resolveIndirect(std::move(local));
          }
        };

    // Launch the kernel on a new thread with worker affinity.
    AsyncRT::WorkItem workItem(std::move(runKernelAndExtendLifetimes));
    workQueue->addTask(std::move(workItem), taskId);
  } else {
    TRACY_ZONE_SCOPED_NCT("-> mgp_generic_execute (sync kernel)",
                          TRACY_COLOR_TEAL,
                          "(Summary: " + dSummary.get().str() + ")");

    // Bind the device context to this thread for the duration of kernel
    // execution.
    if (deviceCtx.getPointer() != nullptr)
      (void)deviceCtx->bindToCurrentThread();

    // Run the kernel on this thread.
    llvm::SmallVector<void *> kernelInputs =
        getKernelInputs(operands, outResults, fConstantResources);
    MAYBE_LOG_SYNC_KERNEL(*dSummary, operands, outResults.get());
    {
      TRACY_ZONE_SCOPED_NCT("Invoke Mojo function", TRACY_COLOR_ORANGE,
                            "(Summary: " + dSummary.get().str() + ")");

      kernelPtr(reinterpret_cast<void **>(kernelInputs.data()));
    }

    if (deviceCtx.getPointer() != nullptr)
      M::Driver::DeviceContext::unbindFromCurrentThread();

    // Must have all been released/reset by the compiled kernel.
    assert(llvm::all_of(operands, [](const AnyAsyncValueRef &ref) {
      return !ref.getPointer();
    }));
  }
}

//===----------------------------------------------------------------------===//
// mgp.device_graph.execute
//===----------------------------------------------------------------------===//

// Replays a constructed device graph and surfaces its recorded outputs as the
// op's results. Runtime counterpart of `mgp.device_graph.create` (which builds
// the graph via the kernel path). Synchronous first cut: replay inline, then
// forward each stored output `AnyAsyncValueRef` into the result slots.
static void mgp_device_graph_execute(M::Driver::DeviceGraphRef &graph,
                                     RemainingArguments deviceContextAndInputs,
                                     Result<Chain> outChain,
                                     RemainingResults outputs,
                                     EncodedLocation loc) {
  TRACY_ZONE_SCOPED_NC("mgp_device_graph_execute", TRACY_COLOR_TEAL);

  // `graph` is operand 0 (packed as a `DeviceGraphRef` by the create region's
  // pack helper). The device context and input operands land in
  // `deviceContextAndInputs`; they are consumed only for async dependency
  // ordering. In the stateless first cut the build baked the live input
  // addresses into the graph, so replay needs no input copy.
  (void)deviceContextAndInputs;

  if (ErrorOrSuccess err = graph->replay()) {
    Error msg = err.takeError();
    std::move(outChain).setToError(msg.copy());
    return std::move(outputs).setAllToError(std::move(msg), loc);
  }

  // The recorded outputs (registered via `add_output` during the build and
  // transferred into the graph by `instantiate`) become the op's value results
  // (device graphs carry no chain, so `getOutputs()` is chain-free). Share the
  // underlying `Tensor`/`TensorBufferRef` via `copy()` -- no deep copy.
  //
  // GraphRT hands us pre-allocated (indirect) result async values that
  // downstream consumers already reference, so we must resolve those in place
  // to the graph's outputs rather than replacing the slot references (which
  // would leave the referenced async values unfulfilled). Mirrors the
  // control-flow primitives and `mgp_generic_execute`.
  ArrayRef<AnyAsyncValueRef> stored = graph->getOutputs();
  assert(stored.size() == outputs.size() &&
         "device graph output count must match execute result count");
  outputs.allocateIndirectResults();

  for (const auto &[output, value] : llvm::zip_equal(outputs, stored))
    output.copy().resolveIndirect(value.copy());

  // Fresh ordering chain: replay finished, so host-side consumers ordered after
  // this op may proceed.
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.list.append
//===----------------------------------------------------------------------===//

static void mgp_list_append(List &inputList, AnyAsyncValueRef element,
                            Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_append", TRACY_COLOR_TEAL);

  if (element.isType<Tensor>()) {
    if (!inputList.getElements().empty() &&
        inputList.getElements().front().getValue().getPointer()->getTypeID() !=
            element.getPointer()->getTypeID()) {
      std::move(output).setToError("Tensor must be from the same device");
      return;
    }
  }

  // Copy input list then append element into it.
  List newList = inputList.copy();
  newList.getElements().push_back(element.copy());
  std::move(output).emplace(std::move(newList));
}

//===----------------------------------------------------------------------===//
// mgp.list.concat
//===----------------------------------------------------------------------===//

static void mgp_list_concat(List &lhs, List &rhs, Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_concat", TRACY_COLOR_TEAL);

  List resultList = lhs.copy();
  resultList.getElements().insert(resultList.getElements().end(),
                                  rhs.getElements().begin(),
                                  rhs.getElements().end());

  std::move(output).emplace(std::move(resultList));
}

//===----------------------------------------------------------------------===//
// mgp.list.create
//===----------------------------------------------------------------------===//

static void mgp_list_create(RemainingArguments operands,
                            Result<List> outputList) {
  TRACY_ZONE_SCOPED_NC("mgp_list_create", TRACY_COLOR_TEAL);

  std::move(outputList).emplace(List{operands.copyAsVector()});
}

//===----------------------------------------------------------------------===//
// mgp.list.get
//===----------------------------------------------------------------------===//

static void mgp_list_get(List &inputList, Tensor &idxTensor,
                         RemainingResults outputs, EncodedLocation loc) {
  TRACY_ZONE_SCOPED_NC("mgp_list_get", TRACY_COLOR_TEAL);

  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        // Check index is in bound.
        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize <= idx) {
          return std::move(outputs).setAllToError(
              Twine("index ") + Twine(idx) +
                  " is out of bound for list of size " + Twine(listSize),
              loc);
        }

        // Normalize index.
        if (idx < 0)
          idx += listSize;

        // We must set the output this way because
        // (1). list can contain arbitrary types of values, so we can't put a
        // single type into `Result<...>`> (2). we want to share the
        // underlying value as `AsyncValueRef` to avoid deep copying expensive
        // objects like `Tensor`.
        *outputs.begin() = inputList.getElements()[idx].getValue().copy();
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.insert
//===----------------------------------------------------------------------===//

static void mgp_list_insert(List &inputList, AnyAsyncValueRef element,
                            Tensor &idxTensor, Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_insert", TRACY_COLOR_TEAL);

  // We must unbox the index value from the tensor.
  // TODO: Should be ssize_t.
  idxTensor.getEltType()
      .dispatch<void>(idxTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *idxDataPtr) {
        auto idx = *idxDataPtr;

        // Check index is in bound.
        int64_t listSize = inputList.getElements().size();
        if (idx < -listSize || listSize < idx) {
          return std::move(output).setToError(
              Twine("index ") + Twine(idx) +
              " is out of bound for list of size " + Twine(listSize));
        }

        // Normalize index.
        if (idx < 0)
          idx += listSize;

        // Copy input list then insert element into it.
        List newList = inputList.copy();
        newList.getElements().insert(newList.getElements().begin() + idx,
                                     element.copy());
        std::move(output).emplace(std::move(newList));
      });
}

//===----------------------------------------------------------------------===//
// mgp.list.slice
//===----------------------------------------------------------------------===//

static void mgp_list_slice(List &inputList, Tensor &startTensor,
                           Tensor &endTensor, Tensor &stepTensor,
                           Result<List> output) {
  TRACY_ZONE_SCOPED_NC("mgp_list_slice", TRACY_COLOR_TEAL);

  auto sliceFunctor = [&](auto start, auto end, auto step) {
    List resultList;

    auto &elems = inputList.getElements();

    // start/end/step
    ssize_t length = elems.size();

    if (step == 0) {
      return std::move(output).setToError(
          "The slice step argument must be a non-zero");
    }

    const bool stepIsPositive = step > 0;

    // normalize negative start and end values
    if (start < 0) {
      start += length;
    }
    if (end < 0) {
      end += length;
    }

    if (stepIsPositive) {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, 0, length);
    } else {
      start = std::clamp<decltype(start)>(start, 0, length - 1);
      end = std::clamp<decltype(end)>(end, -1, length);
    }

    if (stepIsPositive) {
      for (ssize_t i = start; i < end && i < length && i >= 0; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    } else {
      for (ssize_t i = start; i > end; i += step) {
        resultList.getElements().push_back(inputList.getElements()[i]);
      }
    }

    std::move(output).emplace(std::move(resultList));
  };

  startTensor.getEltType()
      .dispatch<void>(startTensor.getBuffer())
      .template when<DType::si32, DType::si64>([&](auto *startDataPtr) {
        auto start = *startDataPtr;

        endTensor.getEltType()
            .dispatch<void>(endTensor.getBuffer())
            .template when<DType::si32, DType::si64>([&](auto *endDataPtr) {
              auto end = *endDataPtr;

              stepTensor.getEltType()
                  .dispatch<void>(stepTensor.getBuffer())
                  .template when<DType::si32, DType::si64>(
                      [&](auto *stepDataPtr) {
                        auto step = *stepDataPtr;
                        sliceFunctor(start, end, step);
                      });
            });
      });
}

//===----------------------------------------------------------------------===//
// mgp.guard
//===----------------------------------------------------------------------===//

static void mgp_guard(Chain inChain, RemainingArguments inputs,
                      RemainingResults results) {
  TRACY_ZONE_SCOPED_NC("mgp_guard", TRACY_COLOR_TEAL);

  assert(inputs.size() == results.size());

  auto inputAsyncVals = inputs.get();
  auto resultAsyncVals = results.get();

  for (const auto &[input, result] :
       llvm::zip(inputAsyncVals, resultAsyncVals)) {
    result = std::move(input);
  }
}

//===----------------------------------------------------------------------===//
// mgp.model
//===----------------------------------------------------------------------===//

static void mgp_model() {
  TRACY_ZONE_SCOPED_NC("mgp_model", TRACY_COLOR_TEAL);

  ASSERT_STREAM(false, << "mgp.model should never be invoked");
}

//===----------------------------------------------------------------------===//
// mgp.extend.lifetime
//===----------------------------------------------------------------------===//

// No-op, will be scheduled once all are ready (preventing the inputs from being
// freed).
static void mgp_nop(RemainingArguments operands, Result<Chain> outChain) {
  TRACY_ZONE_SCOPED_NC("mgp_nop", TRACY_COLOR_TEAL);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.runtime.create
//===----------------------------------------------------------------------===//

void mgp_runtime_create(Chain inChain, StateContext &theContext,
                        Result<Chain> outChain, MEFAttr<uint64_t> aRuntimeSlot,
                        CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NC("mgp_runtime_create", TRACY_COLOR_TEAL);

  // Primitives are invoked by the MEF executor with a fixed frame (chain,
  // state, cpuDevice, etc.); Context is not in the invocation ABI. Callers
  // (engine, mt, mo-opt) create the M::Context, and this is then retrieved
  // here by calling Init::getContext().
  M::ContextRef ctxRef = Init::getContext();

  // Establish an execution engine, to be shared amongst all loaded models
  // across all models using this cpuDevice.
  //
  // TODO(#30647): Currently there's no garbage collection of dylibs loaded into
  //               this engine.
  ErrorOr<KGEN::ExecutionEngine *> engineOr =
      ctxRef->createIfMissing<KGEN::ExecutionEngine>(
          [&]() -> ErrorOr<std::unique_ptr<KGEN::ExecutionEngine>> {
            // Initialize the host target.
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmParser();
            llvm::InitializeNativeTargetAsmPrinter();
            auto targetMachineOr = getTargetMachineForHost();
            if (targetMachineOr.isError())
              return targetMachineOr.takeError();

            auto cfgOr = KGEN::MojoConfig::open();
            if (cfgOr.isError())
              return cfgOr.takeError();
            KGEN::MojoConfig cfg = std::move(*cfgOr);

            // Create the execution engine.
            KGEN::ExecutionEngineOptions options;
            options.registerPerfPlugins =
                cpuDevice.getProfilerDebuginfo() ==
                M::AsyncRT::CPUDeviceOptions::ProfilerDebuginfo::kPerfProfiler;
            options.libraryPaths.push_back(cfg.getMGPRTPath());
            return KGEN::ExecutionEngine::createWithStandardLayers(
                std::move(options), **targetMachineOr);
          });

  if (engineOr.isError())
    return std::move(outChain).setToError(engineOr.takeError());

  theContext.emplaceStateSlot<KernelExecutionState>(
      static_cast<size_t>(*aRuntimeSlot), *engineOr);

  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// mgp.device_context.select_stream
//===----------------------------------------------------------------------===//

static ErrorOr<M::Driver::DeviceContextRef>
mgp_device_context_select_stream(M::Driver::DeviceContextRef &deviceContext,
                                 MEFAttr<uint64_t> streamId) {
  TRACY_ZONE_SCOPED_NC("mgp_device_context_select_stream", TRACY_COLOR_TEAL);
  return deviceContext->selectStream(static_cast<unsigned>(*streamId));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.cache
//===----------------------------------------------------------------------===//

static void mgp_buffer_cache(Chain inChain, StateContext &theContext,
                             AsyncValueRef<TensorBufferRef> buffer,
                             Result<Chain> outChain,
                             MEFAttr<uint64_t> bBufferSlot,
                             CPUDevice &cpuDevice) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_cache", TRACY_COLOR_TEAL);

  auto &tensorStateSlot = theContext.emplaceStateSlot<TensorStateSlot>(
      static_cast<size_t>(*bBufferSlot));

  // Always borrow the buffer.
  ErrorOrSuccess errOr = tensorStateSlot.setBuffer(buffer.get().borrow());

  if (errOr.isError())
    return std::move(outChain).setToError(errOr.takeError());
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.runtime.create", mgp_runtime_create);
  ADD_PRIMITIVE(registry, "mgp.device_context.select_stream",
                mgp_device_context_select_stream);
  ADD_PRIMITIVE(registry, "mgp.buffer.cache", mgp_buffer_cache);
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
  ADD_PRIMITIVE(registry, "mgp.device_graph.execute", mgp_device_graph_execute);
  ADD_PRIMITIVE(registry, "mgp.list.append", mgp_list_append);
  ADD_PRIMITIVE(registry, "mgp.guard", mgp_guard);
  ADD_PRIMITIVE(registry, "mgp.list.concat", mgp_list_concat);
  ADD_PRIMITIVE(registry, "mgp.list.create", mgp_list_create);
  ADD_PRIMITIVE(registry, "mgp.list.get", mgp_list_get);
  ADD_PRIMITIVE(registry, "mgp.list.insert", mgp_list_insert);
  ADD_PRIMITIVE(registry, "mgp.list.slice", mgp_list_slice);
  ADD_PRIMITIVE(registry, "mgp.model", mgp_model);

  ADD_PRIMITIVE(registry, "mgp.extend.lifetime", mgp_nop);

  registerMGPCPUPrimitives(registry);
}
