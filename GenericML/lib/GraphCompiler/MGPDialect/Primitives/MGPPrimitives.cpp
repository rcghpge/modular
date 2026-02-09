//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MGPDialect/Primitives/MGPPrimitives.h"
#include "GenericML/GraphCompiler/MGPDialect/Primitives/KernelLogger.h"
#include "GenericML/Support/CompiledKernelsContext.h"
#include "GenericML/Support/List.h"
#include "KGEN/Support/Configuration.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"
#include "MLRT/AsyncRT/Runtime/AsyncValue.h"
#include "MLRT/AsyncRT/Runtime/AsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Core/Tensor/StateContext.h"
#include "MLRT/Core/Tensor/StateSlot.h"
#include "MLRT/Core/Tensor/Tensor.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"
#include "Support/AssertStream.h"
#include "Support/Configuration.h"
#include "Support/DeviceSpecs.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/SizeUtils.h"
#include "Support/ML/TensorShape.h"
#include "Support/Profiling/Tracy.h"
#include "Support/Threading/HWInfo.h"
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
#include <algorithm>
#include <cstdlib>

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

static std::map<int, int> inferGpuToCpuCoreMapping() {
  // Retrieve the NUMA topology.
  auto numaTopologyOrError = NUMATopology::get();
  if (numaTopologyOrError.isError())
    return {};
  const NUMATopology &numaTopology = *numaTopologyOrError;

  // Retrieve a list of device IDs.
  int numDevices =
      M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);

  // Iterate over device IDs and infer a mapping of each to a CPU core.
  std::map<int, int> gpuToCpuCoreMapping;
  std::map<int, size_t> nextCpuCoreIndexPerNuma;

  for (int i = 0; i < numDevices; i++) {
    auto deviceOrError =
        M::Driver::Device::create(M::Driver::Device::gpuAPI, i);
    if (deviceOrError.isError())
      break;
    auto &device = *deviceOrError;

    auto pciBusIdOrError = device->getPciBusId();
    if (pciBusIdOrError.isError())
      break;

    int numaNode = numaTopology.getNumaNodeForPciBus(*pciBusIdOrError);
    if (numaNode == -1)
      break;

    std::vector<size_t> cpuIds = numaTopology.getCpuIdsForNumaNode(numaNode);
    if (cpuIds.empty())
      break;

    size_t cpuCoreIndex =
        (1 + nextCpuCoreIndexPerNuma[numaNode]++ % (cpuIds.size() - 1));

    gpuToCpuCoreMapping[i] = cpuIds[cpuCoreIndex];
  }
  return gpuToCpuCoreMapping;
}

static std::map<int, int> getGpuToCpuCoreMapping() {
  TRACY_ZONE_SCOPED_NC("getGpuToCpuCoreMapping", TRACY_COLOR_TEAL);

  static std::map<int, int> mapping = inferGpuToCpuCoreMapping();
  return mapping;
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

/// Compute the taskId for a device hint.
/// Returns kDefaultTaskId if no affinity is needed.
/// Always uses workers 1..N-1 for device affinity to avoid potential stalls
/// when worker 0 is the main/donating thread.
static int taskIdForDevice(int deviceHint,
                           M::Driver::DeviceContextRef deviceCtx,
                           size_t numWorkers) {
  if (deviceHint == kNoDevicePreference)
    return kDefaultTaskId;

  assert(deviceHint >= 0 &&
         "expected kNoDevicePreference or non-negative device hint");

  if (numWorkers <= 1)
    return kDefaultTaskId;

  // Initialize only once as Config::open() is expensive.
  static SmallVector<int, 8> deviceTaskCpuIds = []() {
    // If the user explicitly specified the cpu ID they want assigned to
    // this device, then go with that. For example, if the system has 32
    // CPUs per NUMA node and the user wants all 8 GPUs controlled from NUMA
    // node 1, then the user would set
    // MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS="32,33,34,35,36,37,38,39".
    SmallVector<int, 8> deviceTaskCpuIds;
    auto config = Config::open();
    if (!config.isError()) {
      SmallVector<StringRef, 8> deviceTaskCpuIdStrs;
      config->getValueAsList("runtime.device_task_cpu_ids",
                             deviceTaskCpuIdStrs);
      for (size_t i = 0; i < deviceTaskCpuIdStrs.size(); i++) {
        int cpuId;
        if (deviceTaskCpuIdStrs[i].getAsInteger(10, cpuId)) {
          llvm::errs() << "Failed to parse runtime.device_task_cpu_ids[" << i
                       << "] as an integer: " << deviceTaskCpuIdStrs[i] << "\n";
          std::abort();
        }
        if (cpuId == 0) {
          llvm::errs()
              << "CPU ID 0 is not allowed in runtime.device_task_cpu_ids\n";
          std::abort();
        } else if (std::find(deviceTaskCpuIds.begin(), deviceTaskCpuIds.end(),
                             cpuId) != deviceTaskCpuIds.end()) {
          llvm::errs()
              << "Duplicate CPU ID found in runtime.device_task_cpu_ids: "
              << cpuId << "\n";
          std::abort();
        } else {
          deviceTaskCpuIds.push_back(cpuId);
        }
      }
    }
    return deviceTaskCpuIds;
  }();

  if (deviceTaskCpuIds.size() > static_cast<size_t>(deviceHint))
    return deviceTaskCpuIds[deviceHint];

  // Infer the task ID from the GPU to CPU core mapping, based on NUMA topology
  // affinity, if it is available for the target device.
  // Only use the mapping if numWorkers matches the number of physical CPU
  // cores, which indicates the thread pool wasn't limited by cgroups or other
  // means.
  if (deviceCtx.getPointer() != nullptr &&
      numWorkers >= M::getNumPhysicalCores()) {
    auto gpuToCpuCoreMapping = getGpuToCpuCoreMapping();
    if (!gpuToCpuCoreMapping.empty()) {
      int deviceId = deviceCtx->device().id();
      auto it = gpuToCpuCoreMapping.find(deviceId);
      if (it != gpuToCpuCoreMapping.end()) {
        return it->second;
      }
    }
  }

  // Conservative: always skip worker 0 for affinity tasks.
  // This avoids stalls regardless of mainWillDonate setting.
  return 1 + (deviceHint % static_cast<int>(numWorkers - 1));
}

static void mgp_generic_execute(
    StateContext &ctx, RemainingArguments operands, RemainingResults outResults,
    MEFAttr<uint64_t> aSlot, MEFAttr<uint64_t> bModuleIndex,
    MEFAttr<uint64_t> cFunctionIndex, MEFAttr<StringRef> dSummary,
    MEFAttr<bool> eIsAsync, MEFListAttr fConstantResources, Runtime &runtime) {
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

  // Compute taskId for worker affinity. Uses conservative worker 0 avoidance.
  AsyncRT::WorkQueue *workQueue = runtime.getWorkQueue();
  int taskId =
      taskIdForDevice(deviceHint, deviceCtx, workQueue->getParallelismLevel());

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
         deviceCtx = std::move(deviceCtx)]() {
          TRACY_ZONE_SCOPED_NCT("runKernelAndExtendLifetimes", TRACY_COLOR_TEAL,
                                "(Summary: " + dSummary.get().str() + ")");

          llvm::SmallVector<void *> kernelInputs =
              getKernelInputs(takenOperands, outResults, constantResources);
          MAYBE_LOG_ASYNC_KERNEL(*dSummary, takenOperands, outResults);

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
// mgp.buffer.plan
//===----------------------------------------------------------------------===//

static void mgp_buffer_plan(
    RemainingArguments inputs, Result<size_t> highWaterMark,
    RemainingResults results, MEFAttr<ArrayRef<uint64_t>> alignments,
    MEFAttr<ArrayRef<uint64_t>> minPre, MEFAttr<ArrayRef<uint64_t>> minPost,
    MEFAttr<ArrayRef<uint64_t>> maxPre, MEFAttr<ArrayRef<uint64_t>> maxPost,
    Runtime &runtime) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_plan", TRACY_COLOR_TEAL);

  // This function implements a simple greedy algorithm to reuse memory blocks
  // from allocations that do not overlap in lifetime.

  // Allocations should be passed to this primitive in roughly topological order
  // since this algorithm does a single pass through all allocations in order

  // For each allocation:
  //  1. Try to find the smallest existing memory block that can be reused.
  //     A block can be reused if its lifetime doesn't overlap with the
  //     current allocation. This information comes from checking ancestry in
  //     the dependency tree. See MGPBufferLifetimeAnalysis for more details.
  //  2. If a reusable block is found, use its offset for this allocation
  //  (according to alignment constraints) and update its lifetime.
  //  3. If no reusable block is found, allocate a new block at
  //    the end of the pool.

  struct MemoryBlock {
    size_t offset;
    size_t size;
    uint64_t minPre;
    uint64_t maxPre;
    uint64_t minPost;
    uint64_t maxPost;
  };

  size_t numAllocs = inputs.size();
  for (auto &result : results)
    result = AsyncValueRef<size_t>::allocate(runtime);

  auto canReuse = [&](size_t i, const MemoryBlock &block) {
    return maxPre[i] < block.minPre && minPost[i] > block.maxPost;
  };

  assert(!alignments->empty());
  uint64_t maxAlignment = *llvm::max_element(alignments.get());

  SmallVector<MemoryBlock, 64> memoryBlocks;
  memoryBlocks.reserve(inputs.size());

  size_t poolSize = 0;
  [[maybe_unused]] size_t numReused = 0;
  for (size_t i = 0; i < numAllocs; ++i) {
    size_t bestSize = std::numeric_limits<size_t>::max();
    size_t allocSize = inputs[i].get<size_t>();

    // Try to find the smallest memory block we can reuse
    MemoryBlock *bestBlock = nullptr;
    for (MemoryBlock &block : memoryBlocks) {
      if (canReuse(i, block) && allocSize <= block.size) {
        if (block.size < bestSize) {
          bestSize = block.size;
          bestBlock = &block;
        }
      }
    }

    if (bestBlock) {
      // We found a memory block to reuse. Align the offset to this allocation's
      // alignment and emplace the output
      results[i].copy().emplace<size_t>(bestBlock->offset);

      ++numReused;

      // Update lifetime of memory block
      bestBlock->minPre = std::min(bestBlock->minPre, minPre[i]);
      bestBlock->maxPre = std::max(bestBlock->maxPre, maxPre[i]);
      bestBlock->minPost = std::min(bestBlock->minPost, minPost[i]);
      bestBlock->maxPost = std::max(bestBlock->maxPost, maxPost[i]);
    } else {
      // If we couldn't find a block to reuse, increment the poolSize and create
      // a new memory block
      size_t newOffset = llvm::alignTo(poolSize, maxAlignment);
      results[i].copy().emplace<size_t>(newOffset);

      poolSize = newOffset + allocSize;

      memoryBlocks.push_back(
          {newOffset, allocSize, minPre[i], maxPre[i], minPost[i], maxPost[i]});
    }
  }

  // Emplace highWaterMark
  std::move(highWaterMark).emplace(poolSize);

  LLVM_DEBUG({
    size_t totalAllocSize = 0;
    for (const auto &input : inputs) {
      size_t size = input.get<size_t>();
      totalAllocSize += size;
    }

    // Calculate statistics
    size_t memorySaved = totalAllocSize - poolSize;
    double reusePercentage = (static_cast<double>(numReused) / numAllocs) * 100;
    double savingsPercentage =
        (static_cast<double>(memorySaved) / totalAllocSize) * 100;

    // Print statistics
    llvm::dbgs() << "mgp.buffer.plan stats:\n";
    llvm::dbgs() << "  Total Allocations: " << numAllocs << "\n";
    llvm::dbgs() << "  Reused Allocations: " << numReused << " ("
                 << llvm::format("%.2f", reusePercentage) << "%)\n";
    llvm::dbgs() << "  Total Allocation Size: " << totalAllocSize << " bytes\n";
    llvm::dbgs() << "  Final Pool Size: " << poolSize << " bytes\n";
    llvm::dbgs() << "  Memory Saved: " << memorySaved << " bytes ("
                 << llvm::format("%.2f", savingsPercentage) << "%)\n";
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
                        Runtime &runtime) {
  TRACY_ZONE_SCOPED_NC("mgp_runtime_create", TRACY_COLOR_TEAL);

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

            auto cfgOr = KGEN::MojoConfig::open();
            if (cfgOr.isError())
              return cfgOr.takeError();
            KGEN::MojoConfig cfg = std::move(*cfgOr);

            // Create the execution engine.
            KGEN::ExecutionEngineOptions options;
            options.registerPerfPlugins =
                runtime.getProfilerDebuginfo() ==
                M::AsyncRT::RuntimeOptions::ProfilerDebuginfo::kPerfProfiler;
            options.libraryPaths.push_back(cfg.getMGPRTPath());
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
// mgp.buffer.cache
//===----------------------------------------------------------------------===//

static void mgp_buffer_cache(Chain inChain, StateContext &theContext,
                             AsyncValueRef<TensorBufferRef> buffer,
                             Result<Chain> outChain,
                             MEFAttr<uint64_t> bBufferSlot, Runtime &runtime) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_cache", TRACY_COLOR_TEAL);

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
// mgp.buffer.get_cached
//===----------------------------------------------------------------------===//

static void mgp_buffer_get_cached(Chain inChain, StateContext &theContext,
                                  AsyncValueRef<size_t> bBufferSlot,
                                  Result<TensorBufferRef> outBuffer) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_get_cached", TRACY_COLOR_TEAL);

  auto &stateSlot = theContext.getStateSlot(*bBufferSlot);
  ErrorOr<TensorBufferRef> errOr = stateSlot.getBuffer(/*bufferIndex=*/0);
  if (errOr)
    return std::move(outBuffer).setToError(errOr.takeError());

  std::move(outBuffer).emplace(std::move(*errOr));
}

//===----------------------------------------------------------------------===//
// mgp.buffer.remove_cached
//===----------------------------------------------------------------------===//

static void mgp_buffer_remove_cached(Chain inChain, StateContext &theContext,
                                     Result<Chain> outChain,
                                     MEFAttr<uint64_t> bufferSlot) {
  TRACY_ZONE_SCOPED_NC("mgp_buffer_remove_cached", TRACY_COLOR_TEAL);

  auto &stateSlot = theContext.getStateSlot(*bufferSlot);
  stateSlot.clear();
  std::move(outChain).emplace();
}

//===----------------------------------------------------------------------===//
// registerMGPPrimitives
//===----------------------------------------------------------------------===//

void M::registerMGPPrimitives(PrimitiveRegistry &registry) {
  ADD_PRIMITIVE(registry, "mgp.runtime.create", mgp_runtime_create);
  ADD_PRIMITIVE(registry, "mgp.buffer.cache", mgp_buffer_cache);
  ADD_PRIMITIVE(registry, "mgp.buffer.get_cached", mgp_buffer_get_cached);
  ADD_PRIMITIVE(registry, "mgp.buffer.remove_cached", mgp_buffer_remove_cached);
  ADD_PRIMITIVE(registry, "mgp.buffer.plan", mgp_buffer_plan);
  ADD_PRIMITIVE(registry, "mgp.context.create", mgp_context_create);
  ADD_PRIMITIVE(registry, "mgp.generic.execute", mgp_generic_execute);
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
