//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Shared device-to-worker affinity logic used by both the graph compiler
// (MGPPrimitives) and the Mojo runtime (via CompilerRT).
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_ASYNCRT_DEVICEAFFINITY_H
#define MLRT_ASYNCRT_DEVICEAFFINITY_H

#include "MLRT/AsyncRT/Runtime/Algorithms.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>

namespace M::MLRT {

/// Compute the worker task ID for a given GPU device.
///
/// Resolution order:
///   1. Explicit mapping via MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS config/env.
///   2. NUMA-inferred GPU-to-CPU core mapping (Linux x86 only, and only when
///      the thread pool size matches the physical core count).
///   3. Round-robin fallback across workers 1..N-1 (skipping worker 0).
///
/// Returns kDefaultTaskId (-1) when numWorkers <= 1.
int taskIdForDevice(int deviceId, size_t numWorkers);

/// Like parallelForEachNChainWithTaskIds, but automatically pins each element
/// to the worker thread with affinity for device `elementIdx`, using
/// taskIdForDevice to compute the thread assignment from the runtime's work
/// queue parallelism level.
template <typename... CaptureTys, typename ElementFn>
static inline AsyncValueRef<Chain>
launchDeviceCollective(Runtime &runtime, size_t totalCount,
                       ElementFn &&elementFn, CaptureTys &&...captures) {
  WorkQueue *workQueue = runtime.getWorkQueue();
  auto taskIdFn = [workQueue](size_t deviceIdx) {
    return taskIdForDevice(static_cast<int>(deviceIdx),
                           workQueue->getParallelismLevel());
  };
  return parallelForEachNChainWithTaskIds(
      runtime, totalCount, taskIdFn, std::forward<ElementFn>(elementFn),
      std::forward<CaptureTys>(captures)...);
}

/// Like parallelForEachNChainWithTaskIds, but routes each element to a worker
/// thread that is NOT a per-device-affinitized thread. This avoids contention
/// with GPU command dispatch workers while still allowing parallel execution
/// across the remaining pool of worker threads.
///
/// The complement of the device-thread ID set is computed once from totalCount
/// and the runtime's parallelism level, then tasks are distributed round-robin
/// across that set. Falls back to kDefaultTaskId when all workers are
/// device-affinitized (e.g. single-worker pools).
template <typename... CaptureTys, typename ElementFn>
static inline AsyncValueRef<Chain>
launchNonDeviceCollective(Runtime &runtime, size_t totalCount,
                          ElementFn &&elementFn, CaptureTys &&...captures) {
  WorkQueue *workQueue = runtime.getWorkQueue();
  size_t numWorkers = workQueue->getParallelismLevel();

  // Collect the worker IDs claimed by per-device threads (one per element,
  // matching the assignment that launchDeviceCollective would use).
  llvm::SmallVector<int, 8> deviceThreadIds;
  for (auto i : llvm::seq(numWorkers)) {
    int id = taskIdForDevice(static_cast<int>(i), numWorkers);
    if (id >= 0)
      deviceThreadIds.push_back(id);
  }

  // Build the complement: all valid worker IDs not used for device work.
  llvm::SmallVector<int, 8> nonDeviceIds;
  nonDeviceIds.reserve(totalCount);

  for (auto id : llvm::seq(static_cast<int>(numWorkers))) {
    if (!llvm::is_contained(deviceThreadIds, id))
      nonDeviceIds.push_back(id);

    if (nonDeviceIds.size() >= totalCount)
      break;
  }

  auto taskIdFn = [nonDeviceIds = std::move(nonDeviceIds)](size_t i) -> int {
    if (nonDeviceIds.empty())
      return kDefaultTaskId;
    return nonDeviceIds[i % nonDeviceIds.size()];
  };

  return parallelForEachNChainWithTaskIds(
      runtime, totalCount, std::move(taskIdFn),
      std::forward<ElementFn>(elementFn),
      std::forward<CaptureTys>(captures)...);
}

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_DEVICEAFFINITY_H
