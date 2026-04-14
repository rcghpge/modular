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

} // namespace M::MLRT

#endif // MLRT_ASYNCRT_DEVICEAFFINITY_H
