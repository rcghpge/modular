//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Configuration.h"
#include "Support/Threading/HWInfo.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <map>

using namespace llvm;

namespace M::MLRT {

/// Infer a mapping from each GPU device index to a NUMA-local CPU core.
/// Returns an empty map on any failure (unsupported platform, missing
/// topology, device enumeration errors, etc.).
static std::map<int, int> inferGpuToCpuCoreMapping() {
#if MODULAR_HAL_DEVICE_CONTEXT
  // Under --config=hal-device-context the legacy C++ DeviceContext is not
  // linked into Mojo programs, so device enumeration is unavailable here.
  // Returning an empty mapping makes taskIdForDevice fall back to round-robin
  // placement (tier 3), the same behavior as any other inference failure.
  return {};
#else
  const auto &numaTopologyOrError = NUMATopology::get();
  if (numaTopologyOrError.isError())
    return {};
  const NUMATopology &numaTopology = *numaTopologyOrError;

  int numDevices =
      M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI);

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
#endif // MODULAR_HAL_DEVICE_CONTEXT
}

/// Cached wrapper around inferGpuToCpuCoreMapping().
static const std::map<int, int> &getGpuToCpuCoreMapping() {
  static std::map<int, int> mapping = inferGpuToCpuCoreMapping();
  return mapping;
}

/// Parse and validate the runtime.device_task_cpu_ids config (env var
/// MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS). Initialized once; aborts on
/// invalid input (matching the original MGPPrimitives.cpp behavior).
static ArrayRef<int> getDeviceTaskCpuIds() {
  static SmallVector<int, 8> deviceTaskCpuIds = []() {
    SmallVector<int, 8> result;
    auto config = Config::open();
    if (config.isError())
      return result;

    SmallVector<StringRef, 8> strs;
    config->getValueAsList("runtime.device_task_cpu_ids", strs);
    for (size_t i = 0; i < strs.size(); i++) {
      int cpuId;
      if (strs[i].getAsInteger(10, cpuId)) {
        llvm::errs() << "Failed to parse runtime.device_task_cpu_ids[" << i
                     << "] as an integer: " << strs[i] << "\n";
        std::abort();
      }
      if (cpuId == 0) {
        llvm::errs()
            << "CPU ID 0 is not allowed in runtime.device_task_cpu_ids\n";
        std::abort();
      } else if (std::find(result.begin(), result.end(), cpuId) !=
                 result.end()) {
        llvm::errs()
            << "Duplicate CPU ID found in runtime.device_task_cpu_ids: "
            << cpuId << "\n";
        std::abort();
      } else {
        result.push_back(cpuId);
      }
    }
    return result;
  }();
  return deviceTaskCpuIds;
}

int taskIdForDevice(int deviceId, size_t numWorkers) {
  if (numWorkers <= 1)
    return kDefaultTaskId;

  // Tier 1: explicit per-device CPU IDs from config/env.
  auto deviceTaskCpuIds = getDeviceTaskCpuIds();
  if (deviceTaskCpuIds.size() > static_cast<size_t>(deviceId))
    return deviceTaskCpuIds[deviceId];

  // Tier 2: NUMA-inferred GPU-to-CPU core mapping.
  // Only use when numWorkers matches physical core count, indicating the
  // thread pool wasn't limited by cgroups or other means.
  if (numWorkers >= M::getNumPhysicalCores()) {
    auto gpuToCpuCoreMapping = getGpuToCpuCoreMapping();
    if (!gpuToCpuCoreMapping.empty()) {
      auto it = gpuToCpuCoreMapping.find(deviceId);
      if (it != gpuToCpuCoreMapping.end())
        return it->second;
    }
  }

  // Tier 3: round-robin across workers 1..N-1 (skip worker 0).
  return 1 + (deviceId % static_cast<int>(numWorkers - 1));
}

static ArrayRef<int> getNonDeviceTaskIds() {
  static SmallVector<int> nonDeviceIds = [] {
    SmallVector<int> allDeviceTaskIds;

    // Treat taskID 0 as a device thread as well to avoid assigning to thread 0,
    // which is the MEF executor thread.
    allDeviceTaskIds.push_back(0);
    llvm::append_range(allDeviceTaskIds, getDeviceTaskCpuIds());

    for (auto [_, v] : getGpuToCpuCoreMapping())
      allDeviceTaskIds.push_back(v);

    llvm::sort(allDeviceTaskIds);

    allDeviceTaskIds.push_back(std::numeric_limits<int>::max());

    int largestId = allDeviceTaskIds.back();
    int counter = 0;

    SmallVector<int> nextNonDeviceId;
    for (int _ : llvm::seq(largestId)) {
      int next = counter++;

      // Find the next non-device ID entry.
      while (llvm::binary_search(allDeviceTaskIds, next))
        next = counter++;

      nextNonDeviceId.push_back(next);
    }

    return nextNonDeviceId;
  }();

  return nonDeviceIds;
}

int taskIdForNonDevice(int taskId, size_t numWorkers) {
  ArrayRef<int> deviceTaskIds = getNonDeviceTaskIds();

  auto getNextId = [&](int taskId) {
    if (taskId >= (int)deviceTaskIds.size()) {
      int diff = (taskId - (int)deviceTaskIds.size());
      taskId = diff + deviceTaskIds.back();
    } else {
      taskId = deviceTaskIds[taskId];
    }

    return taskId;
  };

  // Get the next available non-device task ID.
  int nextId = getNextId(taskId % static_cast<int>(numWorkers));

  // If it is outside the number of workers, try again from 0.
  if (nextId >= (int)numWorkers)
    nextId = getNextId(0);

  // Give up if no valid ID was found.
  if (nextId >= (int)numWorkers)
    nextId = kDefaultTaskId;

  return nextId;
}

} // namespace M::MLRT
