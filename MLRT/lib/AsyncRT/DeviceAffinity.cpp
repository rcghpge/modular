//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/DeviceAffinity.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/Configuration.h"
#include "Support/Threading/HWInfo.h"
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

} // namespace M::MLRT
