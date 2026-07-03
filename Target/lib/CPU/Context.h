//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_CONTEXT_H
#define TARGET_LIB_CPU_CONTEXT_H

#include "HAL/M_driver_common.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"

namespace MLRT = M::MLRT;

/// A CPU context owns the AsyncRT CPUDevice that backs all memory allocations
/// and worker-thread execution for queues created from it.
struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device,
                            MLRT::CPUDeviceRef cpuDevice)
      : device(device), cpuDevice(std::move(cpuDevice)) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  MLRT::CPUDevice &cpu() const { return *cpuDevice; }
  MLRT::Allocator *allocator() const { return cpuDevice->getAllocator(); }

  M_driver_device_t device;
  MLRT::CPUDeviceRef cpuDevice;
};

#endif // TARGET_LIB_CPU_CONTEXT_H
