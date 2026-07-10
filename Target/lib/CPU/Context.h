//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_CONTEXT_H
#define TARGET_LIB_CPU_CONTEXT_H

#include "AsyncRT/Runtime/CPUDevice.h"
#include "HAL/M_driver_common.h"

namespace AsyncRT = M::AsyncRT;

/// A CPU context owns the AsyncRT CPUDevice that backs all memory allocations
/// and worker-thread execution for queues created from it.
struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device,
                            AsyncRT::CPUDeviceRef cpuDevice)
      : device(device), cpuDevice(std::move(cpuDevice)) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  AsyncRT::CPUDevice &cpu() const { return *cpuDevice; }
  AsyncRT::Allocator *allocator() const { return cpuDevice->getAllocator(); }

  M_driver_device_t device;
  AsyncRT::CPUDeviceRef cpuDevice;
};

#endif // TARGET_LIB_CPU_CONTEXT_H
