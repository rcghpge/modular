//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements context lifecycle for the CPU HAL plugin. A context
// acquires the process-wide AsyncRT CPUDevice, exactly as
// M::Driver::CpuDeviceContext does in its constructor.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Device.h"
#include "HAL/M_driver_context.h"

#include <cassert>

M_driver_context_t M_driver_context::create(M_driver_device_t device) {
  // Get or create the AsyncRT CPUDevice. Request mainWillDonate=false since
  // the HAL caller drives work from its own thread, but allow reusing an
  // existing CPUDevice's options if one already exists.
  AsyncRT::CPUDeviceRef cpuDevice = AsyncRT::getOrCreateCPUDevice(
      AsyncRT::CPUDeviceSource::CPUDeviceContext,
      AsyncRT::CPUDeviceOptions().withMainWillNotDonate(),
      /*allowUsingExistingOptions=*/true);
  return new (std::nothrow) M_driver_context(device, std::move(cpuDevice));
}

void M_driver_context::destroy(M_driver_context_t context) { delete context; }

M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  assert(context && "context must not be null");
  M_driver_context_t result = M_driver_context::create(device);
  if (result == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *context = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_context_t context) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  M_driver_context::destroy(context);
  return M_DRIVER_STATUS_SUCCESS;
}
