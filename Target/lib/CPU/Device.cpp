//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the device-level entry points of the CPU HAL plugin.
// The CPU plugin exposes a single device, mirroring CpuDevice::deviceCount().
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Common.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

#include <cassert>

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  assert(count && "count must not be null");
  if (!driver)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver, int64_t id,
                                      M_driver_device_t *device) {
  assert(device && "device must not be null");
  if (!driver)
    return M_DRIVER_STATUS_NO_DEVICES;
  if (id != 0)
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->device;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_device_property([[maybe_unused]] M_driver_device_t device,
                         [[maybe_unused]] const char *property_name,
                         [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
