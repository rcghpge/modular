//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_common.h"

DriverResult<M_driver_device_t> M_driver_device::create(M_driver_driver &drv,
                                                        int64_t deviceId) {
  DriverResult<int64_t> count = drv.deviceCount();
  if (!count)
    return tl::unexpected(count.error());
  if (deviceId < 0 || deviceId >= *count)
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);

  hipDevice_t hipDev;
  if (drv.hip().hipDeviceGet(&hipDev, static_cast<int>(deviceId)) != hipSuccess)
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);

  return new M_driver_device(drv, static_cast<int>(deviceId));
}

M_driver_status_t
M_driver_device::property([[maybe_unused]] const char *propertyName,
                          [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device::destroy(M_driver_device_t device) {
  if (auto preambleResult = device->deviceMethodPreamble(); !preambleResult) {
    return preambleResult.error();
  }

  if (auto resetResult = device->driver.hip().hipDeviceReset();
      resetResult != hipSuccess) {
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  delete device;
  return M_DRIVER_STATUS_SUCCESS;
}
