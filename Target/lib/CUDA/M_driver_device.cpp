//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  DriverResult<int64_t> result = driver->deviceCount();
  if (!result)
    return result.error();
  *count = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                      int64_t device_id,
                                      M_driver_device_t *device) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  DriverResult<M_driver_device_t> result =
      M_driver_device::create(*driver, device_id);
  if (!result)
    return result.error();
  *device = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  return device->property(property_name, value);
}

M_driver_status_t M_driver_device_attribute(M_driver_device_t device,
                                            int32_t attribute, int32_t *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  return device->attribute(attribute, value);
}
