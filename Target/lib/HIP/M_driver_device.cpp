//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_device_count(M_driver_driver_t handle,
                                        int64_t *count) {
  return Driver::from(handle).deviceCount(*count);
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                      int64_t device_id,
                                      M_driver_device_t *device) {
  (void)driver;
  (void)device_id;
  (void)device;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  (void)device;
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
