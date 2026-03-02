//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

#include <optional>

M_driver_status_t M_driver_create(M_driver_driver_t *driver) {
  std::optional<HIPApi> api = HIPApi::load();
  if (!api)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  if (api->hipInit(0) != hipSuccess) {
    // dropping api automatically unloads
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  }

  *driver = (new Driver(std::move(*api)))->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  delete Driver::from_as_ptr(handle);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_property(const char *property_name, void *value) {
  (void)property_name;
  (void)value;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
