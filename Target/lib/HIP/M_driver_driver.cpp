//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

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

M_driver_status_t M_driver_property([[maybe_unused]] M_driver_driver_t handle,
                                    const char *property_name, void *value) {
  llvm::StringRef name(property_name);
  if (name == M_DRIVER_PROPERTY_API)
    *static_cast<const char **>(value) = "HIP";
  else
    return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
  return M_DRIVER_STATUS_SUCCESS;
}
