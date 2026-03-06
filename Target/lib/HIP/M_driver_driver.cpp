//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "llvm/ADT/StringRef.h"

M_driver_status_t M_driver_create(M_driver_driver_t *driver) {
  DriverResult<M_driver_driver_t> res = M_driver_driver::create();
  if (!res)
    return res.error();

  *driver = *res;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_destroy(M_driver_driver_t handle) {
  if (handle == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_driver::destroy(handle);
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
