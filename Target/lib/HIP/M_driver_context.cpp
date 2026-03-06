//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_context_create(M_driver_driver_t driver,
                                          M_driver_device_t device,
                                          M_driver_context_t *context) {
  (void)driver;
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  DriverResult<M_driver_context_t> result = M_driver_context::create(*device);
  if (!result)
    return result.error();
  *context = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context_destroy(M_driver_driver_t driver,
                                           M_driver_context_t context) {
  (void)driver;
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_context::destroy(context);
}
