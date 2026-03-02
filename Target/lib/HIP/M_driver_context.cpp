//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_context.h"
#include "Error.h"

M_driver_status_t M_driver_context_create(M_driver_driver_t driver,
                                          M_driver_device_t device,
                                          M_driver_context_t *context) {
  (void)driver;
  (void)device;
  (void)context;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_context_destroy(M_driver_driver_t driver,
                                           M_driver_context_t context) {
  (void)driver;
  (void)context;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
