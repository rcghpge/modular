//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_context.h"

#include <cassert>

M_driver_context_t M_driver_context::create(M_driver_device_t device) {
  return new M_driver_context(device);
}

void M_driver_context::destroy(M_driver_context_t context) { delete context; }

extern "C" M_driver_status_t
M_driver_context_create([[maybe_unused]] M_driver_driver_t driver,
                        M_driver_device_t device, M_driver_context_t *context) {
  assert(context && "context must not be null");
  *context = M_driver_context::create(device);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_context_destroy([[maybe_unused]] M_driver_driver_t driver,
                         M_driver_context_t context) {
  M_driver_context::destroy(context);
  return M_DRIVER_STATUS_SUCCESS;
}
