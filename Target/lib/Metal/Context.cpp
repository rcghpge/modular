//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_context.h"

#include <cassert>

using namespace M::Target::Metal;

Context *Context::create(Device *device) { return new Context(device); }

void Context::destroy(Context *context) { delete context; }

extern "C" M_driver_status_t
M_driver_context_create([[maybe_unused]] M_driver_driver_t driver,
                        M_driver_device_t device, M_driver_context_t *context) {
  assert(context && "context must not be null");
  *context = Context::create(Device::from(device))->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_context_destroy([[maybe_unused]] M_driver_driver_t driver,
                         M_driver_context_t context) {
  Context::destroy(Context::from(context));
  return M_DRIVER_STATUS_SUCCESS;
}
