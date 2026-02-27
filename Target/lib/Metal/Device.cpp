//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Context.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

#include <cassert>

using namespace llvm;
using namespace M::Target::Metal;

extern "C" M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                                   int64_t *count) {
  assert(count && "count must not be null");
  if (!Driver::from(driver) || Driver::from(driver)->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = Driver::from(driver)->devices.size();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                                 int64_t id,
                                                 M_driver_device_t *device) {
  assert(device && "device must not be null");
  if (!Driver::from(driver) || Driver::from(driver)->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  if (id < 0 || size_t(id) >= Driver::from(driver)->devices.size())
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = Driver::from(driver)->devices[id].cast();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                                      const char *property_name,
                                                      void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

extern "C" M_driver_status_t
M_driver_device_create_context([[maybe_unused]] M_driver_driver_t driver,
                               M_driver_device_t device,
                               M_driver_context_t *context) {
  assert(context && "context must not be null");
  *context = Context::create(Device::from(device))->cast();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_device_destroy_context([[maybe_unused]] M_driver_driver_t driver,
                                M_driver_context_t context) {
  Context::destroy(Context::from(context));
  return M_DRIVER_STATUS_SUCCESS;
}
