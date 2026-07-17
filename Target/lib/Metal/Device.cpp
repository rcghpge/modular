//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Context.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

#include <dlpack/dlpack.h>

#include <cassert>
#include <cstring>

using namespace llvm;

extern "C" M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                                   int64_t *count) {
  assert(count && "count must not be null");
  if (!driver || driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = driver->devices.size();
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t M_driver_device_get(M_driver_driver_t driver,
                                                 int64_t id,
                                                 M_driver_device_t *device) {
  assert(device && "device must not be null");
  if (!driver || driver->devices.size() == 0)
    return M_DRIVER_STATUS_NO_DEVICES;
  if (id < 0 || size_t(id) >= driver->devices.size())
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->devices[id];
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_device_property([[maybe_unused]] M_driver_device_t device,
                         const char *property_name, void *value) {
  if (std::strcmp(property_name, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) == 0 ||
      std::strcmp(property_name,
                  M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLMetal;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

extern "C" M_driver_status_t
M_driver_device_attribute([[maybe_unused]] M_driver_device_t device,
                          [[maybe_unused]] int32_t attribute,
                          [[maybe_unused]] int32_t *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
