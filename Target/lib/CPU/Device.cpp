//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the device-level entry points of the CPU HAL plugin.
// The CPU plugin exposes a single device, mirroring CpuDevice::deviceCount().
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/HostSystem.h"
#include "Common.h"
#include "Driver.h"
#include "HAL/M_driver_device.h"

namespace AsyncRT = M::AsyncRT;

#include <dlpack/dlpack.h>

#include "llvm/ADT/StringRef.h"
#include <cstring>

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  if (count == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  *count = 1;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver, int64_t id,
                                      M_driver_device_t *device) {
  if (device == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  if (id != 0)
    return M_DRIVER_STATUS_BAD_DEVICE;
  *device = &driver->device;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (property_name == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (llvm::StringRef(property_name) == M_DRIVER_DEVICE_PROPERTY_NAME) {
    llvm::StringRef name = "CPU";
    char *out = static_cast<char *>(value);
    std::memcpy(out, name.data(), name.size());
    out[name.size()] = '\0';
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(property_name, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) == 0 ||
      std::strcmp(property_name,
                  M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCPU;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t
M_driver_device_attribute([[maybe_unused]] M_driver_device_t device,
                          int32_t attribute, int32_t *value) {
  if (device == nullptr)
    return M_DRIVER_STATUS_BAD_DEVICE;
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // Mirrors CpuDeviceContext::getAttribute: only PARALLELISM_LEVEL (1000) is
  // answered; other attributes report a soft error so callers can fall back.
  if (attribute != 1000)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  AsyncRT::CPUDeviceRef cpuDevice = AsyncRT::getOrCreateCPUDevice(
      AsyncRT::CPUDeviceSource::CPUDeviceContext,
      AsyncRT::CPUDeviceOptions().withMainWillNotDonate(),
      /*allowUsingExistingOptions=*/true);
  *value =
      static_cast<int32_t>(cpuDevice->getWorkQueue()->getParallelismLevel());
  return M_DRIVER_STATUS_SUCCESS;
}
