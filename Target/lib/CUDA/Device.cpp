//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Driver.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_device_count(M_driver_driver_t driver,
                                        int64_t *count) {
  unsigned int nvmlCount = 0;
  if (driver->nvml().nvmlDeviceGetCount_v2(&nvmlCount) != NVML_SUCCESS)
    return M_DRIVER_STATUS_NO_DEVICES;
  *count = static_cast<int64_t>(nvmlCount);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device_get(M_driver_driver_t driver, int64_t id,
                                      M_driver_device_t *device) {
  int64_t deviceCount{};
  QUICK_RETURN_IF_ERROR(M_driver_device_count(driver, &deviceCount));
  if (id < 0 || id >= deviceCount)
    return M_DRIVER_STATUS_BAD_DEVICE;

  CUdevice cudaDevice{};
  CUDA_CHECK(driver->cuda().cuDeviceGet(&cudaDevice, id),
             M_DRIVER_STATUS_BAD_DEVICE);

  M_driver_device *dev = driver->findDevice(cudaDevice);
  if (!dev) {
    driver->devices.emplace_back(cudaDevice);
    dev = &driver->devices.back();
  }

  *device = dev;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_device_property([[maybe_unused]] M_driver_device_t device,
                         [[maybe_unused]] const char *property_name,
                         [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
