//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Device.h"

#include "HAL/M_driver_device.h"

#include <dlpack/dlpack.h>

#include <cstring>

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_device_t> M_driver_device::create(M_driver_driver &driver,
                                                        int64_t deviceId) {
  DriverResult<int64_t> count = driver.deviceCount();
  if (!count)
    return tl::unexpected(count.error());
  if (deviceId < 0 || deviceId >= *count)
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);

  CUdevice cudaDevice{};
  if (CUresult r = driver.rawCuda.cuDeviceGet(&cudaDevice, deviceId);
      r != CUDA_SUCCESS)
    return tl::unexpected(cudaErrorStatus(
        &driver, r, "cuDeviceGet device " + llvm::Twine(deviceId)));

  return new M_driver_device(driver, cudaDevice);
}

M_driver_status_t M_driver_device::destroy(M_driver_device_t device) {
  delete device;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_device::property(const char *propertyName,
                                            void *value) {
  if (std::strcmp(propertyName, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDA;
    dl->device_id = static_cast<int32_t>(deviceId);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(propertyName,
                  M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDAHost;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}
