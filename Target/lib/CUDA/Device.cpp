//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Device.h"

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

M_driver_status_t
M_driver_device::property([[maybe_unused]] const char *propertyName,
                          [[maybe_unused]] void *value) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
