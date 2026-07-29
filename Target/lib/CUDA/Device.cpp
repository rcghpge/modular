//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Device.h"
#include "HAL/M_driver_device.h"
#include "llvm/ADT/StringRef.h"

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

M_driver_status_t M_driver_device::property(llvm::StringRef name, void *value) {
  if (name.empty() || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (name == M_DRIVER_DEVICE_PROPERTY_NAME) {
    if (CUresult r = driver.rawCuda.cuDeviceGetName(
            static_cast<char *>(value), M_DRIVER_DEVICE_PROPERTY_MAX_LEN,
            deviceId);
        r != CUDA_SUCCESS)
      return cudaErrorStatus(&driver, r, "cuDeviceGetName");
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_ARCH) {
    int32_t computeCapabilityMajor;
    if (M_driver_status_t status = attribute(75 /* COMPUTE_CAPABILITY_MAJOR */,
                                             &computeCapabilityMajor))
      return status;

    int32_t computeCapabilityMinor;
    if (M_driver_status_t status = attribute(76 /* COMPUTE_CAPABILITY_MINOR */,
                                             &computeCapabilityMinor))
      return status;

    int cc = computeCapabilityMajor * 10 + computeCapabilityMinor;
    std::string sm = "sm_" + std::to_string(cc);

    // For architectures above 90 we use the "a" variants to enable hardware
    // specific features at the cost of backward and forward compatibility.
    if (cc >= 90)
      sm += "a";

    std::strncpy(static_cast<char *>(value), sm.c_str(),
                 M_DRIVER_DEVICE_PROPERTY_MAX_LEN);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDA;
    dl->device_id = static_cast<int32_t>(deviceId);
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (name == M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLCUDAHost;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device::attribute(int32_t attribute,
                                             int32_t *value) {
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  int cuValue = 0;
  if (CUresult r = driver.rawCuda.cuDeviceGetAttribute(
          &cuValue, static_cast<CUdevice_attribute>(attribute), deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&driver, r, "cuDeviceGetAttribute");
  *value = static_cast<int32_t>(cuValue);
  return M_DRIVER_STATUS_SUCCESS;
}
