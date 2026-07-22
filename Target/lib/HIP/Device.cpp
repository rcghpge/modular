//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_common.h"
#include "HAL/M_driver_device.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cstring>
#include <dlfcn.h>

#include <dlpack/dlpack.h>

#include <cstring>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_device_t> M_driver_device::create(M_driver_driver &drv,
                                                        int64_t deviceId) {
  DriverResult<int64_t> count = drv.deviceCount();
  if (!count)
    return tl::unexpected(count.error());
  if (deviceId < 0 || deviceId >= *count)
    return tl::unexpected(M_DRIVER_STATUS_BAD_DEVICE);

  hipDevice_t hipDev;
  if (hipError_t r =
          drv.hip().hipDeviceGet(&hipDev, static_cast<int>(deviceId));
      r != hipSuccess)
    return tl::unexpected(hipErrorStatus(
        &drv, r, "hipDeviceGet device " + llvm::Twine(deviceId)));

  return new M_driver_device(drv, static_cast<int>(deviceId));
}

M_driver_status_t M_driver_device::property(const char *propertyName,
                                            void *value) {
  if (propertyName == nullptr || value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (llvm::StringRef(propertyName) == M_DRIVER_DEVICE_PROPERTY_NAME) {
    hipError_t result = driver.hip().hipDeviceGetName(
        static_cast<char *>(value), M_DRIVER_DEVICE_PROPERTY_NAME_MAX_LEN,
        deviceId);
    if (result != hipSuccess)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (llvm::StringRef(propertyName) == "arch") {
    // The LLVM/compile target arch (e.g. "gfx942") lives in
    // `hipDeviceProp_t::gcnArchName`. The properties entry point is nullable
    // (see HIPApi::getDeviceProperties); without it only this query degrades.
    auto getProps = driver.hip().getDeviceProperties;
    if (getProps == nullptr)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    hipDeviceProp_t prop;
    if (getProps(&prop, deviceId) != hipSuccess)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    // `gcnArchName` may carry target features, e.g. "gfx942:sramecc+:xnack-";
    // the compile target wants the base "gfx942".
    llvm::StringRef arch = llvm::StringRef(prop.gcnArchName).split(':').first;
    size_t n = arch.size();
    if (n >= M_DRIVER_DEVICE_PROPERTY_NAME_MAX_LEN)
      n = M_DRIVER_DEVICE_PROPERTY_NAME_MAX_LEN - 1;
    char *out = static_cast<char *>(value);
    memcpy(out, arch.data(), n);
    out[n] = '\0';
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(propertyName, M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLROCM;
    dl->device_id = static_cast<int32_t>(id());
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (std::strcmp(propertyName,
                  M_DRIVER_DEVICE_PROPERTY_DLPACK_DEVICE_PINNED) == 0) {
    auto *dl = static_cast<M_driver_dlpack_device *>(value);
    dl->device_type = kDLROCMHost;
    dl->device_id = 0;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_device::destroy(M_driver_device_t device) {
  if (auto preambleResult = device->deviceMethodPreamble(); !preambleResult) {
    return preambleResult.error();
  }

  if (hipError_t r = device->getDriver().hip().hipDeviceReset();
      r != hipSuccess)
    return hipErrorStatus(&device->getDriver(), r, "hipDeviceReset");

  delete device;
  return M_DRIVER_STATUS_SUCCESS;
}

// CUDA `CUdevice_attribute` enum values used by callers. We translate them
// into HIP's `hipDeviceAttribute_t` since the two enums have different
// numeric layouts. Extend as new attributes are needed by real callers.
static hipDeviceAttribute_t cudaToHipAttribute(int32_t cudaAttr) {
  switch (cudaAttr) {
  case 6: // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
    return hipDeviceAttributeMaxGridDimY;
  case 16: // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    return hipDeviceAttributeMultiprocessorCount;
  case 75: // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    return hipDeviceAttributeComputeCapabilityMajor;
  case 76: // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    return hipDeviceAttributeComputeCapabilityMinor;
  case 95: // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
    return hipDeviceAttributeCooperativeLaunch;
  default:
    // Sentinel — reported as ARGUMENT_OUT_OF_RANGE below.
    return static_cast<hipDeviceAttribute_t>(-1);
  }
}

M_driver_status_t M_driver_device::attribute(int32_t attribute,
                                             int32_t *value) {
  if (value == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  hipDeviceAttribute_t hipAttr = cudaToHipAttribute(attribute);
  if (static_cast<int>(hipAttr) == -1)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  int hipValue = 0;
  if (hipError_t r =
          getDriver().hip().hipDeviceGetAttribute(&hipValue, hipAttr, id());
      r != hipSuccess)
    return hipErrorStatus(&getDriver(), r, "hipDeviceGetAttribute");
  *value = static_cast<int32_t>(hipValue);
  return M_DRIVER_STATUS_SUCCESS;
}
