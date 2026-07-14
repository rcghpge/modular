//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_common.h"
#include "HAL/M_driver_device.h"
#include "llvm/ADT/Twine.h"

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
