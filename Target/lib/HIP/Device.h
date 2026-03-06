//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_DEVICE_H
#define TARGET_LIB_HIP_DEVICE_H

#include "Driver.h"
#include "Error.h"
#include "HIPApi.h"

struct M_driver_device {
private:
  M_driver_driver &driver;
  int deviceId;

  DriverResult<void> deviceMethodPreamble() {
    hipError_t result = this->driver.hip().hipSetDevice(this->deviceId);
    if (result != hipSuccess) {
      return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
    }

    return {};
  }

public:
  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &drv, int64_t deviceId);

  static M_driver_status_t destroy(M_driver_device_t device);

  explicit M_driver_device(M_driver_driver &driver, int deviceId)
      : driver(driver), deviceId(deviceId) {}

  int id() const { return deviceId; }

  M_driver_driver &getDriver() { return driver; }

  M_driver_status_t property(const char *propertyName, void *value);
};

#endif // TARGET_LIB_HIP_DEVICE_H
