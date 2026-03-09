//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DEVICE_H
#define TARGET_LIB_CUDA_DEVICE_H

#include "Driver.h"
#include "Error.h"

#include <cuda.h>

struct M_driver_device {
private:
  M_driver_driver &driver;
  CUdevice deviceId;

public:
  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &driver, int64_t deviceId);

  static M_driver_status_t destroy(M_driver_device_t device);

  explicit M_driver_device(M_driver_driver &driver, CUdevice id)
      : driver(driver), deviceId(id) {}

  CUdevice id() const { return deviceId; }
  M_driver_driver &getDriver() { return driver; }
  M_driver_status_t property(const char *propertyName, void *value);
};

#endif // TARGET_LIB_CUDA_DEVICE_H
