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
  M_driver_driver &driver;
  CUdevice deviceId;

  [[nodiscard]] static DriverResult<M_driver_device_t>
  create(M_driver_driver &driver, int64_t deviceId);

  static M_driver_status_t destroy(M_driver_device_t device);

  explicit M_driver_device(M_driver_driver &driver, CUdevice id)
      : driver(driver), deviceId(id) {}

  M_driver_status_t property(llvm::StringRef name, void *value);
  M_driver_status_t attribute(int32_t attribute, int32_t *value);
};

#endif // TARGET_LIB_CUDA_DEVICE_H
