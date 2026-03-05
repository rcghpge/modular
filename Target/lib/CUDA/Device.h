//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_DEVICE_H
#define TARGET_LIB_CUDA_DEVICE_H

#include "HAL/M_driver_common.h"

#include <cuda.h>

namespace M::Target::CUDA {

struct Device {
  CUdevice deviceId;

  explicit Device(CUdevice deviceId) : deviceId(deviceId) {}

  M_driver_device_t cast() { return reinterpret_cast<M_driver_device_t>(this); }

  static Device &from(M_driver_device_t handle) {
    return *reinterpret_cast<Device *>(handle);
  }
};

} // namespace M::Target::CUDA

#endif // TARGET_LIB_CUDA_DEVICE_H
