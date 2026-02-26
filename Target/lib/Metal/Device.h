//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DEVICE_H
#define TARGET_LIB_METAL_DEVICE_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"

namespace M::Target::Metal {
struct Device {
  explicit Device(MTL::Device *mDevice) : mDevice(mDevice) {}

  M_driver_device_t cast() { return reinterpret_cast<M_driver_device_t>(this); }
  static Device *from(M_driver_device_t handle) {
    return reinterpret_cast<Device *>(handle);
  }

  MTL::Device *mDevice;
};
} // namespace M::Target::Metal

#endif // TARGET_LIB_METAL_DEVICE_H
