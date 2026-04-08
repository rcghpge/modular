//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DEVICE_H
#define TARGET_LIB_METAL_DEVICE_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver, MTL::Device *mDevice)
      : driver(driver), mDevice(mDevice) {}

  M_driver_driver_t driver;
  MTL::Device *mDevice;
};

#endif // TARGET_LIB_METAL_DEVICE_H
