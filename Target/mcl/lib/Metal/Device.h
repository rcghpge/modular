//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_DEVICE_H
#define TARGET_LIB_METAL_DEVICE_H

#include "Common.h"
#include "MetalCpp.h"
#include "llvm/ADT/StringRef.h"

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver, MTL::Device *mDevice)
      : driver(driver), mDevice(mDevice) {}

  DriverResult<int> getComputeCapability(llvm::StringRef name);
  M_driver_status_t getProperty(llvm::StringRef name, void *value);

  M_driver_driver_t driver;
  MTL::Device *mDevice;
};

#endif // TARGET_LIB_METAL_DEVICE_H
