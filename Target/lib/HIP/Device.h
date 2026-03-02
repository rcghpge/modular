//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_DEVICE_H
#define TARGET_LIB_HIP_DEVICE_H

#include "HAL/M_driver_common.h"

#include <cassert>

class Device {
private:
  int deviceId;

public:
  M_driver_device_t cast() { return reinterpret_cast<M_driver_device_t>(this); }

  static Device &from(M_driver_device_t handle) {
    Device *devicePtr = reinterpret_cast<Device *>(handle);
    assert(devicePtr != nullptr);
    return *devicePtr;
  }

  explicit Device(int deviceId) : deviceId(deviceId) {}

  int id() const { return deviceId; }
};

#endif // TARGET_LIB_HIP_DEVICE_H
