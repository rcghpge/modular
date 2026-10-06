//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_DEVICE_H
#define TARGET_LIB_CPU_DEVICE_H

#include "HAL/M_driver_common.h"

struct M_driver_device {
  explicit M_driver_device(M_driver_driver_t driver) : driver(driver) {}

  M_driver_driver_t driver;
};

#endif // TARGET_LIB_CPU_DEVICE_H
