//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_DRIVER_H
#define TARGET_LIB_CPU_DRIVER_H

#include "Common.h"
#include "Device.h"
#include "Target/lib/Support/Utils.h"

using M::Target::CustomStatus;

struct M_driver_driver {
  M_driver_driver() : device(this) {
    status.messages[M_DRIVER_STATUS_UNIMPLEMENTED] = "Unimplemented";
  }

  [[nodiscard]] static M_driver_driver_t create();
  static void destroy(M_driver_driver_t driver);

  // The CPU plugin exposes exactly one CPU device.
  M_driver_device device;
  CustomStatus status{M_DRIVER_NUM_CUSTOM_STATUS_CODES};
};

#endif // TARGET_LIB_CPU_DRIVER_H
