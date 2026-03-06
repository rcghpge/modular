//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_CONTEXT_H
#define TARGET_LIB_METAL_CONTEXT_H

#include "HAL/M_driver_common.h"

struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device) : device(device) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  M_driver_device_t device;
};

#endif // TARGET_LIB_METAL_CONTEXT_H
