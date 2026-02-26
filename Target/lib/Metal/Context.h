//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_CONTEXT_H
#define TARGET_LIB_METAL_CONTEXT_H

#include "HAL/M_driver_common.h"

namespace M::Target::Metal {
struct Device;

struct Context {
  explicit Context(Device *device) : device(device) {}

  [[nodiscard]] static Context *create(Device *device);
  static void destroy(Context *context);

  M_driver_context_t cast() {
    return reinterpret_cast<M_driver_context_t>(this);
  }
  static Context *from(M_driver_context_t handle) {
    return reinterpret_cast<Context *>(handle);
  }

  Device *device;
};
} // namespace M::Target::Metal

#endif // TARGET_LIB_METAL_CONTEXT_H
