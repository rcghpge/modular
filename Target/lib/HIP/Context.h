//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_CONTEXT_H
#define TARGET_LIB_HIP_CONTEXT_H

#include "Error.h"

struct M_driver_device;

struct M_driver_context {
private:
  M_driver_device &device;

public:
  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  static M_driver_status_t destroy(M_driver_context_t ctx) {
    delete ctx;
    return M_DRIVER_STATUS_SUCCESS;
  }

  explicit M_driver_context(M_driver_device &device) : device(device) {}

  M_driver_device &getDevice() { return device; }
};

#endif // TARGET_LIB_HIP_CONTEXT_H
