//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_CONTEXT_H
#define TARGET_LIB_CUDA_CONTEXT_H

#include "Error.h"

#include <cuda.h>

struct M_driver_device;

struct M_driver_context {
private:
  M_driver_device &device;
  CUcontext primaryCtx;

public:
  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  static M_driver_status_t destroy(M_driver_context_t ctx);

  explicit M_driver_context(M_driver_device &dev, CUcontext ctx)
      : device(dev), primaryCtx(ctx) {}

  M_driver_device &getDevice() { return device; }
  CUcontext getPrimaryCtx() const { return primaryCtx; }
};

#endif // TARGET_LIB_CUDA_CONTEXT_H
