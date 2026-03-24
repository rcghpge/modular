//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_CONTEXT_H
#define TARGET_LIB_CUDA_CONTEXT_H

#include "Error.h"
#include "HAL/M_driver_memory.h"
#include "llvm/ADT/DenseMap.h"

#include <cuda.h>

struct M_driver_device;

struct M_driver_context {
  M_driver_device &device;
  CUcontext primaryCtx;

  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  static M_driver_status_t destroy(M_driver_context_t ctx);

  explicit M_driver_context(M_driver_device &dev, CUcontext ctx)
      : device(dev), primaryCtx(ctx) {}

  M_driver_status_t ensureContextBound();

  DriverResult<void *> allocPinned(uint64_t size);
  DriverResult<void *> allocSync(uint64_t size);
  M_driver_status_t freePinned(void *memory);
  M_driver_status_t freeSync(void *memory);
};

#endif // TARGET_LIB_CUDA_CONTEXT_H
