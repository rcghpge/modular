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

struct AllocInfo {
  M_driver_alloc_mode_t allocMode;
};

struct M_driver_context {
private:
  M_driver_device &device;
  CUcontext primaryCtx;
  llvm::DenseMap<uintptr_t, AllocInfo> allocations;

public:
  [[nodiscard]] static DriverResult<M_driver_context_t>
  create(M_driver_device &dev);

  static M_driver_status_t destroy(M_driver_context_t ctx);

  explicit M_driver_context(M_driver_device &dev, CUcontext ctx)
      : device(dev), primaryCtx(ctx) {}

  M_driver_device &getDevice() { return device; }
  CUcontext getPrimaryCtx() const { return primaryCtx; }

  DriverResult<void *> alloc(M_driver_alloc_mode_t mode, uint64_t size);
  M_driver_status_t free(void *memory);

  // Track allocations so free() knows which API to call.
  void trackAlloc(void *ptr, AllocInfo info) {
    allocations[reinterpret_cast<uintptr_t>(ptr)] = info;
  }

  void untrackAlloc(void *ptr) {
    allocations.erase(reinterpret_cast<uintptr_t>(ptr));
  }

  const AllocInfo *getAllocInfo(void *ptr) const {
    auto it = allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != allocations.end())
      return &it->second;
    return nullptr;
  }
};

#endif // TARGET_LIB_CUDA_CONTEXT_H
