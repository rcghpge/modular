//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_CONTEXT_H
#define TARGET_LIB_METAL_CONTEXT_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"
#include "llvm/ADT/DenseMap.h"

struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device) : device(device) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  void trackBuffer(void *ptr, MTL::Buffer *buffer) {
    allocations[reinterpret_cast<uintptr_t>(ptr)] = buffer;
  }

  void untrackBuffer(void *ptr) {
    allocations.erase(reinterpret_cast<uintptr_t>(ptr));
  }

  MTL::Buffer *getTrackedBuffer(void *ptr) {
    auto it = allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != allocations.end())
      return it->second;
    return nullptr;
  }

  M_driver_device_t device;
  llvm::DenseMap<uintptr_t, MTL::Buffer *> allocations;
};

#endif // TARGET_LIB_METAL_CONTEXT_H
