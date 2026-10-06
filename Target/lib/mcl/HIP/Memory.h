//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_MEMORY_H
#define TARGET_LIB_HIP_MEMORY_H

#include "HAL/M_driver_memory.h"

#include "llvm/ADT/StringRef.h"

struct M_driver_memory {
  void *ptr;
  // False for wrapped (externally-owned) memory: freeing the handle must
  // not free the underlying region.
  bool owning = true;
  // True only for pinned host allocations, whose `ptr` is host-dereferenceable.
  bool hostAccessible = false;

  M_driver_status_t getProperty(llvm::StringRef name, void *value) {
    if (name == M_DRIVER_MEMORY_PROPERTY_ADDRESS) {
      *static_cast<void **>(value) = ptr;
    } else if (name == M_DRIVER_MEMORY_PROPERTY_HOST_ADDRESS) {
      if (!hostAccessible)
        return M_DRIVER_STATUS_INVALID_ARG;
      *static_cast<void **>(value) = ptr;
    } else {
      return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
    }
    return M_DRIVER_STATUS_SUCCESS;
  }
};

#endif // TARGET_LIB_HIP_MEMORY_H
