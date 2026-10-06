//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_MEMORY_H
#define TARGET_LIB_CPU_MEMORY_H

#include "HAL/M_driver_common.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

/// CPU "device" memory is ordinary host memory: the device address and the
/// host-usable pointer are one and the same.
struct M_driver_memory {
  void *ptr = nullptr;
  uint64_t size = 0;
  // False for a region adopted via wrap (or released via unwrap): freeing the
  // handle then releases only the handle, never the underlying allocation.
  bool owning = true;

  M_driver_status_t getProperty(llvm::StringRef name, void *value) const;
};

#endif // TARGET_LIB_CPU_MEMORY_H
