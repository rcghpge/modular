//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_MEMORY_H
#define TARGET_LIB_METAL_MEMORY_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"

#include "llvm/ADT/StringRef.h"

struct M_driver_memory {
  MTL::Buffer *mBuffer;
  // Owning handles release the buffer on free; `unwrap` clears this so a
  // borrowed handle's free releases only the wrapper.
  bool owning = true;
  // Sub-range offset within `mBuffer`: every derived address must add it, or a
  // wrapped sub-range would alias the buffer base. Zero for whole-buffer
  // allocs.
  uint64_t offset = 0;

  M_driver_status_t getProperty(llvm::StringRef name, void *value);
};

#endif // TARGET_LIB_METAL_MEMORY_H
