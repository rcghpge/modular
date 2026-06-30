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
  NS::SharedPtr<MTL::Buffer> mBuffer;

  M_driver_status_t getProperty(llvm::StringRef name, void *value);
};

#endif // TARGET_LIB_METAL_MEMORY_H
