//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_MEMORY_H
#define TARGET_LIB_METAL_MEMORY_H

#include "HAL/M_driver_common.h"
#include "MetalCpp.h"

struct M_driver_memory {
  NS::SharedPtr<MTL::Buffer> buffer;
};

#endif // TARGET_LIB_METAL_MEMORY_H
