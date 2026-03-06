//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_CONTEXT_H
#define TARGET_LIB_CUDA_CONTEXT_H

#include "HAL/M_driver_common.h"

#include <cuda.h>

struct M_driver_context {
  CUcontext primaryCtx;
};

#endif // TARGET_LIB_CUDA_CONTEXT_H
