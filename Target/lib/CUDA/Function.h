//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_FUNCTION_H
#define TARGET_LIB_CUDA_FUNCTION_H

#include <cuda.h>

struct M_driver_function {
  CUmodule module;
  CUfunction function;
};

#endif // TARGET_LIB_CUDA_FUNCTION_H
