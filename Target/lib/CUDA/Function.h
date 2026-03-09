//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CUDA_FUNCTION_H
#define TARGET_LIB_CUDA_FUNCTION_H

#include "Error.h"

#include <cuda.h>

struct M_driver_context;

struct M_driver_function {
  CUmodule module;
  CUfunction function;

  [[nodiscard]] static DriverResult<M_driver_function_t>
  load(M_driver_context &ctx, const void *code, const char *function_name,
       int32_t max_dynamic_shared_bytes, const char *debug_level,
       int32_t optimization_level);

  static M_driver_status_t unload(M_driver_context &ctx,
                                  M_driver_function_t func);
};

#endif // TARGET_LIB_CUDA_FUNCTION_H
