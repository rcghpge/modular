//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_FUNCTION_H
#define TARGET_LIB_HIP_FUNCTION_H

#include "Error.h"
#include "HAL/M_driver_bundle.h"
#include "HIPApi.h"

struct M_driver_context;

struct M_driver_runtime_bundle {
  hipModule_t module;

  [[nodiscard]] static DriverResult<M_driver_runtime_bundle_t>
  load(M_driver_context &ctx, M_driver_static_bundle_t bundle,
       M_driver_bundle_compilation_options_t opts);

  static M_driver_status_t unload(M_driver_context &ctx,
                                  M_driver_runtime_bundle_t bundle);
};

struct M_driver_function {
  hipFunction_t function;

  [[nodiscard]] static DriverResult<M_driver_function_t>
  load(M_driver_context &ctx, M_driver_runtime_bundle_t bundle,
       const char *function_name, uint64_t function_name_len);

  static M_driver_status_t unload(M_driver_function_t func);

  M_driver_status_t
  occupancyMaxActiveBlocks(M_driver_context &ctx, int32_t block_size,
                           uint64_t dynamic_shared_memory_bytes,
                           int32_t *num_blocks);
};

#endif // TARGET_LIB_HIP_FUNCTION_H
