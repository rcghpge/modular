//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_function.h"

M_driver_status_t
M_driver_function_load([[maybe_unused]] M_driver_driver_t driver,
                       [[maybe_unused]] M_driver_context_t context,
                       [[maybe_unused]] const void *code,
                       [[maybe_unused]] const char *function_name,
                       [[maybe_unused]] int32_t max_dynamic_shared_bytes,
                       [[maybe_unused]] const char *debug_level,
                       [[maybe_unused]] int32_t optimization_level,
                       [[maybe_unused]] M_driver_function_t *function) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_unload([[maybe_unused]] M_driver_driver_t driver,
                         [[maybe_unused]] M_driver_context_t context,
                         [[maybe_unused]] M_driver_function_t function) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
