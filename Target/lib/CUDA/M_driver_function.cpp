//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Error.h"
#include "Function.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_function_load(
    M_driver_driver_t driver, M_driver_context_t context, const void *code,
    uint64_t code_size, const char *function_name,
    int32_t max_dynamic_shared_bytes, const char *debug_level,
    int32_t optimization_level, M_driver_function_t *function) {
  (void)driver;
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  DriverResult<M_driver_function_t> result = M_driver_function::load(
      *context, {static_cast<const char *>(code), code_size}, function_name,
      max_dynamic_shared_bytes, debug_level, optimization_level);
  if (!result)
    return result.error();
  *function = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_driver_t driver,
                                           M_driver_context_t context,
                                           M_driver_function_t function) {
  (void)driver;
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_function::unload(*context, function);
}
