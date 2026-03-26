//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_common.h"

M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;

  DriverResult<M_driver_function_t> result = M_driver_function::load(
      *context, bundle, function_name, function_name_len);
  if (!result)
    return result.error();
  *function = *result;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return M_driver_function::unload(function);
}
