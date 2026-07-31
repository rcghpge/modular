//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_bundle.h"
#include "Context.h"
#include "Device.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "HAL/M_driver_common.h"

M_driver_status_t
M_driver_bundle_load(M_driver_context_t context,
                     M_driver_static_bundle_t bundle,
                     M_driver_bundle_compilation_options_t opts,
                     M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  if (runtime_bundle == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_runtime_bundle_t> result =
      M_driver_runtime_bundle::load(*context, bundle, opts);
  if (!result)
    return result.error();
  *runtime_bundle = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_bundle_unload(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (bundle == nullptr)
    return M_DRIVER_STATUS_BAD_BUNDLE;
  return M_driver_runtime_bundle::unload(*context, bundle);
}
