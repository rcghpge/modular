//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the bundle and function entry points for the CPU HAL
// plugin. The CPU plugin does not load code bundles, so these mirror
// CpuDeviceContext::loadFunction by reporting "unimplemented".
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_bundle.h"
#include "HAL/M_driver_function.h"

M_driver_status_t M_driver_bundle_load(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_static_bundle_t bundle,
    [[maybe_unused]] M_driver_bundle_compilation_options_t opts,
    [[maybe_unused]] M_driver_runtime_bundle_t *runtime_bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_bundle_unload([[maybe_unused]] M_driver_context_t context,
                       [[maybe_unused]] M_driver_runtime_bundle_t bundle) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_load([[maybe_unused]] M_driver_context_t context,
                       [[maybe_unused]] M_driver_runtime_bundle_t bundle,
                       [[maybe_unused]] const char *function_name,
                       [[maybe_unused]] uint64_t function_name_len,
                       [[maybe_unused]] M_driver_function_t *function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_function_unload([[maybe_unused]] M_driver_context_t context,
                         [[maybe_unused]] M_driver_function_t function) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
