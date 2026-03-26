//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_FUNCTION_H
#define M_DRIVER_FUNCTION_H

#include "M_driver_bundle.h"
#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a named function from a previously registered runtime code bundle.
 *
 * @param context (in) The context to load the function into.
 * @param bundle (in) A runtime code bundle obtained from
 *                M_driver_bundle_load().
 * @param function_name (in) Null-terminated name of the function to retrieve.
 * @param function_name_len (in) Length of function_name in bytes.
 * @param function (out) The loaded function handle.
 * @return M_DRIVER_STATUS_SUCCESS, or:
 * - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 * - M_DRIVER_STATUS_BAD_BUNDLE if the bundle ptr is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_load(M_driver_context_t context,
                                         M_driver_runtime_bundle_t bundle,
                                         const char *function_name,
                                         uint64_t function_name_len,
                                         M_driver_function_t *function);

typedef M_driver_status_t (*M_driver_function_load_func_ptr_t)(
    M_driver_context_t context, M_driver_runtime_bundle_t bundle,
    const char *function_name, uint64_t function_name_len,
    M_driver_function_t *function);

/**
 * Unload a previously loaded function and release associated resources.
 *
 * @param context (in) The context the function was loaded into.
 * @param function (in) The function to unload.
 * @return M_DRIVER_STATUS_SUCCESS, or:
 * - M_DRIVER_STATUS_BAD_CONTEXT if the context is not valid.
 * - M_DRIVER_STATUS_BAD_FUNCTION if the function ptr is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_unload(M_driver_context_t context,
                                           M_driver_function_t function);

typedef M_driver_status_t (*M_driver_function_unload_func_ptr_t)(
    M_driver_context_t context, M_driver_function_t function);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_FUNCTION_H
