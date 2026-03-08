//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_FUNCTION_H
#define M_DRIVER_FUNCTION_H

#include "M_driver_common.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load a named function from device code.
 *
 * @param driver (in) The driver.
 * @param context (in) The context to load the function into.
 * @param code (in) Pointer to the device code image.
 * @param function_name (in) Null-terminated name of the function to retrieve
 *                     from the loaded module.
 * @param max_dynamic_shared_bytes (in) Maximum dynamic shared memory in bytes
 *                                that this function may use. Pass -1 to leave
 *                                at the driver default.
 * @param debug_level (in) JIT debug level.
 * @param optimization_level (in) JIT optimization level.
 * @param function (out) The loaded function handle.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_function_load(M_driver_driver_t driver, M_driver_context_t context,
                       const void *code, const char *function_name,
                       int32_t max_dynamic_shared_bytes,
                       const char *debug_level, int32_t optimization_level,
                       M_driver_function_t *function);

typedef M_driver_status_t (*M_driver_function_load_func_ptr_t)(
    M_driver_driver_t driver, M_driver_context_t context, const void *code,
    const char *function_name, int32_t max_dynamic_shared_bytes,
    const char *debug_level, int32_t optimization_level,
    M_driver_function_t *function);

/**
 * Unload a previously loaded function and release associated resources.
 *
 * @param driver (in) The driver.
 * @param context (in) The context the function was loaded into.
 * @param function (in) The function to unload.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_function_unload(M_driver_driver_t driver,
                                           M_driver_context_t context,
                                           M_driver_function_t function);

typedef M_driver_status_t (*M_driver_function_unload_func_ptr_t)(
    M_driver_driver_t driver, M_driver_context_t context,
    M_driver_function_t function);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_FUNCTION_H
