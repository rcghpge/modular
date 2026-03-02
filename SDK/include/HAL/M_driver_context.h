//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_CONTEXT_H
#define M_DRIVER_CONTEXT_H

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a context for a device, for use with other functions in this API.
 *
 * @param device (in) The device to create the context for.
 * @param context (out) The context.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE if the device
 *         is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_create(M_driver_driver_t driver,
                                          M_driver_device_t device,
                                          M_driver_context_t *context);

typedef M_driver_status_t (*M_driver_context_create_func_ptr_t)(
    M_driver_driver_t driver, M_driver_device_t device,
    M_driver_context_t *context);

/**
 * Destroy an active context for a device.
 *
 * @param context (in) The context to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         `context` is not a valid handle, or if the context was previously
 *         destroyed.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_destroy(M_driver_driver_t driver,
                                           M_driver_context_t context);

typedef M_driver_status_t (*M_driver_context_destroy_func_ptr_t)(
    M_driver_driver_t driver, M_driver_context_t context);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_CONTEXT_H
