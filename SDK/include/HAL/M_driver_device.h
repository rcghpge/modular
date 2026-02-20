//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef M_DRIVER_DEVICE_H
#define M_DRIVER_DEVICE_H

// This file must not be included directly. #include "M_driver_api.h" instead.

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Retrieve the number of devices supported by this driver implementation.
 *
 * @param count (out) The number of devices supported by the driver. Must
 *              be greater than or equal to 0. In case of error, the pointed-to
 *              value must remain unchanged.
 * @return The status code.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_count(int64_t *count);

typedef M_driver_status_t (*M_driver_device_count_func_ptr_t)(int64_t *count);

/**
 * Retrieve a device by its ID.
 *
 * @param device_id (in) The ID of the device to retrieve.
 *                  Must be in the range [0, `count`-1] where `count` is the
 *                  value previously returned by M_driver_device_count().
 * @param device (out) The device.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE if device_id
           is out of range or if the device_id is within range but the device
           cannot be accessed for any reason.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_get(int64_t device_id,
                                      M_driver_device_t *device);

typedef M_driver_status_t (*M_driver_device_get_func_ptr_t)(
    int64_t device_id, M_driver_device_t *device);

/**
 * Retrieve an integer-valued property of a device, as 64-bit integer.
 *
 * @param device (in) The device to retrieve the property from.
 * @param property_name (in) The name of the property to retrieve.
 * @param value (out) Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * @return Standard Status Protocol or one of
 * - M_DRIVER_STATUS_BAD_DEVICE if the device is not valid.
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_property(M_driver_device_t device,
                                           const char *property_name,
                                           void *value);

typedef M_driver_status_t (*M_driver_device_property_func_ptr_t)(
    M_driver_device_t device, const char *property_name, void *value);

/**
 * Create a context for a device, for use with other functions in this API.
 *
 * @param device (in) The device to create the context for.
 * @param context (out) The context.
 * @return Standard Status Protocol orM_DRIVER_STATUS_BAD_DEVICE if the device
 *         is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_create_context(M_driver_device_t device,
                                                 M_driver_context_t *context);

typedef M_driver_status_t (*M_driver_device_create_context_func_ptr_t)(
    M_driver_device_t device, M_driver_context_t *context);

/**
 * Destroy an active context for a device.
 *
 * @param context (in) The context to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         `context` is not a valid handle, or if the context was previously
 *         destroyed.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_device_destroy_context(M_driver_context_t context);

typedef M_driver_status_t (*M_driver_device_destroy_context_func_ptr_t)(
    M_driver_context_t context);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_DEVICE_H
