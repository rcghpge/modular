//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef M_DRIVER_DRIVER_H
#define M_DRIVER_DRIVER_H

// This file must not be included directly. #include "M_driver_api.h" instead.

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform initialization activities prior to using other functions in this API.
 *
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_initialize(void);

typedef M_driver_status_t (*M_driver_initialize_func_ptr_t)(void);

/**
 * Perform shutdown activities after using all other functions in this API,
 * except for M_driver_status_message().
 *
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_finalize(void);

typedef M_driver_status_t (*M_driver_finalize_func_ptr_t)(void);

/**
 * Retrieve an integer-valued property.
 *
 * @param property_name (in) The name of the property as a null-terminated
 * string.
 * @param value (out) Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * @return Standard Status Protocol or
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_property(const char *property_name, void *value);

typedef M_driver_status_t (*M_driver_property_func_ptr_t)(
    const char *property_name, void *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_DRIVER_H
