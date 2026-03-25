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
 * @param version (inout) The version of the driver interface implemented by
 * both sides. Before entry, Modular fills in the version it was built with.
 * On exit, the plugin MUST overwrite it with the version it implements.
 * @param driver (out) The driver object, which may store any persistent state.
 *
 * @return M_DRIVER_STATUS_SUCCESS if the driver was created successfully.
 *         M_DRIVER_STATUS_INCOMPATIBLE_VERSION if the version is incompatible.
 * @note In the case where the Modular's major version is the same as the
 *       plugin's and Modular's minor version is greater than the plugin's,
 *       the call must succeed and the plugin must return
 *       M_DRIVER_STATUS_SUCCESS. Modular will then decide if/how to proceed.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_create(M_driver_version *version,
                                  M_driver_driver_t *driver);

typedef M_driver_status_t (*M_driver_create_func_ptr_t)(
    M_driver_version *version, M_driver_driver_t *driver);

/**
 * Perform shutdown activities after using all other functions in this API,
 * except for M_driver_status_message().
 *
 * @param driver (in) The driver to destroy.
 *
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_destroy(M_driver_driver_t driver);

typedef M_driver_status_t (*M_driver_destroy_func_ptr_t)(
    M_driver_driver_t driver);

/**
 * Common driver properties
 */
#define M_DRIVER_PROPERTY_API "api"
#define M_DRIVER_PROPERTY_VERSION "version"

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
M_driver_status_t M_driver_property(M_driver_driver_t handle,
                                    const char *property_name, void *value);

typedef M_driver_status_t (*M_driver_property_func_ptr_t)(
    M_driver_driver_t handle, const char *property_name, void *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_DRIVER_H
