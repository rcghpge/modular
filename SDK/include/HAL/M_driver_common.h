//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef M_DRIVER_COMMON_H
#define M_DRIVER_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define M_DRIVER_API_EXPORT __attribute__((visibility("default")))
#define M_DRIVER_API_LOCAL __attribute__((visibility("hidden")))

/**
 * Common type definitions.
 */
typedef int64_t M_driver_status_t;
typedef struct M_driver_driver *M_driver_driver_t;
typedef struct M_driver_device *M_driver_device_t;
typedef struct M_driver_context *M_driver_context_t;

/**
 * Common status codes.
 * Implementations must use negative integers less than the following for
 * errors, and positive integers greater than 0 for warnings.
 */
#define M_DRIVER_STATUS_SUCCESS ((M_driver_status_t)(0))
#define M_DRIVER_STATUS_UNKNOWN_ERROR ((M_driver_status_t)(-1))
#define M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE ((M_driver_status_t)(-2))
#define M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME ((M_driver_status_t)(-3))
#define M_DRIVER_STATUS_NO_DEVICES ((M_driver_status_t)(-4))
#define M_DRIVER_STATUS_BAD_DEVICE ((M_driver_status_t)(-5))
#define M_DRIVER_STATUS_BAD_CONTEXT ((M_driver_status_t)(-6))
#define M_DRIVER_STATUS_BAD_MEMORY ((M_driver_status_t)(-7))
#define M_DRIVER_STATUS_OUT_OF_MEMORY ((M_driver_status_t)(-8))

/**
 * Retrieve human-readable status message corresponding to a status code.
 *
 * @param status (in) A status code.
 * @param message (out) The corresponding status message as a null-terminated
 *                string. Must be "Success" for M_DRIVER_SUCCESS, and
 *                "Unknown Error" for M_DRIVER_UNKNOWN_ERROR.
 * @return Standard Status Protocol or M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE if
 *         the input status code does not match any of the above common status
 *         codes or others that the API implementation emits.
 * @note The pointer at `*message` is owned by the callee. The caller must not
 *       free it.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_status_message(M_driver_status_t status,
                                          const char **message);

typedef M_driver_status_t (*M_driver_status_message_func_ptr_t)(
    M_driver_status_t status, const char **message);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_COMMON_H
