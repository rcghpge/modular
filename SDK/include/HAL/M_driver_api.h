//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#ifndef M_DRIVER_API_H
#define M_DRIVER_API_H

#include "M_driver_common.h"
#include "M_driver_context.h"
#include "M_driver_device.h"
#include "M_driver_driver.h"
#include "M_driver_memory.h"

/**
 * Standard status protocol
 * -----------------------
 * All API-functions MUST return one of the following status codes or one
 * from their respective official descriptions.
 * - M_DRIVER_STATUS_SUCCESS if the operation was successful.
 * - An implementation-defined error code whose human-readable description can
 *   be retrieved by providing it to M_driver_status_message().
 * - M_DRIVER_STATUS_UNKNOWN_ERROR in the worst case where it is impossible to
    determine the actual error.
 */

#endif /* M_DRIVER_API_H */
