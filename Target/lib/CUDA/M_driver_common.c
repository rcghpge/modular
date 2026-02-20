//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "HAL/M_driver_api.h"
#include "error.h"

M_driver_status_t M_driver_status_message(M_driver_status_t status,
                                          const char **message) {
  int64_t const statusIndex = -status;
  int64_t const numStatusMessages =
      sizeof(statusMessages) / sizeof(statusMessages[0]);
  if (statusIndex < 0 || statusIndex > numStatusMessages) {
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  } else {
    *message = statusMessages[statusIndex];
    return M_DRIVER_STATUS_SUCCESS;
  }
}
