//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

const char *status_messages[M_DRIVER_NUM_ERRORS] = {
    /*0*/ "Success",
    /*1*/ "Unknown error",
    /*2*/ "Argument out of range",
    /*3*/ "Unknown property name",
    /*4*/ "No devices",
    /*5*/ "Bad device",
    /*6*/ "Bad context",
    /*7*/ "Bad memory",
    /*8*/ "Out of memory",
    /*9*/ "Unimplemented",
    /*10*/ "Uninitialized",
};

extern "C" M_driver_status_t M_driver_status_message(M_driver_status_t status,
                                                     const char **message) {
  int64_t statusIndex = -status;
  constexpr int64_t numStatusMessages =
      sizeof(status_messages) / sizeof(status_messages[0]);
  if (statusIndex < 0 || statusIndex > numStatusMessages) {
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  } else {
    *message = status_messages[statusIndex];
    return M_DRIVER_STATUS_SUCCESS;
  }
}
