//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
