//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event,
                                        bool enableTiming) {
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

M_driver_status_t M_driver_event_is_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t M_driver_event_elapsed_time(M_driver_context_t context,
                                              M_driver_event_t start,
                                              M_driver_event_t end,
                                              float *elapsed_ms) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
