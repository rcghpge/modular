//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Event.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event,
                                        bool enableTiming) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  DriverResult<M_driver_event_t> result =
      M_driver_event::create(*context, enableTiming);
  if (!result)
    return result.error();
  *event = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (context == nullptr || event == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_event::destroy(*context, event);
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (context == nullptr || event == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_event::synchronize(*context, event);
}

M_driver_status_t M_driver_event_is_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (context == nullptr || event == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_event::isReady(*context, event, is_ready);
}

M_driver_status_t M_driver_event_elapsed_time(M_driver_context_t context,
                                              M_driver_event_t start,
                                              M_driver_event_t end,
                                              float *elapsed_ms) {
  if (context == nullptr || start == nullptr || end == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return M_driver_event::elapsedTime(*context, start, end, elapsed_ms);
}
