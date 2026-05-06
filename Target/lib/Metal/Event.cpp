//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Event.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_api.h"

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

DriverResult<M_driver_event_t> M_driver_event::create(M_driver_context &ctx) {
  MTL::SharedEvent *rawEvent = ctx.device->mDevice->newSharedEvent();
  if (rawEvent == nullptr)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  M_driver_event_t event = new (std::nothrow) M_driver_event();
  if (event == nullptr) {
    rawEvent->release();
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  }
  event->sharedEvent = NS::TransferPtr(rawEvent);
  return event;
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  event->sharedEvent->waitUntilSignaledValue(1, UINT64_MAX);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  *is_ready = (event->sharedEvent->signaledValue() >= 1);
  return M_DRIVER_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// C API dispatch
// ---------------------------------------------------------------------------

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  DriverResult<M_driver_event_t> result = M_driver_event::create(*context);
  if (!result)
    return result.error();
  *event = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::destroy(*context, event);
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::synchronize(*context, event);
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return M_driver_event::isReady(*context, event, is_ready);
}
