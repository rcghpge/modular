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
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::Device *device = ctx.device->mDevice;
  NS::SharedPtr<MTL::SharedEvent> event =
      NS::TransferPtr(device->newSharedEvent());
  if (!event)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_event(std::move(event), /*targetValue=*/0);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  event->event->waitUntilSignaledValue(event->targetValue, UINT64_MAX);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  *is_ready = event->event->signaledValue() >= event->targetValue;
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
