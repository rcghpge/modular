//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Event.h"
#include "Context.h"
#include "Device.h"

DriverResult<M_driver_event_t> M_driver_event::create(M_driver_context &ctx) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  hipEvent_t event;
  hipError_t result =
      hip.hipEventCreateWithFlags(&event, hipEventDisableTiming);
  if (result != hipSuccess)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_event(event);
}

M_driver_status_t M_driver_event::destroy(M_driver_context &ctx,
                                          M_driver_event_t event) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipEventDestroy(event->event);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::synchronize(M_driver_context &ctx,
                                              M_driver_event_t event) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipEventSynchronize(event->event);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event::isReady(M_driver_context &ctx,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipEventQuery(event->event);
  if (result == hipSuccess) {
    *is_ready = true;
    return M_DRIVER_STATUS_SUCCESS;
  }
  if (result == hipErrorNotReady) {
    *is_ready = false;
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_ERROR;
}
