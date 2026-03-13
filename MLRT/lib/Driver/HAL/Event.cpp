//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Event.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Event::~Event() {
  context->device->driver->plugin.event_destroy(context->handle, handle);
}

ErrorOrSuccess Event::synchronize() {
  if (M_driver_status_t status =
          context->device->driver->plugin.event_synchronize(context->handle,
                                                            handle))
    return Error(context->device->driver->getStatusMessage(status));
  return success();
}

ErrorOr<bool> Event::isReady() {
  bool ready = false;
  if (M_driver_status_t status = context->device->driver->plugin.event_is_ready(
          context->handle, handle, &ready))
    return Error(context->device->driver->getStatusMessage(status));
  return ready;
}

} // namespace M::Driver::HAL
