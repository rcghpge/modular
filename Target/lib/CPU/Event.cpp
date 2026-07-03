//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements events for the CPU HAL plugin, mirroring
// CpuDeviceContext::eventCreate and CpuDeviceEvent.
//
//===----------------------------------------------------------------------===//

#include "Event.h"
#include "Common.h"
#include "Context.h"
#include "HAL/M_driver_event.h"
#include "MLRT/AsyncRT/Runtime/Algorithms.h"

#include <new>

M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_flags_t flags,
                                        M_driver_event_t *event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  // CPU events are always host-visible; reject only unknown flag bits.
  if (flags & ~M_DRIVER_EVENT_FLAGS_ALL)
    return M_DRIVER_STATUS_INVALID_ARG;

  // Events are ready by default until they get recorded.
  M_driver_event_t result = new (std::nothrow)
      M_driver_event(context->cpuDevice, context->cpu().getReadyChain().copy());
  if (result == nullptr)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  *event = result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  delete event;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  event->cpuDevice->getWorkQueue()->await({&event->value, 1});
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_is_event_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  if (is_ready == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_ready = event->value.isReady();
  return M_DRIVER_STATUS_SUCCESS;
}
