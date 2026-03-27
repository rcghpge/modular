//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_EVENT_H
#define TARGET_LIB_METAL_EVENT_H

#include "Common.h"
#include "MetalCpp.h"

#include <condition_variable>
#include <mutex>

struct M_driver_context;

struct M_driver_event {
  NS::SharedPtr<MTL::Event> mEvent;
  uint64_t targetValue;

  // CPU-side synchronization state. MTL::Event has no CPU-visible API, so we
  // track readiness ourselves via a command-buffer completion handler.
  std::mutex mutex;
  std::condition_variable cv;
  bool ready;

  M_driver_event(NS::SharedPtr<MTL::Event> mEvent, uint64_t targetValue)
      : mEvent(std::move(mEvent)), targetValue(targetValue), ready(true) {}

  [[nodiscard]] static DriverResult<M_driver_event_t>
  create(M_driver_context &ctx);

  static M_driver_status_t destroy(M_driver_context &ctx,
                                   M_driver_event_t event);

  static M_driver_status_t synchronize(M_driver_context &ctx,
                                       M_driver_event_t event);

  static M_driver_status_t isReady(M_driver_context &ctx,
                                   M_driver_event_t event, bool *is_ready);
};

#endif // TARGET_LIB_METAL_EVENT_H
