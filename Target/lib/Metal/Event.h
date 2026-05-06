//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_EVENT_H
#define TARGET_LIB_METAL_EVENT_H

#include "Common.h"
#include "MetalCpp.h"

struct M_driver_context;

struct M_driver_event {
  NS::SharedPtr<MTL::SharedEvent> sharedEvent;

  M_driver_event() = default;

  explicit M_driver_event(NS::SharedPtr<MTL::SharedEvent> sharedEvent)
      : sharedEvent(std::move(sharedEvent)) {}

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
