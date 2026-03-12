//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides C++ bindings over the Hardware Abstraction Layer (HAL) C
// interface in SDK/Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_EVENT_H
#define MLRT_DRIVER_HAL_EVENT_H

#include "HAL/M_driver_api.h"
#include "Support/ErrorOr.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;

class Event : public ReferenceCounted<Event> {
public:
  ~Event();

  ErrorOrSuccess synchronize();

  ErrorOr<bool> isReady();

private:
  explicit Event(Context *context, M_driver_event_t handle,
                 bool enableTiming = false)
      : context(context), handle(handle), enableTiming(enableTiming) {}

  Context *context;
  M_driver_event_t handle;
  bool enableTiming;

  friend Context;
};

using EventRef = RCRef<Event>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_EVENT_H
