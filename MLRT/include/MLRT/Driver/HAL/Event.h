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
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;
using ContextRef = RCRef<Context>;

class Event : public ReferenceCounted<Event> {
public:
  ~Event();

  ErrorOrSuccess synchronize();

  ErrorOr<bool> isReady();

private:
  explicit Event(ContextRef context, M_driver_event_t handle,
                 M_driver_event_flags_t flags);

  ContextRef context;
  M_driver_event_t handle;
  // Remembered so the destructor can return the handle to the right cache
  // bucket on Context.
  M_driver_event_flags_t flags;

  friend Context;
  friend class Queue;
};

using EventRef = RCRef<Event>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_EVENT_H
