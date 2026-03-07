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

#ifndef MLRT_DRIVER_HAL_QUEUE_H
#define MLRT_DRIVER_HAL_QUEUE_H

#include "HAL/M_driver_api.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;

class Queue;
using QueueRef = RCRef<Queue>;

class Queue : public ReferenceCounted<Queue> {
public:
  ~Queue();

private:
  explicit Queue(Context *context, M_driver_queue_t handle)
      : context(context), handle(handle) {}

  Context *context;
  M_driver_queue_t handle;

  friend class Context;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_QUEUE_H
