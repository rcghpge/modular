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
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;
using ContextRef = RCRef<Context>;

class Buffer;
using BufferRef = RCRef<Buffer>;

class Event;
using EventRef = RCRef<Event>;

class Queue;
using QueueRef = RCRef<Queue>;

class Queue : public ReferenceCounted<Queue> {
public:
  ~Queue();

  ErrorOr<BufferRef> allocBufferAsync(size_t byteSize);

  ErrorOrSuccess copyToDevice(BufferRef dst, void *src, size_t byteSize);
  ErrorOrSuccess copyFromDevice(void *dst, BufferRef src, size_t byteSize);
  ErrorOrSuccess copyIntraDevice(BufferRef dst, BufferRef src, size_t byteSize);
  /// Set every byte of \p dst to \p value.
  ErrorOrSuccess setMemory(BufferRef dst, uint8_t value);
  /// Fill \p dst with a repeated \p valueSize-byte \p value. \p valueSize must
  /// be one of 1, 2, 4, or 8; a \p valueSize of 1 is equivalent to setMemory.
  ErrorOrSuccess fillMemory(BufferRef dst, uint64_t value, size_t valueSize);
  /// Record a new event on this queue. The returned event will be signaled
  /// once all operations enqueued before this call have completed.
  ///
  /// \param flags Capability bitmask. Default `M_DRIVER_EVENT_FLAG_NONE`
  ///   is intra-GPU only; pass `M_DRIVER_EVENT_FLAG_CPU_VISIBLE` to enable
  ///   host-side `synchronize()` / `isReady()` on the returned event.
  ErrorOr<EventRef>
  recordEvent(M_driver_event_flags_t flags = M_DRIVER_EVENT_FLAG_NONE);
  /// Make this queue wait until the given events have been signaled.
  ErrorOrSuccess waitForEvents(M_driver_event_t *events, uint32_t numEvents);
  /// Make this queue wait until the given event has been signaled.
  ErrorOrSuccess waitForEvent(EventRef event);
  ErrorOrSuccess synchronize();

private:
  explicit Queue(ContextRef context, M_driver_queue_t handle);

  ContextRef context;
  M_driver_queue_t handle;

  friend class Context;
  friend class Buffer;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_QUEUE_H
