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

class Queue;
using QueueRef = RCRef<Queue>;

class Queue : public ReferenceCounted<Queue> {
public:
  ~Queue();

  ErrorOr<BufferRef> allocBufferAsync(size_t byteSize);

  ErrorOrSuccess copyToDevice(BufferRef dst, void *src, size_t byteSize);
  ErrorOrSuccess copyFromDevice(void *dst, BufferRef src, size_t byteSize);
  ErrorOrSuccess setMemory(BufferRef dst, uint64_t value, size_t valueSize);
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
