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

#ifndef MLRT_DRIVER_HAL_BUFFER_H
#define MLRT_DRIVER_HAL_BUFFER_H

#include "MLRT/Driver/HAL/Queue.h"
#include "Support/RCRef.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {

class Context;
using ContextRef = RCRef<Context>;

class Queue;
using QueueRef = RCRef<Queue>;

class Buffer : public ReferenceCounted<Buffer> {
public:
  enum class AllocMode : uint8_t {
    kPinned = 1,
    kSync = 2,
    kAsync = 3,
  };

  ~Buffer();

private:
  explicit Buffer(Context *context, AllocMode mode, void *memory,
                  size_t byteSize)
      : context(context), queue(nullptr), memory(memory), byteSize(byteSize),
        mode(mode) {}
  explicit Buffer(Queue *queue, void *memory, size_t byteSize)
      : context(queue->context), queue(queue), memory(memory),
        byteSize(byteSize), mode(AllocMode::kAsync) {}

  Context *context;
  Queue *queue;
  void *memory;
  size_t byteSize;
  AllocMode mode;

  friend Context;
  friend Queue;
};

using BufferRef = RCRef<Buffer>;

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_BUFFER_H
