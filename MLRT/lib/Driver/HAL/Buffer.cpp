//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"
#include "MLRT/Driver/HAL/Queue.h"

namespace M::Driver::HAL {

Buffer::Buffer(ContextRef context, AllocMode mode, M_driver_memory_t memory,
               size_t byteSize)
    : context(context), queue(), handle(memory), byteSize(byteSize),
      mode(mode) {}

Buffer::Buffer(QueueRef queue, M_driver_memory_t memory, size_t byteSize)
    : context(queue->context), queue(queue), handle(memory), byteSize(byteSize),
      mode(AllocMode::kAsync) {}

Buffer::~Buffer() {
  switch (mode) {
  case AllocMode::kPinned:
    context->device->driver->plugin.memory_free_pinned(context->handle, handle);
    break;
  case AllocMode::kSync:
    context->device->driver->plugin.memory_free_sync(context->handle, handle);
    break;
  case AllocMode::kAsync:
    context->device->driver->plugin.memory_free_async(queue->handle, handle);
    break;
  }
}

} // namespace M::Driver::HAL
