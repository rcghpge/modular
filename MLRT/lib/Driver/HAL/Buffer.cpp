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

Buffer::Buffer(ContextRef context, AllocMode mode, void *memory,
               size_t byteSize)
    : context(std::move(context)), memory(memory), byteSize(byteSize),
      mode(mode) {}

Buffer::Buffer(QueueRef queue, void *memory, size_t byteSize)
    : context(queue->context.copy()), queue(std::move(queue)), memory(memory),
      byteSize(byteSize), mode(AllocMode::kAsync) {}

Buffer::~Buffer() {
  switch (mode) {
  case AllocMode::kPinned:
    context->device->driver->plugin.memory_free_pinned(context->handle, memory);
    break;
  case AllocMode::kSync:
    context->device->driver->plugin.memory_free_sync(context->handle, memory);
    break;
  case AllocMode::kAsync:
    context->device->driver->plugin.memory_free_async(queue->handle, memory);
    break;
  }
}

} // namespace M::Driver::HAL
