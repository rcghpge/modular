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
