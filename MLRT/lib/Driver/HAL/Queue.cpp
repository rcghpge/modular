//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Queue.h"
#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Queue::~Queue() {
  context->device->driver->plugin.queue_destroy(context->handle, handle);
}

ErrorOr<BufferRef> Queue::allocBufferAsync(size_t byteSize) {
  void *memory;
  if (M_driver_status_t status =
          context->device->driver->plugin.memory_alloc_async(handle, byteSize,
                                                             &memory))
    return Error(context->device->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow) Buffer(this, memory);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

} // namespace M::Driver::HAL
