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
#include "Support/LogicalResult.h"

namespace M::Driver::HAL {

Queue::Queue(ContextRef context, M_driver_queue_t handle)
    : context(std::move(context)), handle(handle) {}

Queue::~Queue() {
  context->device->driver->plugin.queue_destroy(context->handle, handle);
}

ErrorOr<BufferRef> Queue::allocBufferAsync(size_t byteSize) {
  M_driver_memory_t memory;
  if (M_driver_status_t status =
          context->device->driver->plugin.memory_alloc_async(handle, byteSize,
                                                             &memory))
    return Error(context->device->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow) Buffer(copyRCRef(this), memory, byteSize);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

ErrorOrSuccess Queue::copyToDevice(BufferRef dst, void *src, size_t byteSize) {
  if (M_driver_status_t status =
          context->device->driver->plugin.queue_copy_to_device(
              handle, dst->handle, src, byteSize))
    return Error(context->device->driver->getStatusMessage(status));
  return success();
}

ErrorOrSuccess Queue::copyFromDevice(void *dst, BufferRef src,
                                     size_t byteSize) {
  if (M_driver_status_t status =
          context->device->driver->plugin.queue_copy_from_device(
              handle, dst, src->handle, byteSize))
    return Error(context->device->driver->getStatusMessage(status));
  return success();
}

ErrorOrSuccess Queue::setMemory(BufferRef dst, uint64_t value,
                                size_t valueSize) {
  if (M_driver_status_t status =
          context->device->driver->plugin.queue_set_memory(
              handle, dst->handle, dst->byteSize, value, valueSize))
    return Error(context->device->driver->getStatusMessage(status));
  return success();
}

ErrorOrSuccess Queue::synchronize() {
  if (M_driver_status_t status =
          context->device->driver->plugin.queue_synchronize(handle))
    return Error(context->device->driver->getStatusMessage(status));
  return success();
}

} // namespace M::Driver::HAL
