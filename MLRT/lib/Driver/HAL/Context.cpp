//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Context::~Context() {
  device_->driver->plugin.context_destroy(device_->driver->handle, handle);
}

ErrorOr<BufferRef> Context::allocBuffer(size_t byteSize,
                                        Buffer::AllocMode mode) {
  void *memory;
  if (M_driver_status_t status = device_->driver->plugin.memory_alloc(
          handle, static_cast<M_driver_alloc_mode_t>(mode), byteSize, &memory))
    return Error(device_->driver->getStatusMessage(status));
  Buffer *buffer = new (std::nothrow) Buffer(this, memory);
  if (!buffer)
    return Error("Failed to allocate memory for Buffer object");
  return BufferRef::take(buffer);
}

} // namespace M::Driver::HAL
