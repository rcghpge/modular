//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "BufferImpl.h"
#include "ContextImpl.h"
#include "DeviceImpl.h"
#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/lib/Driver/HAL/DriverImpl.h"

namespace M::Driver::HAL {

Context::~Context() {
  impl->device->driver->context_destroy(impl->device->driver->handle,
                                        impl->handle);
  delete impl;
}

ErrorOr<BufferRef> Context::allocBuffer(size_t byteSize,
                                        Buffer::AllocMode mode) {
  void *memory;
  if (M_driver_status_t status = impl->device->driver->memory_alloc(
          impl->handle, static_cast<M_driver_alloc_mode_t>(mode), byteSize,
          &memory)) {
    const char *message;
    impl->device->driver->status_message(status, &message);
    return Error(message);
  }
  BufferImpl *bufferImpl = new (std::nothrow) BufferImpl(impl, memory);
  if (!bufferImpl)
    return Error("Failed to allocate memory for Buffer implementation object");
  Buffer *buffer = new (std::nothrow) Buffer(bufferImpl);
  if (!buffer) {
    delete bufferImpl;
    return Error("Failed to allocate memory for Buffer object");
  }
  return BufferRef::take(buffer);
}

} // namespace M::Driver::HAL
