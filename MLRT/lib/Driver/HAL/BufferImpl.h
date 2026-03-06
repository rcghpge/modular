//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_BUFFERIMPL_H
#define MLRT_DRIVER_HAL_BUFFERIMPL_H

#include "ContextImpl.h"
#include "DeviceImpl.h"
#include "DriverImpl.h"
#include "HAL/M_driver_api.h"

namespace M::Driver::HAL {

struct ContextImpl;

struct BufferImpl {
  BufferImpl(ContextImpl *context, void *memory)
      : context(context), memory(memory) {}

  ~BufferImpl() {
    context->device->driver->memory_free(context->handle, memory);
  }

  ContextImpl *context;
  void *memory;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_BUFFERIMPL_H
