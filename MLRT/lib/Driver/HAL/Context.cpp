//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Context.h"
#include "ContextImpl.h"
#include "DeviceImpl.h"
#include "MLRT/lib/Driver/HAL/DriverImpl.h"

namespace M::Driver::HAL {

Context::~Context() {
  impl->device->driver->context_destroy(impl->device->driver->handle,
                                        impl->handle);
  delete impl;
}

} // namespace M::Driver::HAL
