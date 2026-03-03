//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Device.h"
#include "ContextImpl.h"
#include "DeviceImpl.h"
#include "DriverImpl.h"
#include "HAL/M_driver_context.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Device::~Device() {
  assert(impl != nullptr);
  delete impl;
  impl = nullptr;
}

ErrorOr<std::string const &> Device::name() const {
  assert(impl != nullptr);
  return impl->name;
}

ErrorOr<int64_t> Device::id() const {
  assert(impl != nullptr);
  return impl->id;
}

ErrorOr<ContextRef> Device::createContext() const {
  M_driver_context_t handle;
  if (M_driver_status_t status = impl->driver->context_create(
          impl->driver->handle, impl->handle, &handle)) {
    const char *message;
    impl->driver->status_message(status, &message);
    return Error(message);
  }
  ContextImpl *contextImpl = new (std::nothrow) ContextImpl(impl, handle);
  if (contextImpl == nullptr)
    return Error("Failed to allocate memory for Context implementation");
  Context *context = new (std::nothrow) Context(contextImpl);
  if (context == nullptr) {
    delete contextImpl;
    return Error("Failed to allocate memory for Context object");
  }
  return ContextRef::take(context);
}

} // namespace M::Driver::HAL
