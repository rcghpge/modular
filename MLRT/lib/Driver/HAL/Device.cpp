//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Device.h"
#include "ContextImpl.h"
#include "DeviceImpl.h"
#include "MLRT/Driver/HAL/Context.h"

namespace M::Driver::HAL {

Device::~Device() {
  assert(impl_ != nullptr);
  delete static_cast<DeviceImpl *>(impl_);
  impl_ = nullptr;
}

ErrorOr<std::string const &> Device::name() const {
  assert(impl_ != nullptr);
  return static_cast<DeviceImpl *>(impl_)->name;
}

ErrorOr<int64_t> Device::id() const {
  assert(impl_ != nullptr);
  return static_cast<DeviceImpl *>(impl_)->id;
}

ErrorOr<ContextRef> Device::createContext() const {
  ContextImpl *impl = new (std::nothrow) ContextImpl;
  if (impl == nullptr)
    return Error("Failed to allocate memory for Context implementation");
  Context *context = new (std::nothrow) Context(impl);
  if (context == nullptr) {
    delete impl;
    return Error("Failed to allocate memory for Context object");
  }
  return ContextRef::take(context);
}

} // namespace M::Driver::HAL
