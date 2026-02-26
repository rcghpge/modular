//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "MLRT/Driver/HAL/Device.h"
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
  return Error("Not implemented");
}

} // namespace M::Driver::HAL
