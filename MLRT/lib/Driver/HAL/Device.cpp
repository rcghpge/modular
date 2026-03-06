//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Device::~Device() {}

ErrorOr<StringRef> Device::name() const { return name_; }

ErrorOr<int64_t> Device::id() const { return id_; }

ErrorOr<ContextRef> Device::createContext() {
  M_driver_context_t contextHandle;
  if (M_driver_status_t status = driver->plugin.context_create(
          driver->handle, handle, &contextHandle)) {
    return Error(driver->getStatusMessage(status));
  }
  Context *context = new (std::nothrow) Context(this, contextHandle);
  if (context == nullptr) {
    return Error("Failed to allocate memory for Context object");
  }
  return ContextRef::take(context);
}

} // namespace M::Driver::HAL
