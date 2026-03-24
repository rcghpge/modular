//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Device::Device(DriverRef driver, M_driver_device_t handle, int64_t id,
               std::string name)
    : driver(std::move(driver)), handle(handle), id(id), name(name) {}

Device::~Device() {}

ErrorOr<StringRef> Device::getName() const { return name; }

ErrorOr<int64_t> Device::getId() const { return id; }

ErrorOr<ContextRef> Device::createContext() {
  M_driver_context_t contextHandle;
  if (M_driver_status_t status =
          driver->plugin.context_create(handle, &contextHandle)) {
    return Error(driver->getStatusMessage(status));
  }
  Context *context = new (std::nothrow) Context(copyRCRef(this), contextHandle);
  if (context == nullptr) {
    return Error("Failed to allocate memory for Context object");
  }
  return ContextRef::take(context);
}

} // namespace M::Driver::HAL
