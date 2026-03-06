//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Driver.h"
#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
#include <memory>

using M::Error;

namespace M::Driver::HAL {

StringRef Driver::getStatusMessage(M_driver_status_t status) {
  if (plugin.status_message) {
    const char *message = nullptr;
    plugin.status_message(status, &message);
    return message;
  }
  return "(plugin not loaded)";
}

ErrorOr<DriverRef> Driver::create(llvm::StringRef pluginSpec) {
  auto driver = std::unique_ptr<Driver>(new (std::nothrow) Driver);
  if (!driver)
    return Error("Failed to allocate memory for Driver object");

  size_t atSignPos = pluginSpec.find('@');
  if (atSignPos != std::string::npos) {
    driver->plugin.name = pluginSpec.substr(0, atSignPos).str();
    driver->plugin.soPath = pluginSpec.substr(atSignPos + 1).str();
  } else {
    driver->plugin.name = pluginSpec.str();
    driver->plugin.soPath = pluginSpec.str();
  }
  driver->plugin.soHandle = dlopen(driver->plugin.soPath.c_str(), RTLD_NOW);
  if (driver->plugin.soHandle == nullptr)
    return Error("Failed to open shared object: " +
                 llvm::Twine(driver->plugin.soPath));

// Load function pointers from the shared object.
#define M_DRIVER_HAL_SYMBOL(fn)                                                \
  {                                                                            \
    std::string symbol = "M_driver_" + std::string(#fn);                       \
    driver->plugin.fn = (M_driver_##fn##_func_ptr_t)dlsym(                     \
        driver->plugin.soHandle, symbol.c_str());                              \
    if (driver->plugin.fn == nullptr)                                          \
      return Error("Failed to find symbol " + llvm::Twine(symbol) + " in " +   \
                   llvm::Twine(driver->plugin.soPath));                        \
  }
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  if (M_driver_status_t status = driver->plugin.create(&driver->handle))
    return Error("Failed to initialize driver plugin from " +
                 driver->plugin.soPath + ": " +
                 driver->getStatusMessage(status));

  // Pre-size data structures for subsequent lazy initialization.
  int64_t numDevices = 0;
  if (M_driver_status_t status =
          driver->plugin.device_count(driver->handle, &numDevices))
    return Error(
        "Failed to get device count during driver implementation setup: " +
        driver->getStatusMessage(status));
  driver->devices.resize(numDevices);

  return DriverRef::take(driver.release());
}

Driver::~Driver() {
  devices.clear();

  plugin.destroy(handle);

#define M_DRIVER_HAL_SYMBOL(name) plugin.name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  if (plugin.soHandle != nullptr) {
    dlclose(plugin.soHandle);
    plugin.soHandle = nullptr;
  }
}

ErrorOr<llvm::StringRef> Driver::name() const { return plugin.name; }

ErrorOr<llvm::StringRef> Driver::pluginPath() const { return plugin.soPath; }

ErrorOr<int64_t> Driver::deviceCount() const { return devices.size(); }

ErrorOr<DeviceRef> Driver::device(int64_t id) {
  // Validate input
  if (id < 0 || id >= int64_t(devices.size()))
    return Error("Invalid device ID - " + llvm::Twine(id) +
                 " - is out of range [0, " + llvm::Twine(devices.size()) +
                 ").");

  // Lazy initialization
  if (!devices[id]) {
    M_driver_device_t deviceHandle = nullptr;
    if (M_driver_status_t status = plugin.device_get(handle, id, &deviceHandle))
      return Error("Failed to get device #" + llvm::Twine(id) + ": " +
                   getStatusMessage(status));

    Device *device = new (std::nothrow)
        Device(this, deviceHandle, id, "Device #" + std::to_string(id));
    if (!device)
      return Error("Failed to allocate memory for Device object");

    devices[id] = DeviceRef::take(device);
  }

  return DeviceRef::copy(devices[id].getPointer());
}

ErrorOr<StringRef> Driver::property(const char *name) {
  const char *value;
  if (M_driver_status_t status = plugin.property(handle, name, &value))
    return Error("Failed to get driver property: " + getStatusMessage(status));
  return value;
}

} // namespace M::Driver::HAL
