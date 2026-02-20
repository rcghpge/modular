//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "MLRT/Driver/HAL/Driver.hpp"
#include "DeviceImpl.hpp"
#include "DriverImpl.hpp"

#include "HAL/M_driver_api.h"

#include <dlfcn.h>
#include <vector>

using M::Error;

#define RETURN_IF_FAILED(expr, message)                                        \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::string const &soPath = static_cast<DriverImpl *>(impl)->soPath;     \
      return Error(message + std::string(" for driver plugin ") + soPath);     \
    }                                                                          \
  } while (false)

// Helper macro - frequently used pattern
#define CALL_M_DRIVER(action, obj, method_call)                                \
  do {                                                                         \
    auto _impl = static_cast<DriverImpl *>(obj);                               \
    M_driver_status_t status = _impl->method_call;                             \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      const char *errorMessage = nullptr;                                      \
      (void)_impl->status_message(status, &errorMessage);                      \
      return Error("Failed to " + std::string(action) +                        \
                   " from driver plugin " + _impl->soPath + ": " +             \
                   std::string(errorMessage));                                 \
    }                                                                          \
  } while (false)

namespace M::Driver::HAL {

// ----------------------------------------------------------------------------
// Private driver implementation.
// ----------------------------------------------------------------------------

void DriverImpl::reset() {
  devices.clear();

#define M_DRIVER_HAL_SYMBOL(name) name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  if (soHandle != nullptr) {
    dlclose(soHandle);
  }
  soHandle = nullptr;
}

#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)

ErrorOrSuccess DriverImpl::setup(std::string const &pluginSpec) {
  size_t atSignPos = pluginSpec.find('@');
  if (atSignPos != std::string::npos) {
    this->name = pluginSpec.substr(0, atSignPos);
    this->soPath = pluginSpec.substr(atSignPos + 1);
  } else {
    this->name = pluginSpec;
    this->soPath = pluginSpec;
  }
  this->soHandle = dlopen(this->soPath.c_str(), RTLD_NOW);
  if (this->soHandle == nullptr) {
    return Error("Failed to open shared object: " + soPath);
  }
// Load function pointers from the shared object.
#define M_DRIVER_HAL_SYMBOL(fn)                                                \
  {                                                                            \
    std::string const symbol = "M_driver_" + std::string(#fn);                 \
    this->fn = (M_driver_##fn##_func_ptr_t)dlsym(soHandle, symbol.c_str());    \
    if (this->fn == nullptr) {                                                 \
      this->reset();                                                           \
      return Error("Failed to find symbol " + symbol + " in " + soPath);       \
    }                                                                          \
  }
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  CALL_M_DRIVER("initialize driver plugin", this, initialize());

  // Pre-size data structures for subsequent lazy initialization.
  int64_t numDevices = 0;
  CALL_M_DRIVER("get device count during driver implementation setup", this,
                device_count(&numDevices));
  devices.resize(numDevices);
  return success();
}

// ----------------------------------------------------------------------------
// Public HAL implementation
// ----------------------------------------------------------------------------

ErrorOr<DriverRef> Driver::create(std::string const &soPath) {
  auto impl = new (std::nothrow) DriverImpl;
  if (impl == nullptr) {
    return Error("Failed to allocate memory for Driver implementation");
  }
  ErrorOrSuccess status = impl->setup(soPath);
  if (status.isError()) {
    delete impl;
    return Error("Failed to load driver plugin: " + soPath);
  }
  Driver *const driver = new (std::nothrow) Driver((void *)impl);
  if (driver == nullptr) {
    delete impl;
    return Error("Failed to allocate memory for Driver object");
  }
  return DriverRef::take(driver);
}

Driver::~Driver() {
  assert(impl_ != nullptr);
  // Wrap large macro in lambda in order to suppress return value,
  // which is disallowed in DTORs.
  (void)[&]() {
    CALL_M_DRIVER("finalize driver plugin", impl_, finalize());
    return Error("_");
  }
  ();
  delete static_cast<DriverImpl *>(impl_);
  impl_ = nullptr;
}

ErrorOr<std::string const &> Driver::name() const {
  assert(impl_ != nullptr);
  return static_cast<DriverImpl *>(impl_)->name;
}

ErrorOr<std::string const &> Driver::pluginPath() const {
  assert(impl_ != nullptr);
  return static_cast<DriverImpl *>(impl_)->soPath;
}

ErrorOr<int64_t> Driver::deviceCount() const {
  assert(impl_ != nullptr);
  int64_t numDevices = 0;
  CALL_M_DRIVER("get device count", impl_, device_count(&numDevices));
  return numDevices;
}

ErrorOr<DeviceRef> Driver::device(int64_t id) {
  // Validate input
  assert(impl_ != nullptr);

  auto impl = static_cast<DriverImpl *>(impl_);
  if (id < 0 || id >= int64_t(impl->devices.size())) {
    return Error("Invalid device ID - " + std::to_string(id) +
                 " - is out of range [0, " +
                 std::to_string(impl->devices.size()) + ").");
  }

  // Lazy initialization
  if (!impl->devices[id]) {
    M_driver_device_t deviceHandle = 0xDEADBEEFFEEDFACEuLL;
    CALL_M_DRIVER("get device #" + std::to_string(id), impl_,
                  device_get(id, &deviceHandle));

    DeviceImpl *deviceImpl = nullptr;
    RETURN_IF_FAILED(deviceImpl = new (std::nothrow) DeviceImpl,
                     "Failed to allocate memory for Device implementation");
    deviceImpl->handle = deviceHandle;
    deviceImpl->id = id;
    deviceImpl->name = "Device #" + std::to_string(id);

    Device *device = nullptr;
    RETURN_IF_FAILED(device = new (std::nothrow) Device((void *)deviceImpl),
                     "Failed to allocate memory for Device object");
    impl->devices[id] = DeviceRef::take(device);
  }

  return DeviceRef::copy(impl->devices[id].getPointer());
}

} // namespace M::Driver::HAL
