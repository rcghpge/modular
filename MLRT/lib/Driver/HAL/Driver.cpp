//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "MLRT/Driver/HAL/Driver.h"
#include "DeviceImpl.h"
#include "DriverImpl.h"

#include "HAL/M_driver_api.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <dlfcn.h>
#include <vector>

using M::Error;

#define RETURN_IF_FAILED(expr, message)                                        \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::string const &soPath = impl->soPath;                                \
      return Error(message + llvm::Twine(" for driver plugin ") + soPath);     \
    }                                                                          \
  } while (false)

// Helper macro - frequently used pattern
#define CALL_M_DRIVER(action, obj, method_call)                                \
  do {                                                                         \
    auto _impl = obj;                                                          \
    M_driver_status_t status = _impl->method_call;                             \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      const char *errorMessage = nullptr;                                      \
      (void)_impl->status_message(status, &errorMessage);                      \
      return Error("Failed to " + llvm::Twine(action) +                        \
                   " from driver plugin " + _impl->soPath + ": " +             \
                   llvm::Twine(errorMessage));                                 \
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
    soHandle = nullptr;
  }
}

ErrorOrSuccess DriverImpl::setup(llvm::StringRef pluginSpec) {
  size_t atSignPos = pluginSpec.find('@');
  if (atSignPos != std::string::npos) {
    this->name = pluginSpec.substr(0, atSignPos).str();
    this->soPath = pluginSpec.substr(atSignPos + 1).str();
  } else {
    this->name = pluginSpec.str();
    this->soPath = pluginSpec.str();
  }
  this->soHandle = dlopen(this->soPath.c_str(), RTLD_NOW);
  if (this->soHandle == nullptr)
    return Error("Failed to open shared object: " + llvm::Twine(soPath));

// Load function pointers from the shared object.
#define M_DRIVER_HAL_SYMBOL(fn)                                                \
  {                                                                            \
    std::string symbol = "M_driver_" + std::string(#fn);                       \
    this->fn = (M_driver_##fn##_func_ptr_t)dlsym(soHandle, symbol.c_str());    \
    if (this->fn == nullptr) {                                                 \
      this->reset();                                                           \
      return Error("Failed to find symbol " + llvm::Twine(symbol) + " in " +   \
                   llvm::Twine(soPath));                                       \
    }                                                                          \
  }
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  CALL_M_DRIVER("initialize driver plugin", this, create(&handle));

  // Pre-size data structures for subsequent lazy initialization.
  int64_t numDevices = 0;
  CALL_M_DRIVER("get device count during driver implementation setup", this,
                device_count(handle, &numDevices));
  devices.resize(numDevices);
  return success();
}

// ----------------------------------------------------------------------------
// Public HAL implementation
// ----------------------------------------------------------------------------

ErrorOr<DriverRef> Driver::create(llvm::StringRef pluginSpec) {
  auto impl = new (std::nothrow) DriverImpl;
  if (impl == nullptr)
    return Error("Failed to allocate memory for Driver implementation");

  ErrorOrSuccess status = impl->setup(pluginSpec);
  if (status.isError()) {
    delete impl;
    return status.takeError();
  }
  Driver *driver = new (std::nothrow) Driver(impl);
  if (driver == nullptr) {
    delete impl;
    return Error("Failed to allocate memory for Driver object");
  }
  return DriverRef::take(driver);
}

Driver::~Driver() {
  assert(impl != nullptr);
  // Ignore return value from API call,which is disallowed in DTORs.
  [[maybe_unused]] auto _ = [&]() {
    CALL_M_DRIVER("finalize driver plugin", impl, destroy(impl->handle));
    return Error("_");
  }();
  delete static_cast<DriverImpl *>(impl);
  impl = nullptr;
}

ErrorOr<llvm::StringRef> Driver::name() const {
  assert(impl != nullptr);
  return impl->name;
}

ErrorOr<llvm::StringRef> Driver::pluginPath() const {
  assert(impl != nullptr);
  return impl->soPath;
}

ErrorOr<int64_t> Driver::deviceCount() const {
  assert(impl != nullptr);
  int64_t numDevices = 0;
  CALL_M_DRIVER("get device count", impl,
                device_count(impl->handle, &numDevices));
  return numDevices;
}

ErrorOr<DeviceRef> Driver::device(int64_t id) {
  // Validate input
  assert(impl != nullptr);

  if (id < 0 || id >= int64_t(impl->devices.size())) {
    return Error("Invalid device ID - " + llvm::Twine(id) +
                 " - is out of range [0, " + llvm::Twine(impl->devices.size()) +
                 ").");
  }

  // Lazy initialization
  if (!impl->devices[id]) {
    M_driver_device_t deviceHandle = 0xDEADBEEFFEEDFACEuLL;
    CALL_M_DRIVER("get device #" + std::to_string(id), impl,
                  device_get(impl->handle, id, &deviceHandle));

    DeviceImpl *deviceImpl = nullptr;
    RETURN_IF_FAILED(deviceImpl =
                         new (std::nothrow) DeviceImpl(impl, deviceHandle),
                     "Failed to allocate memory for Device implementation");
    deviceImpl->handle = deviceHandle;
    deviceImpl->id = id;
    deviceImpl->name = "Device #" + std::to_string(id);

    Device *device = nullptr;
    RETURN_IF_FAILED(device = new (std::nothrow) Device(deviceImpl),
                     "Failed to allocate memory for Device object");
    impl->devices[id] = DeviceRef::take(device);
  }

  return DeviceRef::copy(impl->devices[id].getPointer());
}

} // namespace M::Driver::HAL
