//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides C++ bindings over the Hardware Abstraction Layer (HAL) C
// interface in SDK/Driver.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_DRIVER_HAL_DRIVER_H
#define MLRT_DRIVER_HAL_DRIVER_H

#include "HAL/M_driver_api.h"
#include "MLRT/Driver/HAL/Device.h"
#include "Support/ReferenceCounted.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace M::Driver::HAL {
struct DriverImpl;

class Driver;
using DriverRef = RCRef<Driver>;

class Context;
class Buffer;

class Driver : public ReferenceCounted<Driver> {
public:
  /// Factory function to create a driver from a plugin shared object file.
  /// pluginSpec := name@path
  /// name := any human-readable string / mnemonic for this driver
  /// path := absolute path to the shared object, including the file name
  /// Eg. CUDA@/path/to/cuda_plugin.so
  static ErrorOr<DriverRef> create(llvm::StringRef pluginSpec);

  ~Driver();

  ErrorOr<StringRef> name() const;
  ErrorOr<StringRef> pluginPath() const;

  /// Retrieve the number of devices supported by this driver.
  ErrorOr<int64_t> deviceCount() const;

  /// Retrieve a device by its ID.
  /// The ID must be in the range [0, `deviceCount()`-1].
  ErrorOr<DeviceRef> device(int64_t id);

  ErrorOr<StringRef> property(const char *name);

private:
  explicit Driver() {}

  /// Helper to construct Error(getStatusMessage(status)) more easily.
  StringRef getStatusMessage(M_driver_status_t status);

  /// To avoid naming conllisions the plugin symbols are defined within the
  /// Plugin struct along with other plugin specific info.
  struct Plugin {
    std::string name;
    std::string soPath;
    void *soHandle = nullptr;

#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  } plugin;

  M_driver_driver_t handle;
  SmallVector<DeviceRef, 8> devices;

  friend Device;
  friend Context;
  friend Buffer;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DRIVER_H
