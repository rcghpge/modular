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

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"

namespace M::Driver::HAL {
struct DriverImpl;

class Driver;
using DriverRef = RCRef<Driver>;

class Device;
using DeviceRef = RCRef<Device>;

class Context;

class Driver : public ReferenceCounted<Driver> {
public:
  /// Factory function to create a driver from a plugin shared object file.
  /// pluginSpec := name@path
  /// name := any human-readable string / mnemonic for this driver
  /// path := absolute path to the shared object, including the file name
  /// Eg. CUDA@/path/to/cuda_plugin.so
  static ErrorOr<DriverRef> create(llvm::StringRef pluginSpec);

  ~Driver();

  ErrorOr<llvm::StringRef> name() const;
  ErrorOr<llvm::StringRef> pluginPath() const;

  /// Retrieve the number of devices supported by this driver.
  ErrorOr<int64_t> deviceCount() const;

  /// Retrieve a device by its ID.
  /// The ID must be in the range [0, `deviceCount()`-1].
  ErrorOr<DeviceRef> device(int64_t id);

private:
  explicit Driver(DriverImpl *impl) : impl(impl) {}

  DriverImpl *impl;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DRIVER_H
