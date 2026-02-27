//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#ifndef MLRT_DRIVER_HAL_DRIVERIMPL_H
#define MLRT_DRIVER_HAL_DRIVERIMPL_H

#include "MLRT/Driver/HAL/Device.h"
#include "Support/ErrorOr.h"
#include "Support/RCRef.h"

#include "HAL/M_driver_api.h"

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace M::Driver::HAL {

using DeviceRef = RCRef<Device>;

struct DriverImpl {
  ~DriverImpl() { reset(); }
  void reset();

  // pluginSpec: <name>@<path>
  // name: any human-readable string / mnemonic for this driver
  // path: absolute path to the shared object, including the file name
  // Eg. CUDA@/usr/lib/awesome_hw.so
  ErrorOrSuccess setup(llvm::StringRef const pluginSpec);

  std::string name = std::string();
  std::string soPath = std::string();
  void *soHandle = nullptr;

#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL

  M_driver_driver_t handle;
  std::vector<DeviceRef> devices;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DRIVERIMPL_H
