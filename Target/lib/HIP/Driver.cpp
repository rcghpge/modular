//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  DriverResult<HIPApi> api = HIPApi::load();
  if (!api)
    return tl::unexpected(api.error());

  if (api->hipInit(0) != hipSuccess) {
    // dropping api automatically unloads
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  }

  return new M_driver_driver(std::move(*api));
}

DriverResult<int64_t> M_driver_driver::deviceCount() {
  int countAsInt = 0;
  if (rawHipApi.hipGetDeviceCount(&countAsInt) != hipSuccess)
    return tl::unexpected(M_DRIVER_STATUS_NO_DEVICES);
  return static_cast<int64_t>(countAsInt);
}
