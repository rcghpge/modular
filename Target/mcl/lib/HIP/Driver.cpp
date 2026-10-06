//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

#include <memory>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_driver_t> M_driver_driver::create() {
  DriverResult<HIPApi> api = HIPApi::load();
  if (!api)
    return tl::unexpected(api.error());

  auto driver = std::make_unique<M_driver_driver>(std::move(*api));

  if (hipError_t r = driver->hip().hipInit(0); r != hipSuccess)
    driver->initStatus = hipErrorStatus(driver.get(), r, "hipInit");

  return driver.release();
}

DriverResult<int64_t> M_driver_driver::deviceCount() {
  int countAsInt = 0;
  if (hipError_t r = rawHipApi.hipGetDeviceCount(&countAsInt); r != hipSuccess)
    return tl::unexpected(hipErrorStatus(this, r, "hipGetDeviceCount"));
  return static_cast<int64_t>(countAsInt);
}
