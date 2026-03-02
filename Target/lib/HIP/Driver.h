//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_DRIVER_H
#define TARGET_LIB_HIP_DRIVER_H

#include "HAL/M_driver_common.h"

#include "HIPApi.h"

#include <cassert>

class Driver {
private:
  HIPApi rawHipApi;

public:
  M_driver_driver_t cast() { return reinterpret_cast<M_driver_driver_t>(this); }

  static Driver &from(M_driver_driver_t handle) {
    auto driverPtr = Driver::from_as_ptr(handle);
    return *driverPtr;
  }

  static Driver *from_as_ptr(M_driver_driver_t handle) {
    Driver *driverPtr = reinterpret_cast<Driver *>(handle);
    assert(driverPtr != nullptr);

    return driverPtr;
  }

  Driver(HIPApi &&rawHipApi) : rawHipApi(std::move(rawHipApi)) {}

  ~Driver() { rawHipApi.unload(); }

  HIPApi &hip() { return rawHipApi; }

  M_driver_status_t deviceCount(int64_t &count) {
    int countAsInt = 0;
    if (rawHipApi.hipGetDeviceCount(&countAsInt) != hipSuccess)
      return M_DRIVER_STATUS_NO_DEVICES;
    count = static_cast<int64_t>(countAsInt);
    return M_DRIVER_STATUS_SUCCESS;
  }
};

#endif // TARGET_LIB_HIP_DRIVER_H
