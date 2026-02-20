#ifndef MLRT_DRIVER_HAL_DEVICEIMPL_HPP
#define MLRT_DRIVER_HAL_DEVICEIMPL_HPP

#include <string>
#include <cstdint>
#include "HAL/M_driver_api.h"

namespace M::Driver::HAL {

struct DeviceImpl {
  DeviceImpl() { reset(); }
  ~DeviceImpl() { reset(); }

  void reset() {
    handle = 0xDEADBEEFFEEDFACEuLL;
    name = "(no device)";
    id = -1;
  }

  M_driver_device_t handle;
  std::string name;
  int64_t id;
};

} // namespace M::Driver::HAL

#endif // MLRT_DRIVER_HAL_DEVICEIMPL_HPP
