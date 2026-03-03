//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

namespace M::Driver::HAL {

struct DeviceImpl;

struct ContextImpl {
  ContextImpl(DeviceImpl *device, M_driver_context_t handle)
      : device(device), handle(handle) {}
  ~ContextImpl() { reset(); }

  void reset() { handle = 0xDEADBEEFFEEDFACEuLL; }

  DeviceImpl *device;
  M_driver_context_t handle;
};

} // namespace M::Driver::HAL
