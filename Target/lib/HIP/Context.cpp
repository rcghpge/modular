//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  return new M_driver_context(dev);
}
