//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Function.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Function::~Function() {
  context->device->driver->plugin.function_unload(
      context->device->driver->handle, context->handle, handle);
}

} // namespace M::Driver::HAL
