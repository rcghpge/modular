//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Function.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Function::Function(ContextRef context, M_driver_function_t handle)
    : context(std::move(context)), handle(handle) {}

Function::~Function() {
  context->device->driver->plugin.function_unload(context->handle, handle);
}

} // namespace M::Driver::HAL
