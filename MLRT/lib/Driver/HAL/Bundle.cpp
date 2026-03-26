//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Bundle.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Bundle::~Bundle() {
  context->device->driver->plugin.bundle_unload(context->handle, handle);
}

} // namespace M::Driver::HAL
