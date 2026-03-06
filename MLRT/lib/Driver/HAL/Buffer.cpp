//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Buffer.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Buffer::~Buffer() {
  context->device_->driver->plugin.memory_free(context->handle, memory);
}

} // namespace M::Driver::HAL
