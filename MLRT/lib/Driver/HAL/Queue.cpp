//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/HAL/Queue.h"
#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Driver.h"

namespace M::Driver::HAL {

Queue::~Queue() {
  context->device_->driver->plugin.queue_destroy(context->handle, handle);
}

} // namespace M::Driver::HAL
