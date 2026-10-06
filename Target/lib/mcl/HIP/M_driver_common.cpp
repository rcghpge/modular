//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"

M_driver_status_t
M_driver_status_message([[maybe_unused]] M_driver_driver_t driver,
                        M_driver_status_t status, char *message_buffer,
                        int64_t message_buffer_size) {
  if (driver == nullptr)
    return M_DRIVER_STATUS_BAD_DRIVER;
  return driver->status.getMessage(status, message_buffer, message_buffer_size);
}
