//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Driver.h"
#include "Target/lib/Support/StandardStatus.h"
#include "Target/lib/Support/Utils.h"
#include <iostream>

M_driver_status_t M_driver_status_message(M_driver_driver_t driver,
                                          M_driver_status_t status,
                                          char *message_buffer,
                                          int64_t message_buffer_size) {
  if (message_buffer_size <= 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  if (status > 0)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;
  else if (status > -M_DRIVER_NUM_STANDARD_STATUS_CODES) {
    copyStatusMessage(message_buffer, message_buffer_size,
                      standardStatusMessages[-status]);
    return M_DRIVER_STATUS_SUCCESS;
  } else if (status > M_DRIVER_STATUS_CUSTOM_CODES_BASE)
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;

  auto iter = driver->customStatusMessages.find(status);
  if (iter != driver->customStatusMessages.end()) {
    copyStatusMessage(message_buffer, message_buffer_size, iter->second);
    if (status <=
        M_DRIVER_STATUS_CUSTOM_CODES_BASE - M_DRIVER_NUM_CUSTOM_STATUS_CODES)
      driver->customStatusMessages.erase(iter);
  } else
    return M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE;

  return M_DRIVER_STATUS_SUCCESS;
}
