//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Error.h"
#include "HAL/M_driver_api.h"
#include "Queue.h"

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;

  DriverResult<M_driver_queue_t> result = M_driver_queue::create(*context);
  if (!result)
    return result.error();
  *queue = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return M_driver_queue::destroy(*context, queue);
}
