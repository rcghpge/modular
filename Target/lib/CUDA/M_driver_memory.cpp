//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_memory_alloc(M_driver_context_t context,
                                        M_driver_alloc_mode_t mode,
                                        uint64_t size, void **memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<void *> result = context->alloc(mode, size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_free(M_driver_context_t context,
                                       void *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->free(memory);
}
