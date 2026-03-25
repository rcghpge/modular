//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Error.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_memory_t> result = context->allocPinned(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_memory_alloc_sync([[maybe_unused]] M_driver_context_t context,
                           [[maybe_unused]] uint64_t size,
                           [[maybe_unused]] M_driver_memory_t *memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (memory == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  DriverResult<M_driver_memory_t> result = context->allocSync(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->freePinned(memory);
}

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  return context->freeSync(memory);
}

M_driver_status_t
M_driver_memory_alloc_async([[maybe_unused]] M_driver_queue_t queue,
                            [[maybe_unused]] uint64_t size,
                            [[maybe_unused]] M_driver_memory_t *memory) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

M_driver_status_t
M_driver_memory_free_async([[maybe_unused]] M_driver_queue_t queue,
                           [[maybe_unused]] M_driver_memory_t memory) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
