//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Memory.h"
#include "Context.h"
#include "HAL/M_driver_memory.h"

M_driver_status_t M_driver_memory::getProperty(llvm::StringRef name,
                                               void *value) {
  if (name == M_DRIVER_MEMORY_PROPERTY_ADDRESS) {
    *static_cast<uint64_t *>(value) = mBuffer->gpuAddress();
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME;
}

M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (size == 0) {
    *memory = nullptr;
    return M_DRIVER_STATUS_SUCCESS;
  }
  auto result = context->alloc(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (size == 0) {
    *memory = nullptr;
    return M_DRIVER_STATUS_SUCCESS;
  }
  auto result = context->alloc(size);
  if (!result)
    return result.error();
  *memory = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_SUCCESS;
  return context->free(memory);
}

M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_SUCCESS;
  return context->free(memory);
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

M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value) {
  return memory->getProperty(property_name, value);
}
