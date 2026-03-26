//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_MEMORY_H
#define M_DRIVER_MEMORY_H

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate pinned (page-locked) host memory for the given context.
 *
 * @param context (in) The context to create the memory for.
 * @param size (in) Size in bytes of the allocation.
 * @param memory (out) Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_CONTEXT if the context is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_pinned(M_driver_context_t context,
                                               uint64_t size,
                                               M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_pinned_func_ptr_t)(
    M_driver_context_t context, uint64_t size, M_driver_memory_t *memory);

/**
 * Free pinned (page-locked) host memory for the given context.
 *
 * @param context (in) The context to free the memory from.
 * @param memory (in) The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_pinned(M_driver_context_t context,
                                              M_driver_memory_t memory);

typedef M_driver_status_t (*M_driver_memory_free_pinned_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_t memory);

/**
 * Allocate memory for the given context.
 *
 * @param context (in) The context to create the memory for.
 * @param size (in) Size in bytes of the allocation.
 * @param memory (out) Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_CONTEXT if the context is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_sync(M_driver_context_t context,
                                             uint64_t size,
                                             M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_sync_func_ptr_t)(
    M_driver_context_t context, uint64_t size, M_driver_memory_t *memory);

/**
 * Free memory for the given context.
 *
 * @param context (in) The context to free the memory from.
 * @param memory (in) The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_sync(M_driver_context_t context,
                                            M_driver_memory_t memory);

typedef M_driver_status_t (*M_driver_memory_free_sync_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_t memory);

/**
 * Asynchronously allocate memory on the given queue.
 *
 * @param queue (in) The queue to allocate memory on.
 * @param size (in) Size in bytes of the allocation.
 * @param memory (out) Pointer to the address of the newly allocated memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_QUEUE if the queue is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory);

typedef M_driver_status_t (*M_driver_memory_alloc_async_func_ptr_t)(
    M_driver_queue_t queue, uint64_t size, M_driver_memory_t *memory);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory);
/**
 * Asynchronously free memory on the given queue.
 *
 * @param queue (in) The queue to free the memory from.
 * @param memory (in) The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_QUEUE if the
 *         queue is not valid.
 */
typedef M_driver_status_t (*M_driver_memory_free_async_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_t memory);

/**
 * Common memory properties
 */
#define M_DRIVER_MEMORY_PROPERTY_ADDRESS "address"

/**
 * Retrieve a memory property.
 *
 * @param property_name (in) The name of the property as a null-terminated
 * string.
 * @param value (out) Type-erased pointer to the value of the property. Consult
 * the property table to instantiate the correct type at the call-site.
 * - e.g. "address" to retrieve the device memory address
 * @return Standard Status Protocol or
 * - M_DRIVER_STATUS_UNKNOWN_PROPERTY_NAME if the property name is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_property(M_driver_memory_t memory,
                                           const char *property_name,
                                           void *value);

typedef M_driver_status_t (*M_driver_memory_property_func_ptr_t)(
    M_driver_memory_t memory, const char *property_name, void *value);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_MEMORY_H
