//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_QUEUE_H
#define M_DRIVER_QUEUE_H

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a queue for the given context.
 *
 * @param context (in) The context to create the queue for.
 * @param queue (out) Pointer to store the newly created queue.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue);

typedef M_driver_status_t (*M_driver_queue_create_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t *queue);

/**
 * Destroy a previously created queue.
 *
 * @param context (in) The context the queue was created for.
 * @param queue (in) The queue to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue);

typedef M_driver_status_t (*M_driver_queue_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t queue);

/**
 * Copy data from host memory to device memory.
 *
 * @param queue (in) The queue to use for the copy operation.
 * @param dst (in) Pointer to the destination device memory.
 * @param src (in) Pointer to the source host memory.
 * @param size (in) The number of bytes to copy.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                void *dst, void *src,
                                                uint64_t size);

typedef M_driver_status_t (*M_driver_queue_copy_to_device_func_ptr_t)(
    M_driver_queue_t queue, void *dst, void *src, uint64_t size);

/**
 * Copy data from device memory to host memory.
 *
 * @param queue (in) The queue to use for the copy operation.
 * @param dst (in) Pointer to the destination host memory.
 * @param src (in) Pointer to the source device memory.
 * @param size (in) The number of bytes to copy.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst, void *src,
                                                  uint64_t size);

typedef M_driver_status_t (*M_driver_queue_copy_from_device_func_ptr_t)(
    M_driver_queue_t queue, void *dst, void *src, uint64_t size);

/**
 * Fill device memory with a repeated value.
 *
 * @param queue (in) The queue to use for the set operation.
 * @param dst (in) Pointer to the destination device memory.
 * @param size (in) The number of bytes to fill.
 * @param value (in) The value to fill with.
 * @param value_size (in) The size of the value in bytes.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue, void *dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t value_size);

typedef M_driver_status_t (*M_driver_queue_set_memory_func_ptr_t)(
    M_driver_queue_t queue, void *dst, uint64_t size, uint64_t value,
    uint64_t value_size);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_QUEUE_H
