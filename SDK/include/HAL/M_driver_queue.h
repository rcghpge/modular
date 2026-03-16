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

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_QUEUE_H
