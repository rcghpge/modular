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

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue);

typedef M_driver_status_t (*M_driver_queue_create_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t *queue);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue);

typedef M_driver_status_t (*M_driver_queue_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_queue_t queue);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_QUEUE_H
