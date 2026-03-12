//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_EVENT_H
#define M_DRIVER_EVENT_H

#include "M_driver_common.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event,
                                        bool enableTiming);

typedef M_driver_status_t (*M_driver_event_create_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t *event, bool enableTiming);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_synchronize_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_is_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready);

typedef M_driver_status_t (*M_driver_event_is_ready_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event, bool *is_ready);

M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_elapsed_time(M_driver_context_t context,
                                              M_driver_event_t start,
                                              M_driver_event_t end,
                                              float *elapsed_ms);

typedef M_driver_status_t (*M_driver_event_elapsed_time_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t start, M_driver_event_t end,
    float *elapsed_ms);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_EVENT_H
