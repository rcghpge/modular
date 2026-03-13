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

/**
 * Create an event for the given context.
 *
 * @param context (in) The context to create the event for.
 * @param event (out) The event.
 * @param enableTiming (in) If true, the event records timing information,
 *        enabling use with M_driver_event_elapsed_time(). If false, timing
 *        is disabled and elapsed time queries on this event are invalid.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_create(M_driver_context_t context,
                                        M_driver_event_t *event,
                                        bool enableTiming);

typedef M_driver_status_t (*M_driver_event_create_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t *event, bool enableTiming);

/**
 * Destroy an event.
 *
 * @param context (in) The context the event belongs to.
 * @param event (in) The event to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_destroy(M_driver_context_t context,
                                         M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_destroy_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

/**
 * Block the calling thread until the event has completed.
 *
 * @param context (in) The context the event belongs to.
 * @param event (in) The event to synchronize on.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_synchronize(M_driver_context_t context,
                                             M_driver_event_t event);

typedef M_driver_status_t (*M_driver_event_synchronize_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event);

/**
 * Query whether an event has completed without blocking.
 *
 * @param context (in) The context the event belongs to.
 * @param event (in) The event to query.
 * @param is_ready (out) Set to true if the event has completed, false
 *        otherwise.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_event_is_ready(M_driver_context_t context,
                                          M_driver_event_t event,
                                          bool *is_ready);

typedef M_driver_status_t (*M_driver_event_is_ready_func_ptr_t)(
    M_driver_context_t context, M_driver_event_t event, bool *is_ready);

/**
 * Compute the elapsed time in milliseconds between two events.
 *
 * Both events must have been created with enableTiming set to true.
 *
 * @param context (in) The context both events belong to.
 * @param start (in) The start event.
 * @param end (in) The end event.
 * @param elapsed_ms (out) The elapsed time in milliseconds between start and
 *        end.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
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
