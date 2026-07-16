//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef M_DRIVER_CONTEXT_H
#define M_DRIVER_CONTEXT_H

#include "M_driver_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set @p context as the calling thread's current context.
 *
 * Subsequent calls to vendor libraries that consult the current GPU context
 * will observe @p context. No prior context is saved;
 * callers that need to restore later should pair this with
 * M_driver_context_get_current_driver_context and
 * M_driver_context_set_current_driver_context.
 *
 * @param[in] context The context to make current.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_set_current(M_driver_context_t context);

typedef M_driver_status_t (*M_driver_context_set_current_func_ptr_t)(
    M_driver_context_t context);

/**
 * Get the calling thread's currently active driver-level context as an
 * opaque handle. @p context is used only to dispatch into the plugin; the
 * returned handle is not necessarily related to @p context.
 *
 * @param[in] context A context, used to identify the plugin to call into.
 * @param[out] driver_context The opaque driver-level handle of whatever is
 *             currently active.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context);

typedef M_driver_status_t (
    *M_driver_context_get_current_driver_context_func_ptr_t)(
    M_driver_context_t context, M_driver_native_handle_t *driver_context);

/**
 * Set @p driver_context (an opaque driver-level handle previously obtained
 * from M_driver_context_get_current_driver_context) as the calling thread's
 * current context. Used to restore a previously saved state.
 *
 * @param[in] context A context, used to identify the plugin to call into.
 * @param[in] driver_context The opaque driver-level handle to install.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t driver_context);

typedef M_driver_status_t (
    *M_driver_context_set_current_driver_context_func_ptr_t)(
    M_driver_context_t context, M_driver_native_handle_t driver_context);

/**
 * Query the device memory visible to @p context: the current free byte count
 * and the device's total byte capacity.
 *
 * @param[in] context The context whose device memory to query.
 * @param[out] free Current free device memory, in bytes.
 * @param[out] total Total device memory, in bytes.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         `context` is not a valid handle.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free, uint64_t *total);

typedef M_driver_status_t (*M_driver_context_memory_info_func_ptr_t)(
    M_driver_context_t context, uint64_t *free, uint64_t *total);

/**
 * Create a context for a device, for use with other functions in this API.
 *
 * @param[in] device The device to create the context for.
 * @param[out] context The context.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_DEVICE if the device
 *         is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_create(M_driver_device_t device,
                                          M_driver_context_t *context);

typedef M_driver_status_t (*M_driver_context_create_func_ptr_t)(
    M_driver_device_t device, M_driver_context_t *context);

/**
 * Destroy an active context for a device.
 *
 * @param[in] context The context to destroy.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         `context` is not a valid handle, or if the context was previously
 *         destroyed.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_context_destroy(M_driver_context_t context);

typedef M_driver_status_t (*M_driver_context_destroy_func_ptr_t)(
    M_driver_context_t context);

/**
 * Synchronous copies.
 *
 * Each runs directly on @p context with no queue/stream and blocks the calling
 * thread until the transfer completes.
 */

/**
 * Copy `dst->size` bytes from host memory @p src into device view @p dst.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_to_device_sync(M_driver_context_t context,
                                               M_driver_memory_view_t dst,
                                               const void *src);

typedef M_driver_status_t (*M_driver_copy_to_device_sync_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_view_t dst, const void *src);

/**
 * Copy `src->size` bytes from device view @p src into host memory @p dst.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_from_device_sync(M_driver_context_t context,
                                                 void *dst,
                                                 M_driver_memory_view_t src);

typedef M_driver_status_t (*M_driver_copy_from_device_sync_func_ptr_t)(
    M_driver_context_t context, void *dst, M_driver_memory_view_t src);

/**
 * Copy `dst->size` bytes between two views on @p context's device. Both views
 * must reside on @p context's device.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
                                                  M_driver_memory_view_t dst,
                                                  M_driver_memory_view_t src);

typedef M_driver_status_t (*M_driver_copy_intra_device_sync_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src);

/**
 * Copy `dst->size` bytes from a view on @p src_context's device into a view on
 * @p context's device (a peer copy). @p dst is on @p context's device.
 *
 * @return Standard Status Protocol, or M_DRIVER_STATUS_UNIMPLEMENTED on
 *         backends without peer-copy support.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_copy_inter_device_sync(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context);

typedef M_driver_status_t (*M_driver_copy_inter_device_sync_func_ptr_t)(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_CONTEXT_H
