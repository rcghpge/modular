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

typedef enum M_driver_alloc_mode_t {
  M_DRIVER_ALLOC_MODE_HOST = 1,
  M_DRIVER_ALLOC_MODE_SYNC = 2,
  M_DRIVER_ALLOC_MODE_ASYNC = 3,
} M_driver_alloc_mode_t;

/**
 * Allocate memory for the given context.
 *
 * @param context (in) The context to create the memory for.
 * @param mode (in) Memory allocation mode to use.
 * @param size (in) Size in bytes of the allocation.
 * @param memory (out) The memory.
 * @return Standard Status Protocol, M_DRIVER_STATUS_OUT_OF_MEMORY if the
 *         allocation fails, or M_DRIVER_STATUS_BAD_CONTEXT if the context is
 *         not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_alloc(M_driver_context_t context,
                                        M_driver_alloc_mode_t mode,
                                        uint64_t size, void **memory);

typedef M_driver_status_t (*M_driver_memory_alloc_func_ptr_t)(
    M_driver_context_t context, M_driver_alloc_mode_t mode, uint64_t size,
    void **memory);

/**
 * Free memory for the given context.
 *
 * @param context (in) The context to free the memory from.
 * @param memory (in) The memory to free.
 * @return Standard Status Protocol or M_DRIVER_STATUS_BAD_CONTEXT if the
 *         context is not valid.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_memory_free(M_driver_context_t context,
                                       void *memory);

typedef M_driver_status_t (*M_driver_memory_free_func_ptr_t)(
    M_driver_context_t context, void *memory);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_MEMORY_H
