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
 * @param[in] context The context to create the queue for.
 * @param[out] queue Pointer to store the newly created queue.
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
 * @param[in] context The context the queue was created for.
 * @param[in] queue The queue to destroy.
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
 * @param[in] queue The queue to use for the copy operation.
 * @param[in] dst Pointer to the destination device memory.
 * @param[in] src Pointer to the source host memory.
 * @param[in] size The number of bytes to copy.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_t dst,
                                                void *src, uint64_t size);

typedef M_driver_status_t (*M_driver_queue_copy_to_device_func_ptr_t)(
    M_driver_queue_t queue, M_driver_memory_t dst, void *src, uint64_t size);

/**
 * Copy data from device memory to host memory.
 *
 * @param[in] queue The queue to use for the copy operation.
 * @param[in] dst Pointer to the destination host memory.
 * @param[in] src Pointer to the source device memory.
 * @param[in] size The number of bytes to copy.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_t src,
                                                  uint64_t size);

typedef M_driver_status_t (*M_driver_queue_copy_from_device_func_ptr_t)(
    M_driver_queue_t queue, void *dst, M_driver_memory_t src, uint64_t size);

/**
 * Fill device memory with a repeated value.
 *
 * @param[in] queue The queue to use for the set operation.
 * @param[in] dst Pointer to the destination device memory.
 * @param[in] size The number of bytes to fill.
 * @param[in] value The value to fill with.
 * @param[in] value_size The size of the value in bytes.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t value_size);

typedef M_driver_status_t (*M_driver_queue_set_memory_func_ptr_t)(
    M_driver_queue_t queue, void *dst, uint64_t size, uint64_t value,
    uint64_t value_size);

/**
 * Struct representing a three dimensional launch configuration.
 */
typedef struct M_driver_dim_t {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} M_driver_dim_t;

/**
 * Struct representing a GPU launch configuration for executing an
 * M_driver_function_t. Used in M_driver_queue_execute() via
 * M_driver_queue_execute_config_t when targeting execution on a GPU device.
 */
typedef struct M_driver_queue_execute_config_gpu_t {
  M_driver_dim_t grid;       ///< The grid dimension, AKA ND-range
  M_driver_dim_t block;      ///< The block dimension, AKA work-group size.
  uint32_t shared_mem_bytes; ///< Shared memory size in bytes.
  void *attributes;          ///< Passthrough of CUlaunchAttributes
  uint32_t num_attributes;   ///< Number of elements pointed to by attributes.
} M_driver_queue_execute_config_gpu_t;

// Selects the launch config mode, currently only GPU is supported
typedef enum M_driver_queue_execute_mode_t {
  M_DRIVER_QUEUE_EXECUTE_MODE_GPU, ///< Specify execution on a GPU device.
} M_driver_queue_execute_mode_t;

/**
 * Struct representing a function execution launch configuration, contains
 * the config union which is discriminated by the mode enum.
 */
typedef struct M_driver_queue_execute_config_t {
  M_driver_queue_execute_mode_t mode; ///< Enum discriminates the config union.
  union M_driver_queue_execute_config_union_t {
    /// Execution launch configuration for GPU, used when mode is
    /// M_DRIVER_QUEUE_EXECUTE_MODE_GPU.
    M_driver_queue_execute_config_gpu_t gpu;
  } config; ///< The union discriminated by mode.
} M_driver_queue_execute_config_t;

/**
 * Execute a function on a device's queue.
 *
 * @param[in] queue The queue where the function will be executed.
 * @param[in] function The function which will be executed.
 * @param[in] config The function execution launch configuration.
 * @param[in] args Array of function arguments, packed in a target specific way.
 * @param[in] arg_sizes Array of function argument sizes.
 * @param[in] num_args Number of arguments in the args and arg_sizes arrays.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args);

typedef M_driver_status_t (*M_driver_queue_execute_func_ptr_t)(
    M_driver_queue_t queue, M_driver_function_t function,
    M_driver_queue_execute_config_t *config, void **args, uint64_t *arg_sizes,
    uint32_t num_args);

/**
 * Block the calling thread until all operations on the queue have completed.
 *
 * @param[in] queue The queue to synchronize on.
 * @return Standard Status Protocol.
 */
M_DRIVER_API_EXPORT
M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue);

typedef M_driver_status_t (*M_driver_queue_synchronize_func_ptr_t)(
    M_driver_queue_t queue);

#ifdef __cplusplus
}
#endif

#endif // M_DRIVER_QUEUE_H
