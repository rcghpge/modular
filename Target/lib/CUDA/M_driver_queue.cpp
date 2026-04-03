//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Error.h"
#include "Function.h"
#include "HAL/M_driver_api.h"
#include "Queue.h"

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;

  DriverResult<M_driver_queue_t> result = M_driver_queue::create(*context);
  if (!result)
    return result.error();
  *queue = *result;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_destroy(M_driver_context_t context,
                                         M_driver_queue_t queue) {
  return M_driver_queue::destroy(queue);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_t dst,
                                                void *src, uint64_t size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->copyToDevice(dst, src, size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_t src,
                                                  uint64_t size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->copyFromDevice(dst, src, size);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->setMemory(dst, size, value, value_size);
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (queue == nullptr || event == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return queue->recordEvent(event);
}

M_driver_status_t M_driver_queue_wait_for_event(M_driver_queue_t queue,
                                                M_driver_event_t event) {
  if (queue == nullptr || event == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  return queue->waitForEvent(event);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->synchronize();
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_HANDLE;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  if (config == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;

  ScopedContext scope(queue->context.primaryCtx,
                      queue->context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = queue->context.device.driver.rawCuda;

  auto &gpu = config->config.gpu;
  CUlaunchConfig launchConfig{
      .gridDimX = gpu.grid.x,
      .gridDimY = gpu.grid.y,
      .gridDimZ = gpu.grid.z,
      .blockDimX = gpu.block.x,
      .blockDimY = gpu.block.y,
      .blockDimZ = gpu.block.z,
      .sharedMemBytes = gpu.shared_mem_bytes,
      .hStream = queue->stream,
      .attrs = static_cast<CUlaunchAttribute *>(gpu.attributes),
      .numAttrs = gpu.num_attributes,
  };

  CUDA_CHECK(cuda.cuLaunchKernelEx(&launchConfig, function->function, args,
                                   /*extra=*/nullptr),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  return M_DRIVER_STATUS_SUCCESS;
}
