//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Event.h"

DriverResult<M_driver_queue_t> M_driver_queue::create(M_driver_context &ctx) {
  ScopedContext scope(ctx.primaryCtx, ctx.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  CUDALoader &cuda = ctx.device.driver.rawCuda;

  // Default priority is 0 (lowest); compatible with CUDA specs.
  int constexpr priority = 0;

  // CUDA blocking streams are a legacy feature; interpretation of "blocking"
  // is w.r.t. the default stream and does not mean "synchronous".
  // We want our streams to be independent of any other stream.
  unsigned int constexpr kNonBlocking = 0x1;

  CUstream stream;
  CUDA_CHECK_DR(
      cuda.cuStreamCreateWithPriority(&stream, kNonBlocking, priority),
      M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_queue(ctx, stream);
}

DriverResult<M_driver_memory_t> M_driver_queue::allocAsync(uint64_t size) {
  if (size == 0)
    return nullptr;

  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &cuda = context.device.driver.rawCuda;

  CUdeviceptr devicePtr;
  CUresult result = cuda.cuMemAllocAsync(&devicePtr, size, stream);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  M_driver_memory_t memory =
      new (std::nothrow) M_driver_memory(reinterpret_cast<void *>(devicePtr));
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_queue::freeAsync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;

  CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(memory->ptr);
  delete memory;
  CUresult result = cuda.cuMemFreeAsync(devicePtr, stream);
  if (result != CUDA_SUCCESS)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst, void *src,
                                               uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dst->ptr),
                                    src, size, stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuMemcpyDtoHAsync(
                 dst, reinterpret_cast<CUdeviceptr>(src->ptr), size, stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyDeviceToDevice(M_driver_memory_t dst,
                                                     M_driver_memory_t src,
                                                     uint64_t size) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(dst->ptr),
                                    reinterpret_cast<CUdeviceptr>(src->ptr),
                                    size, stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t valueSize) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = context.device.driver.rawCuda;
  CUdeviceptr dstPtr = reinterpret_cast<CUdeviceptr>(dst->ptr);
  switch (valueSize) {
  case 1:
    CUDA_CHECK(
        cuda.cuMemsetD8Async(dstPtr, static_cast<uint8_t>(value), size, stream),
        M_DRIVER_STATUS_UNKNOWN_ERROR);
    break;
  case 2:
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    CUDA_CHECK(cuda.cuMemsetD16Async(dstPtr, static_cast<uint16_t>(value),
                                     size >> 1, stream),
               M_DRIVER_STATUS_UNKNOWN_ERROR);
    break;
  case 4:
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    CUDA_CHECK(cuda.cuMemsetD32Async(dstPtr, static_cast<uint32_t>(value),
                                     size >> 2, stream),
               M_DRIVER_STATUS_UNKNOWN_ERROR);
    break;
  case 8: {
    if ((size & 7) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    uint32_t hiBits = static_cast<uint32_t>(value >> 32);
    uint32_t loBits = static_cast<uint32_t>(value & 0xfffffffful);
    if (hiBits == loBits) {
      CUDA_CHECK(cuda.cuMemsetD32Async(dstPtr, loBits, size >> 2, stream),
                 M_DRIVER_STATUS_UNKNOWN_ERROR);
    } else {
      size_t len = size >> 3;
      CUdeviceptr hiPtr = dstPtr + sizeof(uint32_t);
      CUDA_CHECK(cuda.cuMemsetD2D32Async(dstPtr, 8, loBits, 1, len, stream),
                 M_DRIVER_STATUS_UNKNOWN_ERROR);
      CUDA_CHECK(cuda.cuMemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream),
                 M_DRIVER_STATUS_UNKNOWN_ERROR);
    }
    break;
  }
  default:
    return M_DRIVER_STATUS_INVALID_ARG;
  }
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuEventRecord(event->event, stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t num_events) {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  for (uint32_t i = 0; i < num_events; ++i)
    CUDA_CHECK(cuda.cuStreamWaitEvent(stream, events[i]->event, 0),
               M_DRIVER_STATUS_UNKNOWN_ERROR);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;

  ScopedContext scope(queue->context.primaryCtx,
                      queue->context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = queue->context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuStreamDestroy(queue->stream),
             M_DRIVER_STATUS_UNKNOWN_ERROR);

  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  ScopedContext scope(context.primaryCtx, context.device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  CUDALoader &cuda = context.device.driver.rawCuda;
  CUDA_CHECK(cuda.cuStreamSynchronize(stream), M_DRIVER_STATUS_UNKNOWN_ERROR);

  return M_DRIVER_STATUS_SUCCESS;
}
