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
  M_driver_device &dev = ctx.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();

  int constexpr kDefaultHipPriority = 0;

  hipStream_t stream;
  hipError_t result = hip.hipStreamCreateWithPriority(
      &stream, hipStreamNonBlocking, kDefaultHipPriority);
  if (result != hipSuccess)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  return new M_driver_queue(ctx, stream);
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;

  M_driver_device &dev = queue->context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipStreamDestroy(queue->stream);
  delete queue;
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst, void *src,
                                               uint64_t size) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result =
      hip.hipMemcpyAsync(dst->ptr, src, size, hipMemcpyHostToDevice, stream);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t size) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result =
      hip.hipMemcpyAsync(dst, src->ptr, size, hipMemcpyDeviceToHost, stream);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t valueSize) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = dst->ptr;

  switch (valueSize) {
  case 1: {
    hipError_t result = hip.hipMemsetAsync(
        dstPtr, static_cast<int>(value & 0xff), size, stream);
    if (result != hipSuccess)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    break;
  }
  case 2: {
    if ((size & 1) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    hipError_t result = hip.hipMemsetD16Async(
        dstPtr, static_cast<unsigned short>(value), size >> 1, stream);
    if (result != hipSuccess)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    break;
  }
  case 4: {
    if ((size & 3) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    hipError_t result = hip.hipMemsetD32Async(dstPtr, static_cast<int>(value),
                                              size >> 2, stream);
    if (result != hipSuccess)
      return M_DRIVER_STATUS_UNKNOWN_ERROR;
    break;
  }
  case 8: {
    if ((size & 7) != 0)
      return M_DRIVER_STATUS_INVALID_ARG;
    uint32_t hiBits = static_cast<uint32_t>(value >> 32);
    uint32_t loBits = static_cast<uint32_t>(value & 0xfffffffful);
    if (hiBits == loBits) {
      hipError_t result = hip.hipMemsetD32Async(
          dstPtr, static_cast<int>(loBits), size >> 2, stream);
      if (result != hipSuccess)
        return M_DRIVER_STATUS_UNKNOWN_ERROR;
    } else {
      size_t len = size >> 3;
      void *hiPtr = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(dstPtr) + sizeof(uint32_t));
      hipError_t result =
          hip.hipMemsetD2D32Async(dstPtr, 8, loBits, 1, len, stream);
      if (result != hipSuccess)
        return M_DRIVER_STATUS_UNKNOWN_ERROR;
      result = hip.hipMemsetD2D32Async(hiPtr, 8, hiBits, 1, len, stream);
      if (result != hipSuccess)
        return M_DRIVER_STATUS_UNKNOWN_ERROR;
    }
    break;
  }
  default:
    return M_DRIVER_STATUS_INVALID_ARG;
  }

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipEventRecord(event->event, stream);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvent(M_driver_event_t event) {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipStreamWaitEvent(stream, event->event, 0);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  M_driver_device &dev = context.getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result = hip.hipStreamSynchronize(stream);
  if (result != hipSuccess)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;

  return M_DRIVER_STATUS_SUCCESS;
}
