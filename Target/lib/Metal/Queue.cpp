//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Memory.h"

#include "HAL/M_driver_queue.h"

DriverResult<M_driver_queue_t>
M_driver_queue::create(M_driver_context_t context) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(context->device->mDevice->newCommandQueue());
  if (!commandQueue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  M_driver_queue_t queue =
      new (std::nothrow) M_driver_queue(context, commandQueue);
  if (!queue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return queue;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

#define RETURN_ERROR(MSG) return M_DRIVER_STATUS_UNKNOWN_ERROR

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst, void *src,
                                               uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
  //                      TRACY_COLOR_PURPLE);

  if (src == nullptr)
    RETURN_ERROR("Source pointer is null");
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  NS::UInteger bufferLength = dst->mBuffer->length();
  if (bufferLength == 0)
    RETURN_ERROR("Metal buffer length is 0 - buffer may be invalid");

  // Check that copy fits within the buffer from the offset
  if (dst->offset + size > bufferLength)
    RETURN_ERROR("Copy size exceeds Metal buffer length");

  // Source is regular host memory - use memcpy approach

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue

  __block size_t capturedByteSize = size;
  __block size_t capturedOffset = dst->offset;
  MTL::Buffer *dstBuf = dst->mBuffer.get();

  // Add completion handler to perform the memcpy when GPU is done
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *dstContents = dstBuf->contents();
    if (dstContents) {
      // Apply offset to destination pointer
      void *dstPtr = static_cast<char *>(dstContents) + capturedOffset;
      std::memcpy(dstPtr, src, capturedByteSize);
    }
  });

  // Commit to queue but wait for memcpy completion
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
  //                      TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (dst == nullptr)
    RETURN_ERROR("Destination pointer cannot be null");

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // For memcpy operations, we need to wait for completion even though we commit
  // to queue Add completion handler to perform the memcpy when GPU is done
  __block size_t capturedOffset = src->offset;
  MTL::Buffer *srcBuf = src->mBuffer.get();

  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    void *srcContents = srcBuf->contents();
    if (srcContents) {
      // Apply offset to source pointer
      void *srcPtr = static_cast<char *>(srcContents) + capturedOffset;
      std::memcpy(dst, srcPtr, size);
    }
  });

  // Commit to queue but wait for memcpy completion
  // When doing cpu memcopy, we need to wait for the command buffer to complete
  // before committing the next command buffer otherwise, the next command
  // buffer will be executed before the current one is completed
  cmdBuffer->commit();
  cmdBuffer->waitUntilCompleted();

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t valueSize) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
  //                      TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    RETURN_ERROR("Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dst->offset;

  size_t count = size / valueSize;
  if (reinterpret_cast<uintptr_t>(dstPtr) % valueSize != 0)
    RETURN_ERROR("destination buffer not properly aligned");

  // Push  an empty command buffer to the queue
  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    RETURN_ERROR("Failed to create command buffer");

  // add completion handler to the command buffer
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *buffer) {
    switch (valueSize) {
    case 1:
      std::memset(dstPtr, static_cast<uint8_t>(value), size);
      break;
    case 2: {
      uint16_t v16 = static_cast<uint16_t>(value);
      uint16_t *p16 = static_cast<uint16_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p16[i] = v16;
      break;
    }
    case 4: {
      uint32_t v32 = static_cast<uint32_t>(value);
      uint32_t *p32 = static_cast<uint32_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p32[i] = v32;
      break;
    }
    case 8: {
      uint64_t *p64 = static_cast<uint64_t *>(dstPtr);
      for (size_t i = 0; i < count; ++i)
        p64[i] = value;
      break;
    }
    }
  });

  // commit the command buffer
  cmdBuffer->commit();
  // When doing cpu memcopy, we need to wait for the command buffer to complete
  // before committing the next command buffer otherwise, the next command
  // buffer will be executed before the current one is completed
  cmdBuffer->waitUntilCompleted();

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue_create(M_driver_context_t context,
                                        M_driver_queue_t *queue) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (queue == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  auto queueOr = M_driver_queue::create(context);
  if (!queueOr)
    return queueOr.error();
  *queue = *queueOr;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_queue_destroy([[maybe_unused]] M_driver_context_t context,
                       M_driver_queue_t queue) {
  return M_driver_queue::destroy(queue);
}

M_driver_status_t M_driver_queue_copy_to_device(M_driver_queue_t queue,
                                                M_driver_memory_t dst,
                                                void *src, uint64_t size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->copyToDevice(dst, src, size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_t src,
                                                  uint64_t size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyFromDevice(dst, src, size);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_t dst,
                                            uint64_t size, uint64_t value,
                                            uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->setMemory(dst, size, value, value_size);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
