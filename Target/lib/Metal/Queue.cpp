//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Queue.h"
#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Event.h"
#include "Function.h"
#include "Memory.h"

#include <algorithm>
#include <dispatch/dispatch.h>

DriverResult<M_driver_queue_t>
M_driver_queue::create(M_driver_context_t context) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  NS::SharedPtr<MTL::CommandQueue> commandQueue =
      NS::TransferPtr(context->device->mDevice->newCommandQueue());
  if (!commandQueue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  // Seed a thread-local autorelease pool on Metal's callback thread.
  // Metal autoreleases AGXG16SDevice on its internal callback thread during
  // command buffer processing; that thread has no pool by default. Submitting
  // a no-op command buffer with a completion handler that creates a
  // thread_local pool ensures all subsequent autoreleases on that thread are
  // caught. The pool persists for the thread's lifetime so this is
  // forward-compatible with async (non-blocking) queue usage.
  // NOTE: dispatch_semaphore is required here (not std::binary_semaphore)
  // because Metal dispatches completion handlers via GCD. Using a GCD-aware
  // wait ensures the handler runs on the correct callback thread.
  dispatch_semaphore_t sem = dispatch_semaphore_create(0);
  MTL::CommandBuffer *warmup = commandQueue->commandBuffer();
  warmup->addCompletedHandler(^(MTL::CommandBuffer *) {
    M_driver_driver::ensureThreadlocalAutoreleasePoolExists();
    dispatch_semaphore_signal(sem);
  });
  warmup->commit();
  dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
  dispatch_release(sem);

  M_driver_queue_t queue = new (std::nothrow) M_driver_queue{
      context, commandQueue, /*mPending=*/{}, /*mPendingWaits=*/{},
      /*mMutex=*/{}};
  if (!queue)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return queue;
}

M_driver_status_t M_driver_queue::destroy(M_driver_queue_t queue) {
  delete queue;
  return M_DRIVER_STATUS_SUCCESS;
}

void M_driver_queue::encodePendingWaits(MTL::CommandBuffer *cb) {
  for (const auto &e : mPendingWaits)
    cb->encodeWait(e.get(), 1);
}

M_driver_status_t M_driver_queue::copyToDevice(M_driver_memory_t dst,
                                               uint64_t dstOffset,
                                               const void *src, uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyToDevice",
  //                      TRACY_COLOR_PURPLE);

  if (src == nullptr)
    return context->device->driver->status.create("Source pointer is null");
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  NS::UInteger bufferLength = dst->mBuffer->length();
  if (bufferLength == 0)
    return context->device->driver->status.create(
        "Metal buffer length is 0 - buffer may be invalid");

  // Check that copy fits within the buffer from the offset
  if (dstOffset + size > bufferLength)
    return context->device->driver->status.create(
        "Copy size exceeds Metal buffer length");

  NS::SharedPtr<MTL::SharedEvent> doneEvent =
      NS::TransferPtr(context->device->mDevice->newSharedEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::SharedEvent for copyToDevice");

  __block size_t capturedByteSize = size;
  __block size_t capturedOffset = dstOffset;
  MTL::Buffer *dstBuf = dst->mBuffer.get();

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  // The CB has no GPU work — the "copy" is a std::memcpy executed CPU-side in
  // the completion handler. Metal's UMA model means there's no dedicated HtoD
  // copy API; host and device share memory, so a plain memcpy is the transfer.
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *) {
    void *dstContents = dstBuf->contents();
    if (dstContents) {
      void *dstPtr = static_cast<char *>(dstContents) + capturedOffset;
      std::memcpy(dstPtr, src, capturedByteSize);
    }
    doneEvent->setSignaledValue(1);
  });
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyFromDevice(void *dst,
                                                 M_driver_memory_t src,
                                                 uint64_t srcOffset,
                                                 uint64_t size) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueCopyFromDevice",
  //                      TRACY_COLOR_PURPLE);
  if (dst == nullptr)
    return context->device->driver->status.create(
        "Destination pointer cannot be null");
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  NS::SharedPtr<MTL::SharedEvent> doneEvent =
      NS::TransferPtr(context->device->mDevice->newSharedEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::SharedEvent for copyFromDevice");

  if (srcOffset + size > src->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal source buffer length");

  __block size_t capturedOffset = srcOffset;
  MTL::Buffer *srcBuf = src->mBuffer.get();

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  // The CB has no GPU work — the "copy" is a std::memcpy executed CPU-side in
  // the completion handler. Metal's UMA model means there's no dedicated HtoD
  // copy API; host and device share memory, so a plain memcpy is the transfer.
  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *) {
    void *srcContents = srcBuf->contents();
    if (srcContents) {
      void *srcPtr = static_cast<char *>(srcContents) + capturedOffset;
      std::memcpy(dst, srcPtr, size);
    }
    doneEvent->setSignaledValue(1);
  });
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::copyIntraDevice(M_driver_memory_t dst,
                                                  uint64_t dstOffset,
                                                  M_driver_memory_t src,
                                                  uint64_t srcOffset,
                                                  uint64_t size) {
  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal destination buffer length");
  if (srcOffset + size > src->mBuffer->length())
    return context->device->driver->status.create(
        "Copy size exceeds Metal source buffer length");

  NS::SharedPtr<MTL::Event> doneEvent =
      NS::TransferPtr(context->device->mDevice->newEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::Event for copyIntraDevice");

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  MTL::BlitCommandEncoder *encoder = cmdBuffer->blitCommandEncoder();
  if (encoder == nullptr)
    return context->device->driver->status.create(
        "Failed to create blit command encoder");
  encoder->copyFromBuffer(src->mBuffer.get(), srcOffset, dst->mBuffer.get(),
                          dstOffset, size);
  encoder->endEncoding();

  cmdBuffer->encodeSignalEvent(doneEvent.get(), 1);
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::setMemory(M_driver_memory_t dst,
                                            uint64_t dstOffset, uint64_t size,
                                            uint8_t value) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueSetMemory",
  //                      TRACY_COLOR_PURPLE);

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Set size exceeds Metal buffer length");

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dstOffset;

  NS::SharedPtr<MTL::SharedEvent> doneEvent =
      NS::TransferPtr(context->device->mDevice->newSharedEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::SharedEvent for setMemory");

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *) {
    std::memset(dstPtr, value, size);
    doneEvent->setSignaledValue(1);
  });
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::fillMemory(M_driver_memory_t dst,
                                             uint64_t dstOffset, uint64_t size,
                                             uint64_t value,
                                             uint64_t valueSize) {
  if (valueSize == 1)
    return setMemory(dst, dstOffset, size, static_cast<uint8_t>(value));

  if (valueSize != 2 && valueSize != 4 && valueSize != 8)
    return M_DRIVER_STATUS_INVALID_ARG;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  if (size == 0)
    return M_DRIVER_STATUS_SUCCESS; // no-op

  if ((size % valueSize) != 0)
    return M_DRIVER_STATUS_INVALID_ARG;

  if (dstOffset + size > dst->mBuffer->length())
    return context->device->driver->status.create(
        "Fill size exceeds Metal buffer length");

  void *dstContents = dst->mBuffer->contents();
  if (dstContents == nullptr)
    return context->device->driver->status.create(
        "Shared buffer contents() returned null");

  // Apply offset to get actual destination pointer
  void *dstPtr = static_cast<char *>(dstContents) + dstOffset;

  size_t count = size / valueSize;
  if (reinterpret_cast<uintptr_t>(dstPtr) % valueSize != 0)
    return context->device->driver->status.create(
        "destination buffer not properly aligned");

  NS::SharedPtr<MTL::SharedEvent> doneEvent =
      NS::TransferPtr(context->device->mDevice->newSharedEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::SharedEvent for fillMemory");

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  cmdBuffer->addCompletedHandler(^(MTL::CommandBuffer *) {
    switch (valueSize) {
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
    doneEvent->setSignaledValue(1);
  });
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

namespace {
MTL::Size toSize(M_driver_dim_t dim) { return {dim.x, dim.y, dim.z}; }
} // namespace

M_driver_status_t
M_driver_queue::execute(M_driver_function_t function,
                        M_driver_queue_execute_config_t *config, void **args,
                        uint64_t *argSizes, uint32_t numArgs) {
  // TRACY_ZONE_SCOPED_NC("MetalDeviceContext::enqueueFunctionExecDirect",
  //                      TRACY_COLOR_PURPLE);

  if (config->mode != M_DRIVER_QUEUE_EXECUTE_MODE_GPU)
    return M_DRIVER_STATUS_EXECUTE_MODE_NOT_SUPPORTED;

  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Validate every argument before opening an encoder. Metal asserts when a
  // command buffer is dropped with an open encoder, so it is not safe to bail
  // out partway through encoding — check all arguments up front.
  const NS::UInteger maxBufferLength =
      context->device->mDevice->maxBufferLength();
  for (uint32_t i = 0; i < function->argumentCount; ++i) {
    if (args[i] == nullptr)
      return context->device->driver->status.create("Null argument at index " +
                                                    std::to_string(i));
    if (argSizes[i] == 0)
      return context->device->driver->status.create(
          "Unknown argument size at index " + std::to_string(i));
    // Reject per-argument buffers at or above the device's per-buffer size
    // limit, mirroring the guard on context allocations. Checked here, before
    // the encoder is opened, so the per-arg newBuffer() below cannot return
    // null mid-encode.
    if (argSizes[i] >= maxBufferLength)
      return context->device->driver->status.create(
          "Failed to allocate MTL Buffer of a size " +
          std::to_string(argSizes[i]) + " bytes which is larger than allowed " +
          std::to_string(maxBufferLength) + " bytes");
  }

  // Validate the dynamic threadgroup (shared) memory request against the device
  // limit before opening an encoder — Metal asserts if a command buffer is
  // dropped with an open encoder, so reject early.
  const uint32_t sharedMemBytes = config->config.gpu.shared_mem_bytes;
  if (sharedMemBytes > 0) {
    NS::UInteger maxSharedMemBytes =
        context->device->mDevice->maxThreadgroupMemoryLength();
    if (sharedMemBytes > maxSharedMemBytes)
      return context->device->driver->status.create(
          "Shared memory byte count " + std::to_string(sharedMemBytes) +
          " exceeds device limit of " + std::to_string(maxSharedMemBytes));
  }

  NS::SharedPtr<MTL::Event> doneEvent =
      NS::TransferPtr(context->device->mDevice->newEvent());
  if (!doneEvent)
    return context->device->driver->status.create(
        "Failed to create MTL::Event for execute");

  std::lock_guard<std::mutex> lock(mMutex);

  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer");

  encodePendingWaits(cmdBuffer);

  MTL::ComputeCommandEncoder *encoder = cmdBuffer->computeCommandEncoder();
  if (encoder == nullptr)
    return context->device->driver->status.create(
        "Failed to create compute command encoder");
  encoder->setComputePipelineState(function->pipelineState.get());

  // Bind all buffer arguments. Arguments were validated above, before the
  // encoder was opened.
  for (uint32_t i = 0; i < function->argumentCount; ++i) {
    // We have more complex argument, such as pointer to a struct. Allocate
    // new MTLBuffer and set it.
    NS::SharedPtr<MTL::Buffer> buf =
        NS::TransferPtr(context->device->mDevice->newBuffer(
            args[i], argSizes[i],
            MTL::ResourceOptions(MTL::StorageModeShared) |
                MTL::ResourceOptions(MTL::CPUCacheModeWriteCombined)));
    encoder->setBuffer(buf.get(), 0, i);
  }

  // Set resources that might be used by the kernel
  // FIXME: only set resources that are used by the kernel
  for (const auto &it : context->allocations)
    encoder->useResource(it.second, MTL::ResourceUsageWrite);

  // Bind dynamic threadgroup (shared) memory if the kernel requested it.
  if (sharedMemBytes > 0)
    encoder->setThreadgroupMemoryLength(sharedMemBytes, /*index=*/0);

  // Dispatch
  MTL::Size gridSize = toSize(config->config.gpu.grid);
  MTL::Size threadgroupSize = toSize(config->config.gpu.block);
  encoder->dispatchThreadgroups(gridSize, threadgroupSize);

  // End encoding
  encoder->endEncoding();

  cmdBuffer->encodeSignalEvent(doneEvent.get(), 1);
  cmdBuffer->commit();
  mPending.push_back(doneEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::synchronize() {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::CommandBuffer *cmdBuffer = nullptr;
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mPending.empty() && mPendingWaits.empty())
      return M_DRIVER_STATUS_SUCCESS;

    // Build a CB that waits on every pending event and any externals from
    // wait_for_events, then host-block on it.
    cmdBuffer = mCommandQueue->commandBuffer();
    if (cmdBuffer == nullptr)
      return context->device->driver->status.create(
          "Failed to create command buffer for synchronize");

    encodePendingWaits(cmdBuffer);

    for (const auto &e : mPending)
      cmdBuffer->encodeWait(e.get(), 1);
    cmdBuffer->commit();

    mPending.clear();
    mPendingWaits.clear();
  }

  cmdBuffer->waitUntilCompleted();
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::recordEvent(M_driver_event_t event) {
  NS::SharedPtr<NS::AutoreleasePool> autoreleasePool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  std::lock_guard<std::mutex> lock(mMutex);

  // Empty-queue host-signal optimization: only available when the user
  // event is CPU-visible (a plain device MTL::Event has no host-side
  // signal API).
  if (mPending.empty() && mPendingWaits.empty() && event->sharedEvent) {
    event->sharedEvent->setSignaledValue(1);
    return M_DRIVER_STATUS_SUCCESS;
  }

  // Build CB that waits on (a) any externals stashed by waitForEvents and
  // (b) every pending in-flight op on this queue, then signals the user
  // event.
  MTL::CommandBuffer *cmdBuffer = mCommandQueue->commandBuffer();
  if (cmdBuffer == nullptr)
    return context->device->driver->status.create(
        "Failed to create command buffer for record_event");

  encodePendingWaits(cmdBuffer);

  for (const NS::SharedPtr<MTL::Event> &e : mPending)
    cmdBuffer->encodeWait(e.get(), 1);
  cmdBuffer->encodeSignalEvent(event->getEvent(), 1);
  cmdBuffer->commit();

  // Anyone waiting on user_event transitively waits on everything we just
  // encoded.
  mPending.clear();
  if (event->sharedEvent)
    mPending.push_back(event->sharedEvent);
  else
    mPending.push_back(event->deviceEvent);

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_queue::waitForEvents(M_driver_event_t *events,
                                                uint32_t numEvents) {
  if (numEvents == 0)
    return M_DRIVER_STATUS_SUCCESS;

  // Stash wait events; the next op submitted on this queue will encode-wait on
  // them before its own work.
  std::lock_guard<std::mutex> lock(mMutex);
  mPendingWaits.reserve(mPendingWaits.size() + numEvents);
  for (uint32_t i = 0; i < numEvents; ++i) {
    if (events[i]->sharedEvent)
      mPendingWaits.push_back(events[i]->sharedEvent);
    else
      mPendingWaits.push_back(events[i]->deviceEvent);
  }
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
                                                M_driver_memory_view_t dst,
                                                const void *src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return queue->copyToDevice(dst->memory, dst->offset, src, dst->size);
}

M_driver_status_t M_driver_queue_copy_from_device(M_driver_queue_t queue,
                                                  void *dst,
                                                  M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyFromDevice(dst, src->memory, src->offset, src->size);
}

M_driver_status_t M_driver_queue_copy_intra_device(M_driver_queue_t queue,
                                                   M_driver_memory_view_t dst,
                                                   M_driver_memory_view_t src) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr || src == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->copyIntraDevice(dst->memory, dst->offset, src->memory,
                                src->offset, dst->size);
}

M_driver_status_t M_driver_queue_set_memory(M_driver_queue_t queue,
                                            M_driver_memory_view_t dst,
                                            uint8_t value) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->setMemory(dst->memory, dst->offset, dst->size, value);
}

M_driver_status_t M_driver_queue_fill(M_driver_queue_t queue,
                                      M_driver_memory_view_t dst,
                                      uint64_t value, uint64_t value_size) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (dst == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return queue->fillMemory(dst->memory, dst->offset, dst->size, value,
                           value_size);
}

M_driver_status_t
M_driver_queue_execute(M_driver_queue_t queue, M_driver_function_t function,
                       M_driver_queue_execute_config_t *config, void **args,
                       uint64_t *arg_sizes, uint32_t num_args) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (function == nullptr)
    return M_DRIVER_STATUS_BAD_FUNCTION;
  return queue->execute(function, config, args, arg_sizes, num_args);
}

M_driver_status_t M_driver_queue_record_event(M_driver_queue_t queue,
                                              M_driver_event_t event) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (event == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->recordEvent(event);
}

M_driver_status_t M_driver_queue_wait_for_events(M_driver_queue_t queue,
                                                 M_driver_event_t *events,
                                                 uint32_t num_events) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (events == nullptr)
    return M_DRIVER_STATUS_BAD_EVENT;
  return queue->waitForEvents(events, num_events);
}

M_driver_status_t M_driver_queue_synchronize(M_driver_queue_t queue) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  return queue->synchronize();
}

M_driver_status_t M_driver_queue_is_stream(M_driver_queue_t queue,
                                           bool *is_stream) {
  if (queue == nullptr)
    return M_DRIVER_STATUS_BAD_QUEUE;
  if (is_stream == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  *is_stream = false;
  return M_DRIVER_STATUS_SUCCESS;
}
