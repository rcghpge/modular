//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Common.h"
#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_context.h"
#include "Memory.h"

#include <cassert>
#include <cstring>
#include <string>

M_driver_context_t M_driver_context::create(M_driver_device_t device) {
  return new M_driver_context(device);
}

void M_driver_context::destroy(M_driver_context_t context) { delete context; }

DriverResult<M_driver_memory_t> M_driver_context::alloc(uint64_t size) {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  // Reject allocations at or above the device's per-buffer size limit with a
  // descriptive error. Otherwise newBuffer() returns null and the caller sees
  // only an opaque out-of-memory result with no hint that the request simply
  // exceeded maxBufferLength.
  size_t maxBufferLength = device->mDevice->maxBufferLength();
  if (size >= maxBufferLength)
    return tl::unexpected(device->driver->status.create(
        "Failed to allocate MTL Buffer of a size " + std::to_string(size) +
        " bytes which is larger than allowed " +
        std::to_string(maxBufferLength) + " bytes"));

  MTL::ResourceOptions options =
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;
  NS::SharedPtr<MTL::Buffer> buffer =
      NS::TransferPtr(device->mDevice->newBuffer(size, options));
  if (!buffer)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  void *contents = buffer->contents();
  if (!contents) {
    buffer->release();
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  }
  M_driver_memory_t memory = new (std::nothrow) M_driver_memory{buffer};
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  trackBuffer(contents, buffer.get());
  return memory;
}

M_driver_status_t M_driver_context::free(M_driver_memory_t memory) {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  untrackBuffer(memory->mBuffer->contents());
  delete memory;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  if (size > 0)
    memcpy(static_cast<char *>(dst->mBuffer->contents()) + dstOffset, src,
           size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  if (size > 0)
    memcpy(dst, static_cast<char *>(src->mBuffer->contents()) + srcOffset,
           size);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  if (size > 0)
    memcpy(static_cast<char *>(dst->mBuffer->contents()) + dstOffset,
           static_cast<char *>(src->mBuffer->contents()) + srcOffset, size);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_context_create(M_driver_device_t device, M_driver_context_t *context) {
  assert(context && "context must not be null");
  *context = M_driver_context::create(device);
  return M_DRIVER_STATUS_SUCCESS;
}

extern "C" M_driver_status_t
M_driver_context_destroy(M_driver_context_t context) {
  M_driver_context::destroy(context);
  return M_DRIVER_STATUS_SUCCESS;
}

// Synchronous copy C ABI: thin wrappers over the M_driver_context methods
// above. Metal targets a single GPU, so a peer (inter-device) copy is
// unimplemented.
extern "C" M_driver_status_t
M_driver_copy_to_device_sync(M_driver_context_t context,
                             M_driver_memory_view_t dst, const void *src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || dst->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (src == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return context->copyToDeviceSync(dst->memory, dst->offset, src, dst->size);
}

extern "C" M_driver_status_t
M_driver_copy_from_device_sync(M_driver_context_t context, void *dst,
                               M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (src == nullptr || src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (dst == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  return context->copyFromDeviceSync(dst, src->memory, src->offset, src->size);
}

extern "C" M_driver_status_t
M_driver_copy_intra_device_sync(M_driver_context_t context,
                                M_driver_memory_view_t dst,
                                M_driver_memory_view_t src) {
  if (context == nullptr)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (dst == nullptr || src == nullptr || dst->memory == nullptr ||
      src->memory == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;
  return context->copyIntraDeviceSync(dst->memory, dst->offset, src->memory,
                                      src->offset, dst->size);
}

extern "C" M_driver_status_t M_driver_copy_inter_device_sync(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_memory_view_t dst,
    [[maybe_unused]] M_driver_memory_view_t src,
    [[maybe_unused]] M_driver_context_t src_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

// Metal has no driver-level current-context concept (command encoding is
// bound to explicit queue/context objects, never to the calling thread), and
// no direct free/total memory query.

extern "C" M_driver_status_t
M_driver_context_set_current([[maybe_unused]] M_driver_context_t context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

extern "C" M_driver_status_t M_driver_context_get_current_driver_context(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_native_handle_t *driver_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

extern "C" M_driver_status_t M_driver_context_set_current_driver_context(
    [[maybe_unused]] M_driver_context_t context,
    [[maybe_unused]] M_driver_native_handle_t driver_context) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}

extern "C" M_driver_status_t
M_driver_context_memory_info([[maybe_unused]] M_driver_context_t context,
                             [[maybe_unused]] uint64_t *free,
                             [[maybe_unused]] uint64_t *total) {
  return M_DRIVER_STATUS_UNIMPLEMENTED;
}
