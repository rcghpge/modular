//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "HAL/M_driver_context.h"
#include "Memory.h"

#include <cassert>
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
