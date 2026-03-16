//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_context.h"

#include <cassert>

M_driver_context_t M_driver_context::create(M_driver_device_t device) {
  return new M_driver_context(device);
}

void M_driver_context::destroy(M_driver_context_t context) { delete context; }

DriverResult<void *> M_driver_context::alloc(uint64_t size) {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::ResourceOptions options =
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;
  MTL::Buffer *buffer = device->mDevice->newBuffer(size, options);
  if (!buffer)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  void *contents = buffer->contents();
  if (!contents) {
    buffer->release();
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  }

  trackBuffer(contents, buffer);
  return contents;
}

M_driver_status_t M_driver_context::free(void *memory) {
  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  MTL::Buffer *buffer = getTrackedBuffer(memory);
  if (!buffer)
    return M_DRIVER_STATUS_BAD_MEMORY;
  buffer->release();
  untrackBuffer(memory);
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
