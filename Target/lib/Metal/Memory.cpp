//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "Context.h"
#include "Device.h"
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_memory_alloc(M_driver_context_t context,
                                        M_driver_alloc_mode_t mode,
                                        uint64_t size, void **memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_BAD_MEMORY;
  if (size == 0) {
    *memory = nullptr;
    return M_DRIVER_STATUS_SUCCESS;
  }

  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::Device *metalDev = context->device->mDevice;
  if (!metalDev)
    return M_DRIVER_STATUS_BAD_CONTEXT;

  MTL::ResourceOptions options =
      MTL::ResourceStorageModeShared | MTL::ResourceCPUCacheModeWriteCombined;
  MTL::Buffer *buffer = metalDev->newBuffer(size, options);
  if (!buffer)
    return M_DRIVER_STATUS_OUT_OF_MEMORY;

  switch (mode) {
  case M_DRIVER_ALLOC_MODE_HOST: {
    void *contents = buffer->contents();
    if (!contents) {
      buffer->release();
      return M_DRIVER_STATUS_OUT_OF_MEMORY;
    }
    *memory = contents;
    context->trackBuffer(contents, buffer);
    return M_DRIVER_STATUS_SUCCESS;
  }
  case M_DRIVER_ALLOC_MODE_SYNC:
  case M_DRIVER_ALLOC_MODE_ASYNC: {
    uint64_t gpuAddress = buffer->gpuAddress();
    if (gpuAddress == 0) {
      buffer->release();
      return M_DRIVER_STATUS_OUT_OF_MEMORY;
    }
    *memory = reinterpret_cast<void *>(gpuAddress);
    context->trackBuffer(reinterpret_cast<void *>(gpuAddress), buffer);
    return M_DRIVER_STATUS_SUCCESS;
  }
  }

  buffer->release();
  return M_DRIVER_STATUS_UNKNOWN_ERROR;
}

M_driver_status_t M_driver_memory_free(M_driver_context_t context,
                                       void *memory) {
  if (!context)
    return M_DRIVER_STATUS_BAD_CONTEXT;
  if (!memory)
    return M_DRIVER_STATUS_SUCCESS;

  auto autoreleasePool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  MTL::Buffer *buf = context->getTrackedBuffer(memory);
  if (!buf)
    return M_DRIVER_STATUS_BAD_MEMORY;

  buf->release();
  context->untrackBuffer(memory);

  return M_DRIVER_STATUS_SUCCESS;
}
