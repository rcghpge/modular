//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
#include "Context.h"
#include "Device.h"

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  CUcontext primaryCtx;
  CUDA_CHECK_DR(
      dev.getDriver().cuda().cuDevicePrimaryCtxRetain(&primaryCtx, dev.id()),
      M_DRIVER_STATUS_BAD_DEVICE);

  return new M_driver_context(dev, primaryCtx);
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  CUDA_CHECK(ctx->device.getDriver().cuda().cuDevicePrimaryCtxRelease(
                 ctx->device.id()),
             M_DRIVER_STATUS_BAD_CONTEXT);
  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<void *> M_driver_context::alloc(M_driver_alloc_mode_t mode,
                                             uint64_t size) {
  if (size == 0)
    return nullptr;

  auto &driver = device.getDriver().cuda();
  CUDA_CHECK_DR(driver.cuCtxSetCurrent(primaryCtx),
                M_DRIVER_STATUS_BAD_CONTEXT);

  void *memAlloc = nullptr;
  CUresult result = CUDA_SUCCESS;

  switch (mode) {
  case M_DRIVER_ALLOC_MODE_HOST:
    result = driver.cuMemAllocHost(&memAlloc, size);
    break;
  case M_DRIVER_ALLOC_MODE_SYNC:
    CUdeviceptr devicePtr;
    result = driver.cuMemAlloc(&devicePtr, size);
    memAlloc = reinterpret_cast<void *>(devicePtr);
    break;
  case M_DRIVER_ALLOC_MODE_ASYNC:
    break;
  }

  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  trackAlloc(memAlloc, {.allocMode = mode});
  return memAlloc;
}

M_driver_status_t M_driver_context::free(void *memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  const AllocInfo *info = getAllocInfo(memory);
  if (info == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;

  auto &driver = device.getDriver().cuda();
  CUDA_CHECK(driver.cuCtxSetCurrent(primaryCtx), M_DRIVER_STATUS_BAD_CONTEXT);

  CUresult result = CUDA_SUCCESS;

  switch (info->allocMode) {
  case M_DRIVER_ALLOC_MODE_HOST:
    result = driver.cuMemFreeHost(memory);
    break;
  case M_DRIVER_ALLOC_MODE_SYNC:
    result = driver.cuMemFree(reinterpret_cast<CUdeviceptr>(memory));
    break;
  case M_DRIVER_ALLOC_MODE_ASYNC:
    break;
  }

  if (result == CUDA_SUCCESS) {
    untrackAlloc(memory);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_ERROR;
}
