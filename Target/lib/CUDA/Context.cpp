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
      dev.driver.rawCuda.cuDevicePrimaryCtxRetain(&primaryCtx, dev.deviceId),
      M_DRIVER_STATUS_BAD_DEVICE);

  return new M_driver_context(dev, primaryCtx);
}

M_driver_status_t M_driver_context::ensureContextBound() {
  auto &driver = device.driver.rawCuda;
  CUcontext current;
  CUDA_CHECK(driver.cuCtxGetCurrent(&current), M_DRIVER_STATUS_BAD_CONTEXT);
  if (current == primaryCtx)
    return M_DRIVER_STATUS_SUCCESS;
  CUDA_CHECK(driver.cuCtxSetCurrent(primaryCtx), M_DRIVER_STATUS_BAD_CONTEXT);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  CUDA_CHECK(ctx->device.driver.rawCuda.cuDevicePrimaryCtxRelease(
                 ctx->device.deviceId),
             M_DRIVER_STATUS_BAD_CONTEXT);
  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<void *> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;
  RETURN_IF_ERROR(ensureContextBound(), tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  void *memAlloc = nullptr;
  CUresult result = driver.cuMemAllocHost(&memAlloc, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  return memAlloc;
}

DriverResult<void *> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;

  RETURN_IF_ERROR(ensureContextBound(), tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  CUdeviceptr devicePtr;
  CUresult result = driver.cuMemAlloc(&devicePtr, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);
  return reinterpret_cast<void *>(devicePtr);
}

M_driver_status_t M_driver_context::freePinned(void *memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  QUICK_RETURN_IF_ERROR(ensureContextBound());
  auto &driver = device.driver.rawCuda;
  CUresult result = driver.cuMemFreeHost(memory);
  if (result != CUDA_SUCCESS)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(void *memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  QUICK_RETURN_IF_ERROR(ensureContextBound());
  auto &driver = device.driver.rawCuda;
  CUresult result = driver.cuMemFree(reinterpret_cast<CUdeviceptr>(memory));
  if (result != CUDA_SUCCESS)
    return M_DRIVER_STATUS_UNKNOWN_ERROR;
  return M_DRIVER_STATUS_SUCCESS;
}
