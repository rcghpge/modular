//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Memory.h"

using M::Target::CUDA::cudaErrorStatus;

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  CUcontext primaryCtx;
  if (CUresult r = dev.driver.rawCuda.cuDevicePrimaryCtxRetain(&primaryCtx,
                                                               dev.deviceId);
      r != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&dev.driver, r, "cuDevicePrimaryCtxRetain"));

  return new M_driver_context(dev, primaryCtx);
}

M_driver_status_t M_driver_context::destroy(M_driver_context_t ctx) {
  if (CUresult r = ctx->device.driver.rawCuda.cuDevicePrimaryCtxRelease(
          ctx->device.deviceId);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&ctx->device.driver, r, "cuDevicePrimaryCtxRelease");
  delete ctx;
  return M_DRIVER_STATUS_SUCCESS;
}

DriverResult<M_driver_memory_t> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  void *memAlloc = nullptr;
  CUresult result = driver.cuMemAllocHost(&memAlloc, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&device.driver, result, "cuMemAllocHost"));
  M_driver_memory_t memory = new (std::nothrow) M_driver_memory(memAlloc);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

DriverResult<M_driver_memory_t> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  RETURN_IF_ERROR(scope.status, tl::unexpected(status));
  auto &driver = device.driver.rawCuda;
  CUdeviceptr devicePtr;
  CUresult result = driver.cuMemAlloc(&devicePtr, size);
  if (result == CUDA_ERROR_OUT_OF_MEMORY)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != CUDA_SUCCESS)
    return tl::unexpected(
        cudaErrorStatus(&device.driver, result, "cuMemAlloc"));
  M_driver_memory_t memory =
      new (std::nothrow) M_driver_memory(reinterpret_cast<void *>(devicePtr));
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

DriverResult<M_driver_memory_t>
M_driver_context::wrapSync(uint64_t address, uint64_t size, bool owning) {
  (void)size;
  if (address == 0)
    return tl::unexpected(M_DRIVER_STATUS_INVALID_ARG);
  M_driver_memory_t memory = new (std::nothrow)
      M_driver_memory{reinterpret_cast<void *>(address), owning};
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_context::freePinned(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &driver = device.driver.rawCuda;
  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  if (CUresult r = driver.cuMemFreeHost(ptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemFreeHost");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &driver = device.driver.rawCuda;
  CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(memory->ptr);
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;
  if (CUresult r = driver.cuMemFree(devicePtr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemFree");
  return M_DRIVER_STATUS_SUCCESS;
}
