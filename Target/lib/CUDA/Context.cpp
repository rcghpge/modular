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
  memory->hostAccessible = true;
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

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyHtoD(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset, src, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyHtoD");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoH(
          dst, reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyDtoH");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyDtoD(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyDtoD");
  // The synchronous device-to-device memcpy does NOT host-block (CUDA performs
  // no host-side synchronization for device-to-device transfers), so wait on
  // the default stream it ran on to honor the blocking contract.
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuStreamSynchronize (sync D2D)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyInterDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size, M_driver_context_t srcContext) {
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  auto &cuda = device.driver.rawCuda;
  if (CUresult r = cuda.cuMemcpyPeer(
          reinterpret_cast<CUdeviceptr>(dst->ptr) + dstOffset, primaryCtx,
          reinterpret_cast<CUdeviceptr>(src->ptr) + srcOffset,
          srcContext->primaryCtx, size);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemcpyPeer");
  // Like the D2D case above, the synchronous peer copy does not host-block, so
  // wait on the default stream to honor the blocking contract.
  if (CUresult r = cuda.cuStreamSynchronize(nullptr); r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r,
                           "cuStreamSynchronize (sync peer)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrent() {
  if (CUresult r = device.driver.rawCuda.cuCtxSetCurrent(primaryCtx);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxSetCurrent");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::getCurrentDriverContext(
    M_driver_native_handle_t *driver_context) {
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  CUcontext current = nullptr;
  if (CUresult r = device.driver.rawCuda.cuCtxGetCurrent(&current);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxGetCurrent");
  *driver_context = reinterpret_cast<M_driver_native_handle_t>(current);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrentDriverContext(
    M_driver_native_handle_t driver_context) {
  if (CUresult r = device.driver.rawCuda.cuCtxSetCurrent(
          reinterpret_cast<CUcontext>(driver_context));
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuCtxSetCurrent");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::memoryInfo(uint64_t *free,
                                               uint64_t *total) {
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  // cuMemGetInfo reports the *current* context's memory, so make this context
  // current for the duration of the query (restored on scope exit).
  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  if (scope.status != M_DRIVER_STATUS_SUCCESS)
    return scope.status;
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  if (CUresult r = device.driver.rawCuda.cuMemGetInfo(&freeBytes, &totalBytes);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemGetInfo");
  *free = freeBytes;
  *total = totalBytes;
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t
M_driver_context::allocPinnedMapped(uint64_t size, M_driver_memory_t *memory,
                                    void **host_ptr, uint64_t *device_address) {
  if (memory == nullptr || host_ptr == nullptr || device_address == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  if (size == 0)
    return M_DRIVER_STATUS_INVALID_ARG;

  ScopedContext scope(primaryCtx, device.driver.rawCuda);
  QUICK_RETURN_IF_ERROR(scope.status);
  M::Target::CUDA::CUDALoader &cuda = device.driver.rawCuda;

  // DEVICEMAP is required so cuMemHostGetDevicePointer succeeds on non-UVA
  // systems; PORTABLE lets any context in the process use the allocation
  // (mirrors CUDADeviceContext::allocateCompletionFlag).
  void *p = nullptr;
  if (CUresult r = cuda.cuMemHostAlloc(
          &p, size, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
      r != CUDA_SUCCESS)
    return cudaErrorStatus(&device.driver, r, "cuMemHostAlloc");
  CUdeviceptr devPtr = 0;
  if (CUresult r = cuda.cuMemHostGetDevicePointer(&devPtr, p, 0);
      r != CUDA_SUCCESS) {
    (void)cuda.cuMemFreeHost(p);
    return cudaErrorStatus(&device.driver, r, "cuMemHostGetDevicePointer");
  }
  M_driver_memory_t wrapper = new (std::nothrow) M_driver_memory(p);
  if (!wrapper) {
    (void)cuda.cuMemFreeHost(p);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  *memory = wrapper;
  *host_ptr = p;
  *device_address = static_cast<uint64_t>(devPtr);
  return M_DRIVER_STATUS_SUCCESS;
}
