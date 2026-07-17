//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Driver.h"
#include "Error.h"
#include "Memory.h"

#include <new>

using M::Target::HIP::hipErrorStatus;

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  return new M_driver_context(dev);
}

DriverResult<M_driver_memory_t> M_driver_context::allocPinned(uint64_t size) {
  if (size == 0)
    return nullptr;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  void *ptr = nullptr;
  hipError_t result = hip.hipHostMalloc(&ptr, size, hipHostMallocDefault);
  if (result == hipErrorOutOfMemory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), result, "hipHostMalloc"));

  auto *memory = new (std::nothrow) M_driver_memory(ptr);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  // Pinned host memory is host-dereferenceable.
  memory->hostAccessible = true;

  return memory;
}

DriverResult<M_driver_memory_t> M_driver_context::allocSync(uint64_t size) {
  if (size == 0)
    return nullptr;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  void *ptr = nullptr;
  hipError_t result = hip.hipMalloc(&ptr, size);
  if (result == hipErrorOutOfMemory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != hipSuccess)
    return tl::unexpected(
        hipErrorStatus(&dev.getDriver(), result, "hipMalloc"));

  auto *memory = new (std::nothrow) M_driver_memory(ptr);
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);

  return memory;
}

DriverResult<M_driver_memory_t>
M_driver_context::wrapSync(uint64_t address, uint64_t size, bool owning) {
  (void)size;
  if (address == 0)
    return tl::unexpected(M_DRIVER_STATUS_INVALID_ARG);
  auto *memory = new (std::nothrow)
      M_driver_memory{reinterpret_cast<void *>(address), owning};
  if (!memory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  return memory;
}

M_driver_status_t M_driver_context::freePinned(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipHostFree(ptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipHostFree");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::freeSync(M_driver_memory_t memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  void *ptr = memory->ptr;
  bool owning = memory->owning;
  delete memory;
  if (!owning)
    return M_DRIVER_STATUS_SUCCESS;

  HIPApi &hip = dev.getDriver().hip();
  if (hipError_t r = hip.hipFree(ptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipFree");

  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyToDeviceSync(M_driver_memory_t dst,
                                                     uint64_t dstOffset,
                                                     const void *src,
                                                     uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  if (hipError_t r = hip.hipMemcpy(dstPtr, src, size, hipMemcpyHostToDevice);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy HtoD");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyFromDeviceSync(void *dst,
                                                       M_driver_memory_t src,
                                                       uint64_t srcOffset,
                                                       uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpy(dst, srcPtr, size, hipMemcpyDeviceToHost);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy DtoH");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyIntraDeviceSync(M_driver_memory_t dst,
                                                        uint64_t dstOffset,
                                                        M_driver_memory_t src,
                                                        uint64_t srcOffset,
                                                        uint64_t size) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r =
          hip.hipMemcpy(dstPtr, srcPtr, size, hipMemcpyDeviceToDevice);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpy DtoD");
  // hipMemcpy is synchronous, but wait on the default stream so the blocking
  // contract holds by construction (matching the CUDA sync path).
  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync D2D)");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::copyInterDeviceSync(
    M_driver_memory_t dst, uint64_t dstOffset, M_driver_memory_t src,
    uint64_t srcOffset, uint64_t size, M_driver_context_t srcContext) {
  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();
  HIPApi &hip = dev.getDriver().hip();
  void *dstPtr = static_cast<char *>(dst->ptr) + dstOffset;
  void *srcPtr = static_cast<char *>(src->ptr) + srcOffset;
  if (hipError_t r = hip.hipMemcpyPeer(dstPtr, dev.id(), srcPtr,
                                       srcContext->getDevice().id(), size);
      r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r, "hipMemcpyPeer");
  // Wait on the default stream so the blocking contract holds by construction
  // (matching the CUDA sync peer path).
  if (hipError_t r = hip.hipStreamSynchronize(nullptr); r != hipSuccess)
    return hipErrorStatus(&dev.getDriver(), r,
                          "hipStreamSynchronize (sync peer)");
  return M_DRIVER_STATUS_SUCCESS;
}

// HIP has no driver-level current-context object; the per-thread state is the
// current device id, which round-trips through the opaque native handle.

M_driver_status_t M_driver_context::setCurrent() {
  if (hipError_t r = device.getDriver().hip().hipSetDevice(device.id());
      r != hipSuccess)
    return hipErrorStatus(&device.getDriver(), r, "hipSetDevice");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::getCurrentDriverContext(
    M_driver_native_handle_t *driver_context) {
  if (driver_context == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = device.getDriver();
  int currentId = -1;
  if (hipError_t r = driver.hip().hipGetDevice(&currentId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipGetDevice");
  *driver_context = static_cast<M_driver_native_handle_t>(currentId);
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::setCurrentDriverContext(
    M_driver_native_handle_t driver_context) {
  M_driver_driver &driver = device.getDriver();
  int targetId = static_cast<int>(driver_context);
  if (hipError_t r = driver.hip().hipSetDevice(targetId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipSetDevice");
  return M_DRIVER_STATUS_SUCCESS;
}

M_driver_status_t M_driver_context::memoryInfo(uint64_t *free,
                                               uint64_t *total) {
  if (free == nullptr || total == nullptr)
    return M_DRIVER_STATUS_INVALID_ARG;
  M_driver_driver &driver = device.getDriver();
  // hipMemGetInfo reports the *current* device's memory. Make this context's
  // device current for the query, then restore the previous device (mirrors
  // the CUDA plugin's ScopedContext save/restore).
  int prevId = -1;
  if (hipError_t r = driver.hip().hipGetDevice(&prevId); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipGetDevice");
  if (hipError_t r = driver.hip().hipSetDevice(device.id()); r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipSetDevice");
  size_t freeBytes = 0;
  size_t totalBytes = 0;
  hipError_t r = driver.hip().hipMemGetInfo(&freeBytes, &totalBytes);
  (void)driver.hip().hipSetDevice(prevId);
  if (r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipMemGetInfo");
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

  if (auto preamble = device.deviceMethodPreamble(); !preamble)
    return preamble.error();

  M_driver_driver &driver = device.getDriver();
  // Mirrors HIPDeviceContext::allocateCompletionFlag.
  void *p = nullptr;
  if (hipError_t r = driver.hip().hipHostMalloc(
          &p, size, hipHostMallocPortable | hipHostMallocMapped);
      r != hipSuccess)
    return hipErrorStatus(&driver, r, "hipHostMalloc");
  void *devPtr = nullptr;
  if (hipError_t r = driver.hip().hipHostGetDevicePointer(&devPtr, p, 0);
      r != hipSuccess) {
    (void)driver.hip().hipHostFree(p);
    return hipErrorStatus(&driver, r, "hipHostGetDevicePointer");
  }
  M_driver_memory_t wrapper = new (std::nothrow) M_driver_memory{p};
  if (!wrapper) {
    (void)driver.hip().hipHostFree(p);
    return M_DRIVER_STATUS_OUT_OF_MEMORY;
  }
  *memory = wrapper;
  *host_ptr = p;
  *device_address = reinterpret_cast<uint64_t>(devPtr);
  return M_DRIVER_STATUS_SUCCESS;
}
