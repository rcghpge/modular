//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Error.h"
#include "Memory.h"

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
