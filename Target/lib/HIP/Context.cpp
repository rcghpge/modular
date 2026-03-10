//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Context.h"
#include "Device.h"
#include "Error.h"
#include "HAL/M_driver_memory.h"

DriverResult<M_driver_context_t>
M_driver_context::create(M_driver_device &dev) {
  return new M_driver_context(dev);
}

DriverResult<void *> M_driver_context::alloc(M_driver_alloc_mode_t mode,
                                             uint64_t size) {
  if (mode == M_DRIVER_ALLOC_MODE_ASYNC)
    // async allocation without a stream/queue doesn't make any sense
    return tl::unexpected(M_DRIVER_STATUS_UNIMPLEMENTED);

  if (size == 0)
    return nullptr;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return tl::unexpected(preamble.error());

  HIPApi &hip = dev.getDriver().hip();
  void *ptr = nullptr;
  hipError_t result = hipErrorUnknown;

  switch (mode) {
  case M_DRIVER_ALLOC_MODE_HOST:
    result = hip.hipHostMalloc(&ptr, size, hipHostMallocDefault);
    break;
  case M_DRIVER_ALLOC_MODE_SYNC:
    result = hip.hipMalloc(&ptr, size);
    break;
  case M_DRIVER_ALLOC_MODE_ASYNC:
    // TODO: mark unreachable in actually standard way
    llvm_unreachable(
        "MODE_ASYNC is caught earlier, and ill-defined without a queue");
    break;
  }

  if (result == hipErrorOutOfMemory)
    return tl::unexpected(M_DRIVER_STATUS_OUT_OF_MEMORY);
  if (result != hipSuccess)
    return tl::unexpected(M_DRIVER_STATUS_UNKNOWN_ERROR);

  trackAlloc(ptr, {.allocMode = mode});
  return ptr;
}

M_driver_status_t M_driver_context::free(void *memory) {
  if (memory == nullptr)
    return M_DRIVER_STATUS_SUCCESS;

  M_driver_device &dev = getDevice();
  if (auto preamble = dev.deviceMethodPreamble(); !preamble)
    return preamble.error();

  const AllocInfo *info = getAllocInfo(memory);
  if (info == nullptr)
    return M_DRIVER_STATUS_BAD_MEMORY;

  if (info->allocMode == M_DRIVER_ALLOC_MODE_ASYNC)
    // async allocation without a stream/queue doesn't make any sense
    return M_DRIVER_STATUS_UNIMPLEMENTED;

  HIPApi &hip = dev.getDriver().hip();
  hipError_t result;

  switch (info->allocMode) {
  case M_DRIVER_ALLOC_MODE_HOST:
    result = hip.hipHostFree(memory);
    break;
  case M_DRIVER_ALLOC_MODE_SYNC:
    result = hip.hipFree(memory);
    break;
  case M_DRIVER_ALLOC_MODE_ASYNC:
    // TODO: mark unreachable in actually standard way
    llvm_unreachable(
        "MODE_ASYNC is caught earlier, and ill-defined without a queue");
    break;
  }

  if (result == hipSuccess) {
    untrackAlloc(memory);
    return M_DRIVER_STATUS_SUCCESS;
  }
  return M_DRIVER_STATUS_UNKNOWN_ERROR;
}
