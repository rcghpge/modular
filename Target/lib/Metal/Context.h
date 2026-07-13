//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_CONTEXT_H
#define TARGET_LIB_METAL_CONTEXT_H

#include "MetalCpp.h"
#include "Target/lib/Metal/Common.h"

#include <map>

struct M_driver_context {
  explicit M_driver_context(M_driver_device_t device) : device(device) {}

  [[nodiscard]] static M_driver_context_t create(M_driver_device_t device);
  static void destroy(M_driver_context_t context);

  DriverResult<M_driver_memory_t> alloc(uint64_t size);
  M_driver_status_t free(M_driver_memory_t memory);

  M_driver_status_t copyToDeviceSync(M_driver_memory_t dst, uint64_t dstOffset,
                                     const void *src, uint64_t size);
  M_driver_status_t copyFromDeviceSync(void *dst, M_driver_memory_t src,
                                       uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDeviceSync(M_driver_memory_t dst,
                                        uint64_t dstOffset,
                                        M_driver_memory_t src,
                                        uint64_t srcOffset, uint64_t size);

  void trackBuffer(void *ptr, MTL::Buffer *buffer) {
    allocations[reinterpret_cast<uintptr_t>(ptr)] = buffer;
  }

  void untrackBuffer(void *ptr) {
    allocations.erase(reinterpret_cast<uintptr_t>(ptr));
  }

  MTL::Buffer *getTrackedBuffer(void *ptr) {
    auto it = allocations.find(reinterpret_cast<uintptr_t>(ptr));
    if (it != allocations.end())
      return it->second;
    return nullptr;
  }

  bool isTrackedBuffer(void *ptr) {
    return allocations.find(reinterpret_cast<uintptr_t>(ptr)) !=
           allocations.end();
  }

  struct BufferWithOffset {
    MTL::Buffer *buffer;
    size_t offset;

    bool isValid() { return buffer != nullptr; }
  };

  BufferWithOffset getBufferWithOffset(const void *ptr) const {
    uintptr_t ptrAddress = reinterpret_cast<uintptr_t>(ptr);
    // Find first allocation with base address > addr
    auto it = allocations.upper_bound(ptrAddress);
    if (it != allocations.begin()) {
      --it; // Step back to candidate allocation
      MTL::Buffer *buffer = it->second;
      auto bufferAddress = uintptr_t(buffer->contents());
      if (ptrAddress >= bufferAddress &&
          ptrAddress < bufferAddress + buffer->length()) {
        return {buffer, static_cast<size_t>(ptrAddress - bufferAddress)};
      }
    }
    return {nullptr, 0};
  }

  M_driver_device_t device;
  std::map<uintptr_t, MTL::Buffer *> allocations;
};

#endif // TARGET_LIB_METAL_CONTEXT_H
