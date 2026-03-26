//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "MetalCpp.h"

struct M_driver_queue {
  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context_t context);
  static M_driver_status_t destroy(M_driver_queue_t);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, void *src,
                                 uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t size,
                              uint64_t value, uint64_t valueSize);

  M_driver_context_t context;
  NS::SharedPtr<MTL::CommandQueue> mCommandQueue;
};
