//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_HIP_QUEUE_H
#define TARGET_LIB_HIP_QUEUE_H

#include "HIPApi.h"
#include "Memory.h"

struct M_driver_context;

struct M_driver_queue {
  M_driver_context &context;
  hipStream_t stream;

  explicit M_driver_queue(M_driver_context &context, hipStream_t stream)
      : context(context), stream(stream) {}

  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context &ctx);

  static M_driver_status_t destroy(M_driver_queue_t queue);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, void *src,
                                 uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t size,
                              uint64_t value, uint64_t valueSize);

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvent(M_driver_event_t event);

  M_driver_status_t synchronize();
};

#endif // TARGET_LIB_HIP_QUEUE_H
