//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_CPU_QUEUE_H
#define TARGET_LIB_CPU_QUEUE_H

#include "HAL/M_driver_common.h"
#include "HAL/M_driver_queue.h"
#include "MLRT/AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/AsyncRT/Runtime/WorkQueue.h"

#include <cstdint>
#include <mutex>

namespace MLRT = M::MLRT;

/// A CPU queue is an ordered chain of async work items. Each enqueued
/// operation is sequenced after the previous one via `streamChain`, exactly
/// as M::Driver::CpuDeviceContext sequences work on its single stream. Because
/// the chain is totally ordered, the queue reports itself as a stream.
struct M_driver_queue {
  explicit M_driver_queue(M_driver_context_t context,
                          MLRT::AnyAsyncValueRef streamChain)
      : context(context), streamChain(std::move(streamChain)) {}

  [[nodiscard]] static M_driver_queue_t create(M_driver_context_t context);
  static M_driver_status_t destroy(M_driver_queue_t queue);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                 const void *src, uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t offset,
                              uint64_t size, uint8_t value);
  M_driver_status_t fillMemory(M_driver_memory_t dst, uint64_t offset,
                               uint64_t size, uint64_t value,
                               uint64_t valueSize);

  M_driver_status_t freeAsync(M_driver_memory_t memory);

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvents(M_driver_event_t *events, uint32_t numEvents);

  M_driver_status_t synchronize();

  M_driver_context_t context;

  // Chain of the last enqueued operation, which may already have completed.
  // Guarded by `streamChainMutex`.
  MLRT::AnyAsyncValueRef streamChain;
  std::mutex streamChainMutex;

private:
  // Enqueue `work` after the current chain tail, advancing the tail. Mirrors
  // CpuDeviceContext::enqueueWork.
  M_driver_status_t enqueueWork(MLRT::TaskFunction &&work);
};

#endif // TARGET_LIB_CPU_QUEUE_H
