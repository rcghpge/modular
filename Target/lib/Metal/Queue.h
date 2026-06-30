//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "HAL/M_driver_queue.h"
#include "MetalCpp.h"

#include <mutex>
#include <vector>

struct M_driver_queue {
  [[nodiscard]] static DriverResult<M_driver_queue_t>
  create(M_driver_context_t context);
  static M_driver_status_t destroy(M_driver_queue_t);

  M_driver_status_t copyToDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                 const void *src, uint64_t size);
  M_driver_status_t copyFromDevice(void *dst, M_driver_memory_t src,
                                   uint64_t srcOffset, uint64_t size);
  M_driver_status_t copyIntraDevice(M_driver_memory_t dst, uint64_t dstOffset,
                                    M_driver_memory_t src, uint64_t srcOffset,
                                    uint64_t size);
  M_driver_status_t setMemory(M_driver_memory_t dst, uint64_t dstOffset,
                              uint64_t size, uint8_t value);
  M_driver_status_t fillMemory(M_driver_memory_t dst, uint64_t dstOffset,
                               uint64_t size, uint64_t value,
                               uint64_t valueSize);

  M_driver_status_t synchronize();

  M_driver_status_t recordEvent(M_driver_event_t event);
  M_driver_status_t waitForEvents(M_driver_event_t *events, uint32_t numEvents);

  M_driver_status_t execute(M_driver_function_t function,
                            M_driver_queue_execute_config_t *config,
                            void **args, uint64_t *argSizes, uint32_t numArgs);

  void encodePendingWaits(MTL::CommandBuffer *cb);

  M_driver_context_t context;
  NS::SharedPtr<MTL::CommandQueue> mCommandQueue;

  // Completion events for in-flight ops on this queue.
  //
  // Every queue op builds its own MTL::CommandBuffer and ends it with
  // `encodeSignalEvent` on a freshly created event — the "done
  // event" for that op. We push that done event into `mPending` so the *next*
  // op's CB can encode-wait on every entry here before doing its own work if
  // necessary.
  //
  // mPending is cleared out in two places:
  //   - `synchronize()`: the queue has been drained, nothing in-flight.
  //   - `recordEvent()`: the user's event is encoded to signal after every
  //     current entry, then *replaces* mPending as the sole gating event —
  //     anyone waiting on the user event transitively waits on everything
  //     that came before, so we can drop the prior entries.
  std::vector<NS::SharedPtr<MTL::Event>> mPending;

  // External events stashed by `waitForEvents` to encode during subsequent
  // operations. This is required because each operation creates a new
  // command-buffer for encoding.
  std::vector<NS::SharedPtr<MTL::Event>> mPendingWaits;

  std::mutex mMutex;
};
