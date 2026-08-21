//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Regression tests for GEX-4095 / MXSERV-344. The MemoryManager outlives every
// DeviceContext, and on a VMM allocation miss it used to reach back into
// sibling contexts to fire their deferred free-list flushes. selectStream()
// hands out short-lived view contexts, so that reach-back could land on a view
// being torn down: deadlock or use-after-teardown, reproduced as a 300s hang on
// B200.
//
// These tests drive that interleaving through public APIs only -- concurrent
// side-stream view churn against base-context allocation pressure heavy enough
// to miss -- so they stay meaningful regardless of how the miss path recovers
// memory. Note this is a race: a passing run is evidence, not proof.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

class SelectStreamSiblingFreeListTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available";
#if MODULAR_MACOS
    GTEST_SKIP() << "GPU memory manager disabled on macOS";
#endif
    Device::API api = (*device)->api();
    if (api != Device::cudaAPI && api != Device::hipAPI)
      GTEST_SKIP() << "selectStream side-stream views require CUDA or HIP";
  }

  ErrorOr<DeviceContextRef> baseContext() {
    return DeviceContext::create(Device::gpuAPI);
  }

  void ensureSideStream(DeviceContext &base) {
    if (base.numStreams() >= 2)
      return;
    ErrorOr<DeviceStreamRef> stream = base.createStream();
    ASSERT_FALSE(stream.isError()) << stream.getError();
  }

  static void dropViewAfterWork(DeviceContextRef &view) {
    EXPECT_NO_ERR(view->pollPendingEvents());
    EXPECT_NO_ERR(view->synchronize());
    view.reset();
  }
};

TEST_F(SelectStreamSiblingFreeListTest, SelectStreamSideStreamAllocates) {
  ASSIGN_NO_ERR(DeviceContextRef, base, baseContext());
  ensureSideStream(*base);

  ASSIGN_NO_ERR(DeviceContextRef, view, base->selectStream(1));
  ASSERT_NE(view.getPointer(), base.getPointer());

  ASSIGN_NO_ERR(DeviceBufferRef, buffer, view->enqueueCreateBuffer(4096));
  EXPECT_NE(buffer->ptr(), nullptr);
  EXPECT_NO_ERR(view->synchronize());
  dropViewAfterWork(view);
}

TEST_F(SelectStreamSiblingFreeListTest,
       ConcurrentViewChurnUnderSiblingFlushPressure) {
  ASSIGN_NO_ERR(DeviceContextRef, base, baseContext());
  ensureSideStream(*base);
  if (!base->device().memoryManager().isVmmEnabled())
    GTEST_SKIP() << "the sibling-recovery path runs only on a VMM allocation "
                    "miss; enable VMM in this test target's env";

  // The sibling drain is only reached once the pool cannot satisfy a request
  // from cache, so a workload that always hits would never enter the path this
  // test is here to race against. Against the target's 1 GiB budget, hold half
  // and then ask for more than the remainder: the miss is deterministic, and
  // capping what we hold leaves the view contexts room to keep allocating
  // rather than starving them into spurious failures.
  constexpr size_t kMiB = 1024 * 1024;
  constexpr size_t kHoldBytes = 16 * kMiB;
  constexpr int kHoldBlocks = 32;           // 512 MiB held
  constexpr size_t kMissBytes = 768 * kMiB; // > the ~512 MiB left
  std::atomic<uint64_t> missRounds{0};

  std::atomic<bool> stop{false};
  std::thread allocPressure([&]() {
    MemoryManagerFreeList &baseFreeList = base->freeList();
    const Device::AllocMode mode = Device::AllocMode::SYNC;
    std::vector<void *> livePtrs;
    while (!stop.load(std::memory_order_relaxed)) {
      for (int i = 0; i < kHoldBlocks; ++i) {
        ErrorOr<void *> ptr = baseFreeList.allocate(kHoldBytes, mode);
        if (ptr.isError())
          break;
        livePtrs.push_back(ptr.takeValue());
      }
      // Cannot fit: drives the miss path, which drains sibling freelists while
      // views are being retired underneath it.
      ErrorOr<void *> miss = baseFreeList.allocate(kMissBytes, mode);
      if (miss.isError())
        missRounds.fetch_add(1, std::memory_order_relaxed);
      else
        EXPECT_NO_ERR(baseFreeList.free(miss.takeValue()));

      for (void *ptr : livePtrs)
        EXPECT_NO_ERR(baseFreeList.free(ptr));
      livePtrs.clear();
      EXPECT_NO_ERR(base->pollPendingEvents());
    }
  });

  constexpr int kIterations = 512;
  for (int i = 0; i < kIterations; ++i) {
    ASSIGN_NO_ERR(DeviceContextRef, view, base->selectStream(1));
    {
      ASSIGN_NO_ERR(DeviceBufferRef, buffer, view->enqueueCreateBuffer(4096));
      EXPECT_NE(buffer->ptr(), nullptr);
      EXPECT_NO_ERR(view->synchronize());
    }
    dropViewAfterWork(view);
  }

  stop.store(true, std::memory_order_relaxed);
  allocPressure.join();

  // Guards the test against silently degrading into a no-op if the budget or
  // the miss path changes: with no miss, nothing ever drains the siblings.
  EXPECT_GT(missRounds.load(), 0u)
      << "allocation never missed, so the sibling drain was never exercised";
}

} // namespace
