//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for VMM partial-chunk harvest ("within-a-chunk defragmentation"). VMM
// is enabled process-wide via the target's `env` with chunk_percent=100, so the
// whole pool is a SINGLE chunk. The load-bearing test fragments that one chunk
// internally (free every other granule-sized buffer) so the largest contiguous
// free block is smaller than a later request, yet no chunk is wholly free. With
// memory-manager-only mode there is no driver backdoor, so the request can only
// succeed if defrag harvests whole free granules from inside the live chunk.
// This is the case the wholly-free-chunk-only defrag (MemoryManagerVMMTest, run
// at chunk_percent=25) cannot satisfy. Single-GPU (owner-device access only).
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <vector>

using namespace M::Driver;
using M::Error;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

constexpr size_t kGiB = size_t(1024) * 1024 * 1024;
constexpr size_t kMiB = size_t(1024) * 1024;

class MemoryManagerVMMPartialHarvestTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      GTEST_SKIP() << "GPU not available, skipping VMM tests";
#if MODULAR_MACOS
    GTEST_SKIP() << "VMM not supported on macOS, skipping VMM tests";
#else
    if (!(*device)->vmmSupported())
      GTEST_SKIP() << "device does not support VMM, skipping VMM tests";
#endif
  }

  ErrorOr<DeviceRef> dev() { return Device::create(Device::gpuAPI); }

  // Fresh, independent device MemoryManager with an explicit byte budget. VMM,
  // chunk-percent (100), memory-manager-only and self-check come from the env.
  std::unique_ptr<MemoryManager> makeCache(Device &device, size_t budgetBytes) {
    ErrorOr<Device::MemInfo> memInfo = device.getMemoryInfo();
    if (memInfo.isError())
      return nullptr;
    return std::unique_ptr<MemoryManager>(MemoryManager::onDevice(
        device, *memInfo, device.allocationAlignment(),
        /*memoryLimit=*/static_cast<int64_t>(budgetBytes)));
  }
};

// Write a sentinel into the first bytes of every `pageSize`-sized slice of
// [ptr, ptr+size) and read it back, proving the whole range is physically
// backed and accessible (an unbacked granule would fault here).
ErrorOrSuccess writeAndVerify(DeviceContext &ctx, void *ptr, size_t size,
                              size_t pageSize) {
  const std::vector<uint8_t> pattern(64, 0xAB);
  for (size_t off = 0; off < size; off += pageSize) {
    ErrorOrSuccess e = ctx.enqueueCopyToDevice(static_cast<char *>(ptr) + off,
                                               pattern.data(), pattern.size());
    if (e.isError())
      return e.takeError();
  }
  if (ErrorOrSuccess e = ctx.synchronize(); e.isError())
    return e.takeError();

  for (size_t off = 0; off < size; off += pageSize) {
    std::vector<uint8_t> readback(64, 0);
    ErrorOrSuccess e = ctx.enqueueCopyFromDevice(
        readback.data(), static_cast<char *>(ptr) + off, readback.size());
    if (e.isError())
      return e.takeError();
    if (ErrorOrSuccess s = ctx.synchronize(); s.isError())
      return s.takeError();
    if (readback != pattern)
      return Error("VMM buffer read-back mismatch: range not correctly backed");
  }
  return M::success();
}

// The fix: a 1 GiB request that exceeds the largest contiguous free block
// (256 MiB) succeeds by harvesting whole free granules from inside the single,
// partially-used chunk. Also asserts the harvest left the still-in-use buffers
// intact and that the allocator never hands out an unbacked hole.
TEST_F(MemoryManagerVMMPartialHarvestTest,
       HarvestReclaimsGranulesFromInsideOneChunk) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB; // chunk_percent=100 => one 4 GiB chunk
  const size_t kGran = 256 * kMiB; // == VMM granule (VMM_PAGE_MB=256)
  const size_t kBig = 1 * kGiB;    // 4 granules; > largest contiguous (256 MiB)

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // Fill the single chunk with 16 granule-sized buffers.
  std::vector<void *> bufs;
  for (int i = 0; i < 16; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kGran, mode));
    bufs.push_back(p);
  }

  // Free every other buffer: 2 GiB free, but as 8 scattered 256 MiB holes
  // inside ONE partially-used chunk. No chunk is wholly free and
  // (memory-manager-only) there is no backdoor, so the wholly-free-chunk defrag
  // path cannot help.
  for (size_t i = 0; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));

  // The request can only succeed by harvesting whole free granules from inside
  // the live chunk; prove the assembled buffer is fully backed across granules.
  ASSIGN_NO_ERR(void *, big, fl.allocate(kBig, mode));
  EXPECT_NE(big, nullptr);
  EXPECT_NO_ERR(writeAndVerify(*ctx, big, kBig, kGran));

  // Harvest must not have disturbed the still-in-use buffers (odd indices).
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(writeAndVerify(*ctx, bufs[i], kGran, kGran));

  // Safety: the allocator must never hand out an unbacked hole. Keep requesting
  // granule-sized buffers; every returned pointer must be physically backed,
  // and because re-backing the harvested holes would exceed the (at-cap,
  // no-backdoor) budget, allocation must eventually fail cleanly rather than
  // return a hole.
  std::vector<void *> more;
  bool sawCleanFailure = false;
  for (int i = 0; i < 8; ++i) {
    ErrorOr<void *> p = fl.allocate(kGran, mode);
    if (p.isError()) {
      sawCleanFailure = true;
      break;
    }
    EXPECT_NO_ERR(writeAndVerify(*ctx, *p, kGran, kGran));
    more.push_back(*p);
  }
  EXPECT_TRUE(sawCleanFailure)
      << "at-cap allocator must refuse rather than return an unbacked hole";

  // Teardown.
  EXPECT_NO_ERR(fl.free(big));
  for (void *p : more)
    EXPECT_NO_ERR(fl.free(p));
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));
  cache->clear(); // standalone manager must be cleared before destruction
}

// getMemoryInfo() must not double-count harvested (unbacked) holes. After
// harvest moves physical out of interior holes, those holes remain as free
// descs but have no backing; re-mapping them would consume the (at-cap)
// uncommitted budget, so `free` must report only the still-backed free memory.
// The reported value is also cross-checked against what can actually be
// allocated, guarding against both over- and under-reporting.
TEST_F(MemoryManagerVMMPartialHarvestTest,
       GetMemoryInfoExcludesHarvestedHoles) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB; // chunk_percent=100 => one 4 GiB chunk
  const size_t kGran = 256 * kMiB; // == VMM granule (VMM_PAGE_MB=256)
  const size_t kBig = 1 * kGiB;    // 4 granules; > largest contiguous hole

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();

  std::vector<void *> bufs;
  for (int i = 0; i < 16; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kGran, mode));
    bufs.push_back(p);
  }
  // Free every other buffer: 8 scattered 256 MiB holes (2 GiB), none harvested
  // yet, so all are backed and fully allocatable.
  for (size_t i = 0; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));
  EXPECT_EQ(cache->getMemoryInfo().free, 8 * kGran);

  // A request larger than any contiguous hole triggers harvest of 4 granules;
  // the 4 harvested holes stay in the free-list but become unbacked.
  ASSIGN_NO_ERR(void *, big, fl.allocate(kBig, mode));
  EXPECT_NE(big, nullptr);

  // free must now report only the 4 still-backed holes (1 GiB), NOT 2 GiB.
  // Pre-fix this over-reported by harvestedBytes_ (would read ~2 GiB).
  const size_t reported = cache->getMemoryInfo().free;
  EXPECT_EQ(reported, 4 * kGran);

  // Cross-check the report against reality: exactly `reported` worth of
  // granule-sized buffers must allocate, and the next must fail cleanly.
  std::vector<void *> more;
  for (size_t got = 0; got < reported; got += kGran) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kGran, mode));
    more.push_back(p);
  }
  EXPECT_TRUE(fl.allocate(kGran, mode).isError())
      << "getMemoryInfo over-reported free: more allocatable than reported";

  EXPECT_NO_ERR(fl.free(big));
  for (void *p : more)
    EXPECT_NO_ERR(fl.free(p));
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));
  cache->clear();
}

// A request larger than everything that can be reclaimed (free-granule pool +
// harvestable granules + uncommitted budget) is an honest aggregate OOM: the
// allocator must return a clean error, leave the still-in-use buffers intact,
// and never hand out an unbacked hole.
TEST_F(MemoryManagerVMMPartialHarvestTest, AggregateOomReturnsCleanError) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB; // chunk_percent=100 => one 4 GiB chunk
  const size_t kGran = 256 * kMiB;

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  std::vector<void *> bufs;
  for (int i = 0; i < 16; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kGran, mode));
    bufs.push_back(p);
  }
  // Free 8 alternating granules: at most 8 granules (2 GiB) are reclaimable,
  // and the pool/budget are empty (at cap). The 8 odd buffers stay in use.
  for (size_t i = 0; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(writeAndVerify(*ctx, bufs[i], kGran, kGran));

  // Request 9 granules' worth: exceeds the 8 reclaimable granules, so this is a
  // genuine aggregate OOM and must fail cleanly rather than crash or harvest a
  // ninth (live) granule.
  EXPECT_TRUE(fl.allocate(9 * kGran, mode).isError());

  // The in-use buffers must be untouched by the failed attempt.
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(writeAndVerify(*ctx, bufs[i], kGran, kGran));

  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));
  cache->clear();
}

} // namespace
