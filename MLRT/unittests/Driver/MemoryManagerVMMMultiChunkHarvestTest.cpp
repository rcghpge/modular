//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Tests for VMM partial-chunk harvest across MULTIPLE chunks. The target's env
// sets chunk_percent=25, so the 4 GiB pool is carved into four 1 GiB chunks
// (four 256 MiB granules each). The load-bearing test frees granules so that
// every chunk is left partially used (none wholly free) and no single chunk has
// enough contiguous free space for the request -- the allocation can only
// succeed by harvesting whole free granules from across several partially-used
// chunks and remapping them into one fresh contiguous VA range. This exercises
// the cross-chunk path, the per-chunk reclaimable-slot cache, and the
// back-to-front harvest order. memory-manager-only mode removes the driver
// backdoor. Single-GPU (owner-device access only).
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using namespace M::Driver;
using M::Error;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

constexpr size_t kGiB = size_t(1024) * 1024 * 1024;
constexpr size_t kMiB = size_t(1024) * 1024;

class MemoryManagerVMMMultiChunkHarvestTest : public ::testing::Test {
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
  // chunk-percent (25), memory-manager-only and self-check come from the env.
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
  const std::vector<uint8_t> pattern(64, 0xCD);
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

// A 1 GiB request (4 granules) succeeds even though every one of the four
// partially-used chunks has only two free granules: harvest gathers free
// granules from across chunks and assembles them into one contiguous VA range.
TEST_F(MemoryManagerVMMMultiChunkHarvestTest,
       HarvestAssemblesAcrossMultiplePartialChunks) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 4 * kGiB; // chunk_percent=25 => four 1 GiB chunks
  const size_t kGran = 256 * kMiB; // == VMM granule (VMM_PAGE_MB=256)
  const size_t kBig = 1 * kGiB;    // 4 granules; no single chunk has 4 free

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // 16 granule-sized buffers fill four 1 GiB chunks (4 granules each), created
  // on demand: buffer i lives in chunk i/4, slot i%4.
  std::vector<void *> bufs;
  for (int i = 0; i < 16; ++i) {
    ASSIGN_NO_ERR(void *, p, fl.allocate(kGran, mode));
    bufs.push_back(p);
  }

  // Free slots {0, 2} of every chunk (even global indices). Each chunk is now
  // 2 free / 2 in use: no chunk is wholly free, and no chunk has more than two
  // free granules, so a 4-granule request cannot be met from any single chunk.
  for (size_t i = 0; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(fl.free(bufs[i]));

  // Only cross-chunk harvest can satisfy this; verify it is fully backed.
  ASSIGN_NO_ERR(void *, big, fl.allocate(kBig, mode));
  EXPECT_NE(big, nullptr);
  EXPECT_NO_ERR(writeAndVerify(*ctx, big, kBig, kGran));

  // Harvest must not have disturbed the still-in-use buffers (odd indices).
  for (size_t i = 1; i < bufs.size(); i += 2)
    EXPECT_NO_ERR(writeAndVerify(*ctx, bufs[i], kGran, kGran));

  // 4 of the 8 free granules were harvested into `big`; the remaining 4 stay
  // backed and allocatable, and getMemoryInfo must report exactly those.
  EXPECT_EQ(cache->getMemoryInfo().free, 4 * kGran);

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

// Randomized churn: hundreds of mixed allocate/free operations of varied
// granule-aligned and sub-granule sizes. With a budget larger than typical live
// usage there is headroom, so this naturally drives the full machinery --
// repeated harvest, re-back SUCCESS (from the free-granule pool and via
// vmmCreate), re-back failure near cap, coalescing of descs spanning
// backed+unbacked granules, and wholly-free-chunk retirement -- across states
// too varied to construct by hand. Correctness is enforced two ways:
//   * Every live allocation carries a unique 8-byte id written at the start of
//     each granule it spans. Handing out an unbacked hole would fault on that
//     write; any overlap/aliasing is caught by reading the id back.
//   * The process-wide self-check (SELF_CHECK=True) validates the
//     harvestedBytes_ / granuleBacked_ invariants after every allocate.
TEST_F(MemoryManagerVMMMultiChunkHarvestTest, RandomizedChurnStaysConsistent) {
  const Device::AllocMode mode = Device::AllocMode::SYNC;
  const size_t kBudget = 6 * kGiB;
  const size_t kGran = 256 * kMiB; // env VMM_PAGE_MB=256

  ASSIGN_NO_ERR(DeviceRef, device, dev());
  std::unique_ptr<MemoryManager> cache = makeCache(*device, kBudget);
  ASSERT_NE(cache, nullptr);
  MemoryManagerFreeList &fl = *cache->freeList();
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  // Write `id`'s 8 bytes at the start of every granule of [p, p+sz); reading a
  // value other than `id` later means overlap, and an unbacked granule faults.
  auto idBytes = [](uint64_t id) {
    std::vector<uint8_t> b(8);
    for (int i = 0; i < 8; ++i)
      b[i] = static_cast<uint8_t>(id >> (8 * i));
    return b;
  };
  auto writeId = [&](void *p, size_t sz, uint64_t id) -> ErrorOrSuccess {
    const std::vector<uint8_t> pat = idBytes(id);
    for (size_t off = 0; off < sz; off += kGran)
      if (ErrorOrSuccess e = ctx->enqueueCopyToDevice(
              static_cast<char *>(p) + off, pat.data(), pat.size());
          e.isError())
        return e.takeError();
    return ctx->synchronize();
  };
  auto verifyId = [&](void *p, size_t sz, uint64_t id) -> ErrorOrSuccess {
    const std::vector<uint8_t> want = idBytes(id);
    for (size_t off = 0; off < sz; off += kGran) {
      std::vector<uint8_t> got(8, 0);
      if (ErrorOrSuccess e = ctx->enqueueCopyFromDevice(
              got.data(), static_cast<char *>(p) + off, got.size());
          e.isError())
        return e.takeError();
      if (ErrorOrSuccess s = ctx->synchronize(); s.isError())
        return s.takeError();
      if (got != want)
        return Error(
            "sentinel mismatch: overlap or unbacked memory handed out");
    }
    return M::success();
  };

  struct Live {
    void *ptr;
    size_t size;
    uint64_t id;
  };
  std::vector<Live> live;
  std::mt19937 rng(0xC0FFEEu);
  const size_t sizes[] = {64 * kMiB,  128 * kMiB, 192 * kMiB,
                          256 * kMiB, 512 * kMiB, 768 * kMiB};
  uint64_t nextId = 1;

  for (int iter = 0; iter < 400; ++iter) {
    const bool doAlloc = live.empty() || (rng() % 100) < 60;
    if (doAlloc) {
      const size_t sz = sizes[rng() % (sizeof(sizes) / sizeof(sizes[0]))];
      ErrorOr<void *> p = fl.allocate(sz, mode);
      if (p.isError())
        continue; // OOM under pressure is expected; keep churning
      const uint64_t id = nextId++;
      EXPECT_NO_ERR(writeId(*p, sz, id));
      live.push_back({*p, sz, id});
    } else {
      const size_t idx = rng() % live.size();
      EXPECT_NO_ERR(verifyId(live[idx].ptr, live[idx].size, live[idx].id));
      EXPECT_NO_ERR(fl.free(live[idx].ptr));
      live[idx] = live.back();
      live.pop_back();
    }
  }

  // Every survivor must still hold its own id (no aliasing/corruption).
  for (const Live &l : live)
    EXPECT_NO_ERR(verifyId(l.ptr, l.size, l.id));
  for (const Live &l : live)
    EXPECT_NO_ERR(fl.free(l.ptr));
  cache->clear();
}

} // namespace
