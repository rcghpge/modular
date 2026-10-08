//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "_core/Driver/ChunkedStagingRegion.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/AsyncValueRef.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Support/Chain.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/ErrorOr.h"
#include "Support/Log.h"
#include "nanobind/nanobind.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <unistd.h>
#include <utility>
#include <vector>

using namespace M;
using namespace Driver;
namespace nb = nanobind;

namespace {

using Slice = std::pair<char *, size_t>;

#if defined(__linux__)
constexpr int kPrSetVma = 0x53564d41;
constexpr int kPrSetVmaAnonName = 0;
#endif

// volatile so the compiler cannot elide the store that faults the page in.
void touchSlice(char *p, size_t len, size_t page) {
  if (len == 0)
    return;
  for (size_t off = 0; off < len; off += page)
    *reinterpret_cast<volatile char *>(p + off) = 0;
  *reinterpret_cast<volatile char *>(p + len - 1) = 0;
}

// Distinct VMA names stop the kernel merging the chunks, so touch workers
// fault them without contending on one VMA lock.
char *mapNamedContiguous(size_t total, size_t chunk, unsigned numChunks,
                         size_t page, size_t *mapLenOut) {
  size_t mapLen = ((total + page - 1) / page) * page;
  void *res = mmap(nullptr, mapLen, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (res == MAP_FAILED)
    throw std::runtime_error(
        std::string("ChunkedStagingRegion: mmap failed: ") +
        std::strerror(errno));
  char *span = static_cast<char *>(res);
#if defined(__linux__)
  constexpr size_t kVmaNameMax = 32;
  for (unsigned i = 0; i < numChunks; ++i) {
    size_t off = static_cast<size_t>(i) * chunk;
    size_t nameLen = std::min(chunk, mapLen - off);
    char nm[kVmaNameMax];
    std::snprintf(nm, sizeof(nm), "maxpinned%u", i);
    if (prctl(kPrSetVma, kPrSetVmaAnonName, span + off, nameLen, nm) != 0) {
      // All-or-nothing, and only a perf optimization.
      MLOG_WARN("ChunkedStagingRegion: naming anonymous VMA failed ({}); "
                "continuing without per-chunk VMA names",
                std::strerror(errno));
      break;
    }
  }
#else
  (void)chunk;
  (void)numChunks;
#endif
  *mapLenOut = mapLen;
  return span;
}

// Dropping a wrap only queues its unregister, and handlers are polled rather
// than pushed, so each step needs its own drain. Blocking is fine in teardown.
void releaseRegion(const DeviceContextRef &dctx,
                   std::vector<M::Driver::DeviceBufferRef> &wraps, char *base,
                   size_t mapLen) {
  // Reached from a destructor, and unmapping anyway is the lesser leak.
  auto drain = [&] {
    if (ErrorOrSuccess e = dctx->synchronize(); e.isError())
      MLOG_WARN("ChunkedStagingRegion release: synchronize failed ({})",
                e.getError());
    if (ErrorOrSuccess e = dctx->pollPendingEvents(); e.isError())
      MLOG_WARN("ChunkedStagingRegion release: polling events failed ({})",
                e.getError());
  };
  drain(); // in-flight DMA must land before the wraps go
  wraps.clear();
  drain(); // queued unregisters must run before the pages go
  munmap(base, mapLen);
}

} // namespace

Python::ChunkedStagingRegion::ChunkedStagingRegion(size_t byteSize,
                                                   size_t rowBytes,
                                                   PyDeviceTy device,
                                                   unsigned threads,
                                                   size_t chunkBytes)
    : device_(std::move(device)), byteSize_(byteSize), rowBytes_(rowBytes) {
  if (device_->isHost())
    throw std::invalid_argument(
        "ChunkedStagingRegion requires a non-host device "
        "(GPU/Accelerator).");
  if (byteSize == 0)
    throw std::invalid_argument("ChunkedStagingRegion: byte_size must be "
                                "> 0");
  if (rowBytes == 0)
    throw std::invalid_argument("ChunkedStagingRegion: row_bytes must be "
                                "> 0");
  if (byteSize % rowBytes != 0)
    throw std::invalid_argument(
        "ChunkedStagingRegion: byte_size " + std::to_string(byteSize) +
        " is not a multiple of row_bytes " + std::to_string(rowBytes));
  if (chunkBytes == 0)
    throw std::invalid_argument("ChunkedStagingRegion: chunk_bytes must be "
                                "> 0");

#if !defined(__linux__)
  MLOG_WARN("ChunkedStagingRegion is tuned for Linux; on this platform the "
            "per-chunk VMA-naming optimization is unavailable");
#endif

  DeviceContextRef dctx = device_->ctx;
  M::AsyncRT::CPUDeviceRef cpuDeviceRef =
      Python::getCpuDevice(-1)->getAsyncRTCPUDevice();

  size_t page = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
  // A boundary inside a row breaks transfers; a boundary mid-page makes two
  // chunks share a page and the second wrap is rejected as overlapping.
  size_t alignment = std::lcm(page, rowBytes);
  chunkBytes_ =
      std::max<size_t>(alignment, (chunkBytes / alignment) * alignment);
  unsigned numChunks =
      static_cast<unsigned>((byteSize + chunkBytes_ - 1) / chunkBytes_);

  base_ = mapNamedContiguous(byteSize, chunkBytes_, numChunks, page, &mapLen_);

  std::vector<Slice> slices(numChunks);
  for (unsigned i = 0; i < numChunks; ++i) {
    size_t off = static_cast<size_t>(i) * chunkBytes_;
    slices[i] = {base_ + off, std::min(chunkBytes_, byteSize - off)};
  }

  using AsyncChain = M::AsyncRT::AsyncValueRef<M::AsyncRT::Chain>;

  std::vector<AsyncChain> touchDone;
  std::vector<AsyncChain> wrapDone;
  touchDone.reserve(numChunks);
  wrapDone.reserve(numChunks);
  for (unsigned i = 0; i < numChunks; ++i) {
    touchDone.push_back(AsyncChain::allocate(*cpuDeviceRef));
    wrapDone.push_back(AsyncChain::allocate(*cpuDeviceRef));
  }

  std::vector<M::Driver::DeviceBufferRef> wraps(numChunks);
  std::atomic<bool> anyFailed{false};
  std::string firstErrMsg;
  std::mutex errMu;

  auto wrapChunk = [&](unsigned i) {
    // Metal rejects a non-page-multiple length, which the last chunk usually
    // has. mmap already page-rounded the mapping, so this stays inside it.
    size_t wrapLen = ((slices[i].second + page - 1) / page) * page;
    ErrorOr<M::Driver::DeviceBufferRef> wrapOr =
        dctx->wrapHostMemory(slices[i].first, wrapLen);
    if (wrapOr.isError()) {
      std::lock_guard<std::mutex> lk(errMu);
      if (!anyFailed.exchange(true, std::memory_order_relaxed))
        firstErrMsg = wrapOr.getError();
      return;
    }
    wraps[i] = wrapOr.takeValue();
  };

  constexpr int kWrapWorker = 0;
  const size_t numWorkers = cpuDeviceRef->getWorkQueue()->getParallelismLevel();
  // Extra touch lanes only contend on memory bandwidth, so leave kWrapWorker.
  const size_t touchLanes =
      numWorkers > 1 ? std::min<size_t>(std::max(1u, threads), numWorkers - 1)
                     : 0;

  auto enqueueWrap = [&](unsigned i) {
    M::AsyncRT::addTask(
        *cpuDeviceRef,
        [&, i]() {
          wrapChunk(i);
          wrapDone[i].copy().emplace();
        },
        kWrapWorker);
  };

  {
    nb::gil_scoped_release nogil;

    // Gating on the previous wrap keeps wrapping serial while touch runs free.
    touchDone[0].andThenSync([&]() { enqueueWrap(0); });
    for (unsigned i = 1; i < numChunks; ++i) {
      M::AsyncRT::andThenSync(
          std::make_tuple(touchDone[i].copy(), wrapDone[i - 1].copy()),
          [&, i](AsyncChain, AsyncChain) { enqueueWrap(i); });
    }

    M::AsyncRT::parallelForEachNChainWithTaskIds(
        *cpuDeviceRef, static_cast<size_t>(numChunks),
        [touchLanes](size_t i) -> int {
          if (touchLanes == 0)
            return kDefaultTaskId;
          return static_cast<int>(1 + (i % touchLanes));
        },
        [&](size_t i) {
          touchSlice(slices[i].first, slices[i].second, page);
          touchDone[i].copy().emplace();
        });

    M::AsyncRT::await(wrapDone[numChunks - 1]);
  }

  if (anyFailed.load(std::memory_order_acquire)) {
    {
      nb::gil_scoped_release nogil;
      releaseRegion(dctx, wraps, base_, mapLen_);
    }
    base_ = nullptr;
    throw std::runtime_error("ChunkedStagingRegion: host memory wrap failed: " +
                             firstErrMsg);
  }

  wraps_ = std::move(wraps);
}

Python::ChunkedStagingRegion::ChunkedStagingRegion(
    ChunkedStagingRegion &&other) noexcept
    : device_(std::move(other.device_)), wraps_(std::move(other.wraps_)),
      base_(other.base_), mapLen_(other.mapLen_), byteSize_(other.byteSize_),
      rowBytes_(other.rowBytes_), chunkBytes_(other.chunkBytes_) {
  other.base_ = nullptr;
}

Python::ChunkedStagingRegion::~ChunkedStagingRegion() {
  if (base_ == nullptr)
    return;
  nb::gil_scoped_release nogil;
  releaseRegion(device_->ctx, wraps_, base_, mapLen_);
}
