//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the portable buffer cache used by device contexts.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H
#define ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <mutex>

namespace llvm {
class raw_fd_ostream;
} // namespace llvm

namespace M::AsyncRT {

class BufferCacheFreeList;

class BufferCache {
public:
  class Chunk;
  class Desc;

  class DescMap {
  public:
    DescMap(int mapSizeLog2);
    ~DescMap();

    void add(Desc *desc);
    Desc *remove(void *ptr);

  private:
    static uint32_t hash(void *ptr);

    size_t bucketCount_;
    size_t bucketMask_;
    Desc **buckets_;
  };

  ~BufferCache();

  size_t allocationAlignment() const;

  bool isHost() const;

  void clear();

  BufferCacheFreeList *freeList() const;

  // Get memory information about the buffer cache
  Device::MemInfo getMemoryInfo() const;

  // Check if the buffer cache is enabled (has non-zero capacity)
  bool isEnabled() const;

  // Factory methods configure the buffer cache based on defaults or the
  // environment.
  static BufferCache *onDevice(const Device &device, Device::MemInfo memInfo,
                               size_t alignment);
  static BufferCache *onHost(const Device &device, size_t alignment);

private:
  friend class Chunk;
  friend class BufferCacheFreeList;

  BufferCache(const Device &device, size_t maxSize, size_t chunkSize,
              size_t alignment, bool isHost);

  const Device &device() const;

  void selfCheck() const;

  bool allocateChunk();
  bool freeLocked(void *buf);

  // Avoid malloc under lock.
  Desc *popDesc();
  void pushDesc(Desc *desc);

  // Check if at least one of the devices has determined that chunk sizes should
  // grow to the maximum cache size based on the heuristics used during
  // allocation.
  static bool shouldUseMaxSize();
  // Change this process' state to use the full buffer cache for future chunk
  // allocations.
  static void enableUseMaxSize();
  // Use a std::atomic to keep track of the state whether or not we use the full
  // buffer cache for future chunk allocations. Since this is a boolean value
  // that can only change from 0 (false) to non-zero (true) using a mutex is not
  // necessary.
  static std::atomic<int32_t> useMaxSize_;

  // Attempt to grow the chunk size to newChunkSize. The eventual chunk size
  // will be capped so that future chunk allocations do not overflow
  // maxCacheSize_.
  void growChunkSizeTo(size_t newChunkSize);

  const Device &device_;

  // This buffer cache is managing "pinned" host memory.
  const bool isHost_;

  const size_t maxCacheSize_;
  size_t chunkSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  mutable std::mutex mu_;
  // List of all chunks allocated from the driver and their total size.
  size_t cacheSize_ = 0;
  Chunk *chunks_ = nullptr;

  // List of unallocated memory regions.
  std::unique_ptr<BufferCacheFreeList> freeList_;

  // Free list of descriptors to avoid malloc under lock.
  Desc *unusedDescs_ = nullptr;

  // Map of allocated ptr->Desc*: Since the memory handled by the BufferCache is
  // not accessible by the CPU we cannot store meta-data colocated with the
  // managed memory.
  DescMap allocatedMap_;

  // For logging and debugging.
  std::string &getDeviceDescription() const;
  void logErr(llvm::Twine msg) const;
  void logErr(llvm::Twine msg, size_t info, int64_t size,
              bool printInfoSize = true) const;

  bool doSelfCheck_ = false;
  bool doLog_ = false;
  size_t freeSize_ = 0;
  size_t freeCount_ = 0;
  size_t chunkCount_ = 0;
  mutable std::string deviceDescription_;
};

class BufferCacheFreeList {
public:
  BufferCacheFreeList(const DeviceContext *context, BufferCache &bufferCache,
                      BufferCacheFreeList *parent);
  ~BufferCacheFreeList();

  // Release blocks owned by this free list to the parent if available.
  void flushToParent();

  // Allocate and free memory by using the `BufferCache`. If the allocation
  // cannot be serviced by the `BufferCache` due to allocation size or cache
  // capacity, then the allocation request gets transparently forwarded to the
  // associated `DeviceContext`.
  ErrorOr<void *> allocate(size_t size, Device::AllocMode mode);
  ErrorOrSuccess free(void *buf);

  bool empty();

private:
  friend class BufferCache;
  friend class BufferCache::Chunk;

  // Define the number of "double-buffered freelists" used to implement the
  // BufferCacheFreeList. The oldest list will be flushed to the parent once the
  // pending event handler fires.
  static constexpr int kNumFreeBufferLists = 2;
  static_assert(llvm::isPowerOf2_64(kNumFreeBufferLists),
                "Must be a non-zero power of two.");

  void selfCheck() const;
  bool contains(BufferCache::Desc *desc) const;

  const DeviceContext *context() const;

  BufferCache::Desc *allocateDesc(size_t size);
  void freeDesc(BufferCache::Desc *desc);

  void addToFreeList(BufferCache::Desc *desc);
  void removeFromFreeList(BufferCache::Desc *desc);

  void flushFreeBuffersToParent(int id);

  const DeviceContext *ctx_;

  BufferCache &bufferCache_;
  BufferCacheFreeList *parent_;

  BufferCache::Desc *freeBuffers_[kNumFreeBufferLists];
  int primaryFreeBuffers_ = 0;
  bool handlerPending_ = false;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H
