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

  BufferCache(const Device &device_, Device::MemInfo memInfo, size_t alignment);
  ~BufferCache();

  size_t allocationAlignment() const;

  void clear();

  BufferCacheFreeList *freeList() const;

private:
  friend class Chunk;
  friend class BufferCacheFreeList;

  const Device &device() const;

  void selfCheck() const;

  bool allocateChunk();
  bool freeLocked(void *buf);

  // Avoid malloc under lock.
  Desc *popDesc();
  void pushDesc(Desc *desc);

  const Device &device_;
  const size_t maxCacheSize_;
  const size_t chunkSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  std::mutex mu_;
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
  ErrorOrSuccess free(void *buf, bool isHost);

  bool empty();

private:
  friend class BufferCache;
  friend class BufferCache::Chunk;

  void selfCheck() const;
  bool contains(BufferCache::Desc *desc) const;

  const DeviceContext *context() const;

  BufferCache::Desc *allocateDesc(size_t size);
  void freeDesc(BufferCache::Desc *desc);

  void addToFreeList(BufferCache::Desc *desc);
  void removeFromFreeList(BufferCache::Desc *desc);

  const DeviceContext *ctx_;

  BufferCache &bufferCache_;
  BufferCacheFreeList *parent_;

  BufferCache::Desc *freeBuffers_ = nullptr;

  // For logging and debugging.
  size_t ffreeSize_ = 0;
  size_t ffreeCount_ = 0;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H
