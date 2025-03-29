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

#include <AsyncRT/DeviceContext/DeviceContext.h>

#include <cstdint>
#include <mutex>

namespace llvm {
class raw_fd_ostream;
} // namespace llvm

namespace M::AsyncRT {

class BufferCache {
public:
  class Chunk;
  struct Desc;

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

  BufferCache(const DeviceContext &ctx, size_t chunkSize, size_t maxCacheSize,
              size_t alignment);
  ~BufferCache();

  void clear();

  // Allocate and free memory by using the `BufferCache`. If the allocation
  // cannot be serviced by the `BufferCache` due to allocation size or cache
  // capacity, then the allocation request gets transparently forwarded to the
  // `DeviceContext`.
  ErrorOr<void *> allocate(size_t size, DeviceContext::AllocMode mode);
  ErrorOrSuccess free(void *buf, bool isHost);

private:
  friend class Chunk;

  void selfCheck();

  void addToFreeList(Desc *desc);
  void removeFromFreeList(Desc *desc);

  bool allocateChunk();
  void *allocateLocked(size_t size);
  bool freeLocked(void *buf);

  // Avoid malloc under lock.
  Desc *popDesc();
  void pushDesc(Desc *desc);

  const DeviceContext &ctx_;
  const size_t chunkSize_;
  const size_t maxCacheSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  std::mutex mu_;
  // List of all chunks allocated from the driver and their total size.
  size_t cacheSize_ = 0;
  Chunk *chunks_ = nullptr;

  // List of unallocated memory regions.
  Desc *freeBuffers_ = nullptr;

  // Free list of descriptors to avoid malloc under lock.
  Desc *unusedDescs_ = nullptr;

  // Map of allocated ptr->Desc*: Since the memory handled by the BufferCache is
  // not accessible by the CPU we cannot store meta-data colocated with the
  // managed memory.
  DescMap allocatedMap_;

  // For logging and debugging.
  llvm::raw_fd_ostream &errOut();
  bool doSelfCheck_ = false;
  bool doLog_ = false;
  size_t freeSize_ = 0;
  size_t freeCount_ = 0;
  size_t chunkCount_ = 0;
  std::string deviceDescription_;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H
