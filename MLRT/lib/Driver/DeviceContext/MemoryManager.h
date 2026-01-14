//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the portable memory manager used by device contexts.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H
#define MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <mutex>

namespace llvm {
class raw_fd_ostream;
} // namespace llvm

namespace M::Driver {

class MemoryManagerFreeList;

class MemoryManager {
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

  ~MemoryManager();

  size_t allocationAlignment() const;

  bool isHost() const;

  void clear();

  MemoryManagerFreeList *freeList() const;

  // Get memory information about the memory manager
  Device::MemInfo getMemoryInfo() const;

  // Check if the memory manager is enabled (has non-zero capacity)
  bool isEnabled() const;

  // Factory methods configure the memory manager based on defaults or the
  // environment.
  // `memInfo` contains the total and free memory of the device.
  // Setting memInfo.free to 0 will disable the memory manager.
  // `memoryLimit` (bytes) is the maximum memory limit for the memory manager.
  // If -1, then the memory manager will use the default memory limit.
  // Note that MEMORY_MANAGER_SIZE_LIMIT environment variable overrides this.
  static MemoryManager *onDevice(const Device &device, Device::MemInfo memInfo,
                                 size_t alignment, int64_t memoryLimit = -1);
  static MemoryManager *onHost(const Device &device, size_t alignment);

private:
  friend class Chunk;
  friend class MemoryManagerFreeList;

  MemoryManager(const Device &device, size_t maxSize, size_t chunkSize,
                size_t alignment, bool isHost, bool useOnlyMemoryManagerMode);

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
  // Change this process' state to use the full memory manager for future chunk
  // allocations.
  static void enableUseMaxSize();
  // Use a std::atomic to keep track of the state whether or not we use the full
  // memory manager for future chunk allocations. Since this is a boolean value
  // that can only change from 0 (false) to non-zero (true) using a mutex is not
  // necessary.
  static std::atomic<int32_t> useMaxSize_;

  // Attempt to grow the chunk size to newChunkSize. The eventual chunk size
  // will be capped so that future chunk allocations do not overflow
  // maxCacheSize_.
  void growChunkSizeTo(size_t newChunkSize);

  const Device &device_;

  // This memory manager is managing "pinned" host memory.
  const bool isHost_;

  // Whether to only allocate memory from the memory manager.
  // This setting is useful for bazel to enforce the cap on gpu memory used by
  // a particular test. This enables better binpacking of tests on a CI runner.
  const bool useOnlyMemoryManagerMode_;

  const size_t maxCacheSize_;
  size_t chunkSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  mutable std::mutex mu_;
  // List of all chunks allocated from the driver and their total size.
  size_t cacheSize_ = 0;
  Chunk *chunks_ = nullptr;

  // List of unallocated memory regions.
  std::unique_ptr<MemoryManagerFreeList> freeList_;

  // Free list of descriptors to avoid malloc under lock.
  Desc *unusedDescs_ = nullptr;

  // Map of allocated ptr->Desc*: Since the memory handled by the MemoryManager
  // is not accessible by the CPU we cannot store meta-data colocated with the
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

class MemoryManagerFreeList {
public:
  MemoryManagerFreeList(const DeviceContext *context,
                        MemoryManager &memoryManager,
                        MemoryManagerFreeList *parent);
  ~MemoryManagerFreeList();

  // Release blocks owned by this free list to the parent if available.
  void flushToParent();

  // Allocate and free memory by using the `MemoryManager`. If the allocation
  // cannot be serviced by the `MemoryManager` due to allocation size or cache
  // capacity, then the allocation request gets transparently forwarded to the
  // associated `DeviceContext`.
  ErrorOr<void *> allocate(size_t size, Device::AllocMode mode);
  ErrorOrSuccess free(void *buf);

  bool empty();

  // Enable out-of-bounds access checks
  void enableRedzones();

private:
  friend class MemoryManager;
  friend class MemoryManager::Chunk;

  // Define the number of "double-buffered freelists" used to implement the
  // MemoryManagerFreeList. The oldest list will be flushed to the parent once
  // the pending event handler fires.
  static constexpr int kNumFreeBufferLists = 2;
  static_assert(llvm::isPowerOf2_64(kNumFreeBufferLists),
                "Must be a non-zero power of two.");

  void selfCheck() const;
  bool contains(MemoryManager::Desc *desc) const;

  const DeviceContext *context() const;

  MemoryManager::Desc *allocateDesc(size_t size);
  void freeDesc(MemoryManager::Desc *desc);

  void addToFreeList(MemoryManager::Desc *desc);
  void removeFromFreeList(MemoryManager::Desc *desc);

  void flushFreeBuffersToParent(int id);

  struct RedzoneInfo {
    std::string stackTraceOnAlloc_;
    size_t size_;
  };

  ErrorOr<void *> redzoneFillAndWrap(void *unwrappedPtr, size_t size);
  ErrorOrSuccess validateRedzones(void *wrappedPtr, const RedzoneInfo &size);
  void printRedzone(const uint8_t *dataPtr, const uint8_t *displayPtr);

  const DeviceContext *ctx_;

  MemoryManager &memoryManager_;
  MemoryManagerFreeList *parent_;

  MemoryManager::Desc *freeBuffers_[kNumFreeBufferLists];
  int primaryFreeBuffers_ = 0;
  bool handlerPending_ = false;

  bool redzoneEnable_ = false;
  size_t redzoneSize_ = 0;
  uint8_t redzonePattern_ = 0;
  DenseMap<void *, RedzoneInfo> redzoneDeviceAllocSizeMap_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H
