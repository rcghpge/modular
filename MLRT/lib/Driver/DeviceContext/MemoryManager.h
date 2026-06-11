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
#include <vector>

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

    // Rewrite `allocatedBy_` on every live descriptor whose current value is
    // `from`, replacing it with `to`. Used when a `MemoryManagerFreeList` is
    // about to be destroyed to redirect descriptors it allocated but does not
    // yet hold (still in use elsewhere) to a longer-lived parent freelist, so
    // a later `free()` does not dispatch through a dangling pointer.
    void redirectAllocatedBy(MemoryManagerFreeList *from,
                             MemoryManagerFreeList *to);

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
  static MemoryManager *onDevice(Device &device, Device::MemInfo memInfo,
                                 size_t alignment, int64_t memoryLimit = -1);
  static MemoryManager *onHost(Device &device, size_t alignment);

private:
  friend class Chunk;
  friend class MemoryManagerFreeList;

  MemoryManager(Device &device, size_t maxSize, size_t chunkSize,
                size_t alignment, bool isHost, bool useOnlyMemoryManagerMode,
                bool vmmEnabled = false, size_t vmmPageSize = 0);

  Device &device() const;

  void selfCheck() const;
  Error allocationMissError(const MemoryManagerFreeList *freeList,
                            size_t wrappedSize) const;
  bool allocateChunk();
  bool freeLocked(void *buf);

  // ---- VMM (defragmenting allocator) ----
  //
  // Acquire physical+virtual backing for a chunk of `allocSize` bytes. When
  // `vmmEnabled_`, reserves VA + creates/maps/sets-access granules (reusing the
  // free-granule pool first) and fills `handles`; otherwise calls chunkAlloc.
  // Returns false on failure (leaving nothing allocated), like a failed
  // chunkAlloc. Caller holds `mu_`.
  bool acquireChunkBacking(size_t allocSize, void **base,
                           std::vector<uint64_t> *handles, size_t *pageSize);
  // Tear down a VMM chunk's backing: unmap each granule, return handles to the
  // free-granule pool (or release if `releaseToDriver`), and free the VA.
  // Caller holds `mu_`.
  void releaseChunkBacking(Chunk *chunk, bool releaseToDriver);
  // Defragment to satisfy a request larger than any contiguous free block:
  // reserve fresh contiguous VA, recycle granules from the free-granule pool
  // and from wholly-free VMM chunks, map them into the new VA, and materialize
  // the result as a new in-use chunk owned by `freeList`. Returns the buffer
  // pointer, or an Error if not enough physical memory is reclaimable. Caller
  // must NOT hold `mu_` (this acquires it).
  ErrorOr<void *> defragForRequest(size_t size,
                                   MemoryManagerFreeList *freeList);

  // Registry of per-stream (child) free-lists, so a VMM allocation miss can
  // flush sibling streams' completed frees to the root free-list (a freed block
  // sits in its per-stream buffer until that stream polls events).
  void registerChildFreeList(MemoryManagerFreeList *fl);
  void unregisterChildFreeList(MemoryManagerFreeList *fl);
  // Poll pending events on every registered child context except `except`,
  // firing completed free->root flush handlers. Caller must NOT hold `mu_`.
  void flushSiblingFreeLists(const DeviceContext *except);

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

  Device &device_;

  // This memory manager is managing "pinned" host memory.
  const bool isHost_;

  // Whether to only allocate memory from the memory manager.
  // This setting is useful for bazel to enforce the cap on gpu memory used by
  // a particular test. This enables better binpacking of tests on a CI runner.
  const bool useOnlyMemoryManagerMode_;

  // VMM (CUDA Virtual Memory Management) defragmenting allocator. When enabled,
  // chunks are backed by remappable physical granules so a large contiguous
  // allocation can be assembled from physically-scattered free memory.
  const bool vmmEnabled_;
  // Physical granule size (bytes); a multiple of the device VMM granularity.
  // All VMM reservations/mappings are rounded up to this.
  const size_t pageSize_;
  // Pool of physical granule handles that are currently unmapped and available
  // for reuse (fed by retiring wholly-free VMM chunks).
  std::vector<uint64_t> freeGranulePool_;

  // Registered per-stream (child) free-lists and a lock guarding the list.
  // Used by flushSiblingFreeLists() on an allocation miss.
  std::mutex childFreeListsMu_;
  std::vector<MemoryManagerFreeList *> childFreeLists_;

  const size_t maxCacheSize_;
  size_t chunkSize_;

  const size_t alignmentOffset_;
  const size_t alignmentMask_;

  mutable std::mutex mu_;
  // List of all chunks allocated from the driver and their total size.
  size_t cacheSize_ = 0;
  std::unique_ptr<Chunk> chunks_ = nullptr;

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
  std::string lastChunkAllocFailureDetail_;
  mutable std::string deviceDescription_;
};

class MemoryManagerFreeList {
public:
  MemoryManagerFreeList(DeviceContext *context, MemoryManager &memoryManager,
                        MemoryManagerFreeList *parent,
                        bool disableFlush = false, std::string label = "");
  ~MemoryManagerFreeList();

  // Allocate and free memory by using the `MemoryManager`. If the allocation
  // cannot be serviced by the `MemoryManager` due to allocation size or cache
  // capacity, then the allocation request gets transparently forwarded to the
  // associated `DeviceContext`.
  ErrorOr<void *> allocate(size_t size, Device::AllocMode mode);
  ErrorOrSuccess free(void *buf);

  bool empty();

  // Enable out-of-bounds access checks
  void enableRedzones();

  void logLargestBlocksInfo(size_t wrappedSize, std::string missReason) const;

  // Dump a detailed snapshot of this free list (and its parents) plus the
  // per-chunk allocation layout to stderr. Unlike logLargestBlocksInfo, this
  // is unconditional: it does not depend on the memory_manager_log config
  // flag. Intended for OOM diagnostic: call only when an allocation is about
  // to fail (driver memAlloc errored out, or a no-fallback path is taking an
  // Error return). Acquires the memory manager mutex; caller must not hold it.
  void dumpFragmentationDiagnostic(size_t requestedSize,
                                   llvm::Twine reason) const;

private:
  friend class MemoryManager;
  friend class MemoryManager::Chunk;

  // Define the number of "double-buffered freelists" used to implement the
  // MemoryManagerFreeList. The oldest list will be flushed to the parent once
  // the pending event handler fires.
  static constexpr int kNumFreeBufferLists = 2;
  static_assert(llvm::isPowerOf2_64(kNumFreeBufferLists),
                "Must be a non-zero power of two.");

  // Release cached free buffers to the parent freelist and reattribute any
  // still-in-use descriptors allocated through this freelist to the parent,
  // so a later `free()` does not dispatch through `desc->allocatedBy_` after
  // this freelist is destroyed. Called only from the destructor.
  void flushToParent();

  void selfCheck() const;
  bool contains(MemoryManager::Desc *desc) const;

  DeviceContext *context();
  const DeviceContext *context() const;

  MemoryManager::Desc *allocateDesc(size_t size);
  void freeDesc(MemoryManager::Desc *desc);

  // Free a descriptor allocated by the current free list.
  ErrorOrSuccess freeAllocatedDesc(MemoryManager::Desc *desc, void *ptr);

  void addToFreeList(MemoryManager::Desc *desc);
  void removeFromFreeList(MemoryManager::Desc *desc);

  void flushFreeBuffersToParent(int id);

  // Log helpers that check doLog_ and prepend the free list label.
  void logMsg(llvm::Twine msg) const;
  void logMsg(llvm::Twine msg, size_t info, int64_t size) const;

  struct RedzoneInfo {
    std::string stackTraceOnAlloc_;
    size_t size_;
  };

  ErrorOr<void *> redzoneFillAndWrap(void *unwrappedPtr, size_t size);
  ErrorOrSuccess validateRedzones(void *wrappedPtr, const RedzoneInfo &size);
  void printRedzone(const uint8_t *dataPtr, const uint8_t *displayPtr);

  DeviceContext *ctx_;

  MemoryManager &memoryManager_;
  MemoryManagerFreeList *parent_;

  MemoryManager::Desc *freeBuffers_[kNumFreeBufferLists];
  int primaryFreeBuffers_ = 0;
  bool handlerPending_ = false;

  bool disableFlush_ = false;

  bool redzoneEnable_ = false;
  size_t redzoneSize_ = 0;
  uint8_t redzonePattern_ = 0;
  DenseMap<void *, RedzoneInfo> redzoneDeviceAllocSizeMap_;

  std::string label_;
};

} // namespace M::Driver

#endif // MLRT_LIB_DRIVER_DEVICECONTEXT_MEMORYMANAGER_H
