//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the portable memory manager used by device contexts.
//
//===----------------------------------------------------------------------===//

#include "MemoryManager.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "Support/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>

// Helpers for logging.
#define TWINE(val) (llvm::Twine(val))

#define TWINE_PTR(ptr)                                                         \
  (llvm::Twine("0x") + llvm::Twine::utohexstr(reinterpret_cast<uintptr_t>(ptr)))

static constexpr int KB = 1024;                  // NOLINT
static constexpr size_t GB = 1024 * 1024 * 1024; // NOLINT

namespace M::AsyncRT {

class MemoryManager::Chunk {
public:
  Chunk(MemoryManager &cache, void *base, size_t size);
  ~Chunk() = default;

  Desc *splitFreeDesc(Desc *freeDesc, size_t size);

  // merged block descriptor to the freelist `owner`.
  void mergeFreeBlockDesc(MemoryManagerFreeList *owner, Desc *desc);

private:
  friend class MemoryManager;
  friend class MemoryManagerFreeList;

  void selfCheck() const;

  MemoryManager &cache_;
  void *base_ = nullptr;
  size_t size_ = 0;
  Chunk *next_ = nullptr;
  Desc *descs_ = nullptr; // Ordered list of all memory regions in this chunk.
};

// Describes a memory region managed by the memory manager. These descriptors
// are maintained out of line of the allocated blocks so they can be accessed by
// the host, while the memory resides on a device.
class MemoryManager::Desc {
public:
  Desc() { reset(); }

  void reset() {
    ptr_ = nullptr;
    size_ = 0;
    chunk_ = nullptr;
    next_ = nullptr;
    prev_ = nullptr;
    nextFree_ = nullptr;
    prevFree_ = nullptr;
    owner_ = nullptr;
    nextMap_ = nullptr;
  }

  MemoryManagerFreeList *owner() const { return owner_; }
  bool isFree() const { return owner_ != nullptr; }
  bool isInUse() const { return !isFree(); }

  void setInUse() { owner_ = nullptr; }
  void setFree(MemoryManagerFreeList *owner) {
    assert(owner != nullptr);
    owner_ = owner;
  }

private:
  friend class MemoryManager;
  friend class MemoryManager::Chunk;
  friend class MemoryManager::DescMap;
  friend class MemoryManagerFreeList;

  // Fields owned by MemoryManager::Chunk.
  // Note: Fields read by other parts of the code are marked as atomic.
  char *ptr_;
  std::atomic<size_t> size_;

  // Sub allocation of which chunk.
  Chunk *chunk_;

  Desc *next_;
  Desc *prev_;

  // Fields owned by the MemoryManagerFreeList:
  // These fields are only accessed by the MemoryManagerFreeList and do not need
  // to be protected.
  //
  // Doubly-linked list of free buffers.
  Desc *nextFree_;
  Desc *prevFree_;

  // Remember the freelist owning the descriptor.
  // - nullptr: in use
  // - other values: the owning free list
  // Note: Atomic since it can be accessed by multiple threads while iterating
  // throuch descriptors checking for ownership. The value is only modified by
  // the freelist owning the descriptor.
  std::atomic<MemoryManagerFreeList *> owner_;

  // Fields owned by MemoryManager::DescMap.
  Desc *nextMap_;
};

MemoryManager::DescMap::DescMap(int mapSizeLog2)
    : bucketCount_(1 << mapSizeLog2), bucketMask_(bucketCount_ - 1),
      buckets_(new Desc *[bucketCount_]) {
  for (size_t i = 0; i < bucketCount_; ++i) {
    buckets_[i] = nullptr;
  }
}

MemoryManager::DescMap::~DescMap() { delete[] buckets_; }

uint32_t MemoryManager::DescMap::hash(void *ptr) {
  // From https://gist.github.com/badboy/6267743#64-bit-to-32-bit-hash-functions
  uintptr_t hashVal = reinterpret_cast<uintptr_t>(ptr);
  hashVal = (~hashVal) + (hashVal << 18); // key = (key << 18) - key - 1;
  hashVal = hashVal ^ (hashVal >> 31);
  hashVal = hashVal * 21; // key = (key + (key << 2)) + (key << 4);
  hashVal = hashVal ^ (hashVal >> 11);
  hashVal = hashVal + (hashVal << 6);
  hashVal = hashVal ^ (hashVal >> 22);
  return static_cast<uint32_t>(hashVal);
}

void MemoryManager::DescMap::add(Desc *desc) {
  assert(desc->nextMap_ == nullptr);
  assert(!desc->isFree());

  uint32_t h = hash(desc->ptr_);
  Desc **bucket = &buckets_[h & bucketMask_];
  desc->nextMap_ = *bucket;
  *bucket = desc;
}

MemoryManager::Desc *MemoryManager::DescMap::remove(void *ptr) {
  uint32_t h = hash(ptr);
  Desc **bucket = &buckets_[h & bucketMask_];
  Desc *result = *bucket;
  Desc *prevEntry = nullptr;
  while ((result != nullptr) && (result->ptr_ != ptr)) {
    prevEntry = result;
    result = result->nextMap_;
  }
  if (result != nullptr) {
    if (prevEntry == nullptr) {
      *bucket = result->nextMap_;
    } else {
      prevEntry->nextMap_ = result->nextMap_;
    }
    result->nextMap_ = nullptr;
  }
  return result;
}

MemoryManager::Chunk::Chunk(MemoryManager &cache, void *base, size_t size)
    : cache_(cache), base_(base), size_(size) {
  descs_ = cache.popDesc();
  descs_->reset();
  descs_->ptr_ = static_cast<char *>(base_);
  descs_->size_ = size_;
  descs_->chunk_ = this;

  cache_.freeList_->addToFreeList(descs_);
}

MemoryManager::Desc *MemoryManager::Chunk::splitFreeDesc(Desc *freeDesc,
                                                         size_t size) {
  // Split the descriptor. Keep the current freeDesc in the free list.
  assert(freeDesc->chunk_ == this);
  assert(freeDesc->isFree());

  // Remove the descriptor from the free list so that we can enter it at the
  // right place below.
  MemoryManagerFreeList *owner = freeDesc->owner();
  owner->removeFromFreeList(freeDesc);

  // Get a new descriptor for the result. And split off the first part of the
  // freeDesc to the result.
  Desc *result = cache_.popDesc();
  result->chunk_ = this;
  result->ptr_ = freeDesc->ptr_;
  result->size_ = size;

  // Update the existing descriptor with the remainder of the block.
  freeDesc->ptr_ = freeDesc->ptr_ + size;
  freeDesc->size_ = freeDesc->size_ - size;

  // Insert the new descriptor into buffer list.
  result->next_ = freeDesc;
  result->prev_ = freeDesc->prev_;
  if (descs_ == freeDesc) {
    assert(freeDesc->prev_ == nullptr);
    descs_ = result;
  } else {
    freeDesc->prev_->next_ = result;
  }
  freeDesc->prev_ = result;

  // Reenter freeDesc into the free list.
  owner->addToFreeList(freeDesc);

  assert(result->prevFree_ == nullptr);
  assert(result->nextFree_ == nullptr);
  assert(result->isInUse());
  return result;
}

void MemoryManager::Chunk::mergeFreeBlockDesc(MemoryManagerFreeList *owner,
                                              Desc *desc) {
  assert(desc != nullptr);
  assert(desc->isInUse());

  // The chunk where the block is located and into which we are attempting to
  // merge.
  MemoryManager::Chunk *chunk = desc->chunk_;
  size_t bufSize = desc->size_;
  MemoryManager &memoryManager = chunk->cache_;
  void *ptr = desc->ptr_;

  // Check if we can merge with the next buffer.
  const char *codePath = "no entry";
  MemoryManager::Desc *next = desc->next_;
  MemoryManager::Desc *prev = desc->prev_;
  if ((next != nullptr) && (next->owner() == owner)) {
    // Check if we can also merge with the previous buffer.
    if ((prev != nullptr) && (prev->owner() == owner)) {
      // Remove prev and next from free list before merging.
      owner->removeFromFreeList(prev);
      owner->removeFromFreeList(next);

      // Merge prev, desc and next, keeping prev on the free list.
      prev->size_ += bufSize + next->size_;
      // Remove desc and next from buffers.
      prev->next_ = next->next_;
      if (next->next_ != nullptr) {
        next->next_->prev_ = prev;
      }
      // Add the merged block into the free list.
      owner->addToFreeList(prev);
      codePath = "merge prev and next";
      memoryManager.pushDesc(desc);
      memoryManager.pushDesc(next);
    } else {
      // Remove next from free list before merging.
      owner->removeFromFreeList(next);

      // Merge desc and next, keeping next on the free list.
      next->ptr_ = desc->ptr_;
      next->size_ += bufSize;
      // Remove desc from buffers.
      if (chunk->descs_ == desc) {
        chunk->descs_ = next;
        next->prev_ = nullptr;
      } else {
        desc->prev_->next_ = next;
        next->prev_ = desc->prev_;
      }
      // Add the merged block into the free list.
      owner->addToFreeList(next);
      codePath = "merge next";
      memoryManager.pushDesc(desc);
    }
  } else if ((prev != nullptr) && (prev->owner() == owner)) {
    // Remove prev from free list before merging.
    owner->removeFromFreeList(prev);

    // Merge prev and desc, keeping prev on the free list.
    prev->size_ += bufSize;
    // Remove desc from buffers.
    prev->next_ = desc->next_;
    if (desc->next_ != nullptr) {
      desc->next_->prev_ = prev;
    }
    // Add the merged block into the free list.
    owner->addToFreeList(prev);
    codePath = "merge prev";
    memoryManager.pushDesc(desc);
  } else {
    // Add desc to the free list.
    owner->addToFreeList(desc);
    codePath = "no merge";
  }
  if (memoryManager.doLog_) {
    memoryManager.logErr("mergeFreeBlockDesc(" + TWINE_PTR(ptr) + " ; " +
                         TWINE(bufSize) + ") -  (" + codePath + ")");
  }
}

void MemoryManager::Chunk::selfCheck() const {
#define SELF_CHECK(cond)                                                       \
  if (!(cond)) {                                                               \
    cache_.logErr("!!! selfCheck !!! - " #cond);                               \
    abort();                                                                   \
  }

  // First buffer points to base of pre-allocated memory.
  SELF_CHECK(descs_->ptr_ == base_);

  Desc *curDesc = descs_;
  Desc *aheadDesc = curDesc;
  SELF_CHECK(curDesc != nullptr);
  SELF_CHECK(curDesc->prev_ == nullptr);

  // Check no missing buffers, no cycle
  while (curDesc != nullptr) {
    if (curDesc->isFree()) {
      MemoryManagerFreeList *owner = curDesc->owner();
      if (!owner->contains(curDesc)) {
        cache_.logErr(" !!! selfCheck !!! owner: " + TWINE_PTR(owner) +
                      " does not contain: " + TWINE_PTR(curDesc));
        abort();
      }
      SELF_CHECK((curDesc->prevFree_ == nullptr) ||
                 (curDesc->prevFree_->owner_ == owner));
      SELF_CHECK((curDesc->nextFree_ == nullptr) ||
                 (curDesc->nextFree_->owner_ == owner));
    } else {
      SELF_CHECK(curDesc->nextFree_ == nullptr);
      SELF_CHECK(curDesc->prevFree_ == nullptr);
    }
    Desc *nextDesc = curDesc->next_;
    if (nextDesc != nullptr) {
      SELF_CHECK((curDesc->ptr_ + curDesc->size_) == nextDesc->ptr_);
      SELF_CHECK(nextDesc->prev_ == curDesc);
    } else {
      SELF_CHECK((curDesc->ptr_ + curDesc->size_) ==
                 (static_cast<const char *>(base_) + size_));
    }
    // Step ahead.
    if ((aheadDesc != nullptr) && (aheadDesc->next_ != nullptr)) {
      aheadDesc = aheadDesc->next_->next_;
      if (aheadDesc == curDesc) {
        cache_.logErr("!!! selfCheck !!! cycle - buffers");
        abort();
      }
    }
    curDesc = nextDesc;
  }

#undef SELF_CHECK
}

MemoryManagerFreeList::MemoryManagerFreeList(const DeviceContext *context,
                                             MemoryManager &memoryManager,
                                             MemoryManagerFreeList *parent)
    : ctx_(context), memoryManager_(memoryManager), parent_(parent) {
  // Either parent and context are both null for the free list associated with
  // the MemoryManager or both are set for free lists associated with a
  // DeviceContext.
  assert(((ctx_ == nullptr) && (parent_ == nullptr)) ||
         ((ctx_ != nullptr) && (parent_ != nullptr)));
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    freeBuffers_[i] = nullptr;
  }
}

MemoryManagerFreeList::~MemoryManagerFreeList() { flushToParent(); }

void MemoryManagerFreeList::flushToParent() {
  std::lock_guard<std::mutex> lock(memoryManager_.mu_);
  if (memoryManager_.doLog_) {
    memoryManager_.logErr("Flush all to parent from " + TWINE_PTR(this));
  }
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    flushFreeBuffersToParent(i);
  }
}

void MemoryManagerFreeList::flushFreeBuffersToParent(int id) {
  if (parent_ == nullptr)
    return;

  // Early return if no chunks have been allocated yet.
  if (memoryManager_.chunks_ == nullptr)
    return;

  if (memoryManager_.doLog_) {
    memoryManager_.logErr("flushing " + TWINE_PTR(this) + " id: " + TWINE(id) +
                          " to parent " + TWINE_PTR(parent_));
  }
  size_t bufSize = 0;
  MemoryManager::Desc *curDesc = freeBuffers_[id];
  while (curDesc != nullptr) {
    bufSize += curDesc->size_;
    MemoryManager::Desc *nextDesc = curDesc->nextFree_;

    // "Remove" from free list. Avoid manipulating the list itself since it
    // will be entirely reset after this loop.
    assert(curDesc->isFree());
    curDesc->nextFree_ = nullptr;
    curDesc->prevFree_ = nullptr;
    curDesc->setInUse(); // Remove ownership by this.
    memoryManager_.freeCount_--;
    memoryManager_.freeSize_ -= curDesc->size_;

    // Merge the block with surrounding free blocks if possible and add it
    // to the parent free list.
    curDesc->chunk_->mergeFreeBlockDesc(parent_, curDesc);
    curDesc = nextDesc;
  }
  // Reset the current freelist to empty.
  freeBuffers_[id] = nullptr;

  if (memoryManager_.doLog_) {
    memoryManager_.logErr("flushing " + TWINE_PTR(this) + " id: " + TWINE(id) +
                              " to parent " + TWINE_PTR(parent_) + " : Done",
                          0, bufSize);
  }
}

bool MemoryManagerFreeList::empty() {
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    if (freeBuffers_[i] != nullptr)
      return false;
  }
  return true;
}

void MemoryManagerFreeList::addToFreeList(MemoryManager::Desc *desc) {
  assert(!desc->isFree());
  assert(desc->nextFree_ == nullptr);
  assert(desc->prevFree_ == nullptr);
  assert(desc->chunk_ != nullptr);

  size_t descSize = desc->size_;

  desc->setFree(this);
  // Add the descriptor to the currently active (aka newest) free buffers. Each
  // list of free buffers is ordered descending by size.
  MemoryManager::Desc *&activeList = freeBuffers_[primaryFreeBuffers_];
  if (memoryManager_.doLog_) {
    memoryManager_.logErr(
        "In addToFreeList for (" + TWINE_PTR(this) +
        ") using free buffer id: " + TWINE(primaryFreeBuffers_) + " address " +
        TWINE_PTR(desc->ptr_));
  }
  if (activeList == nullptr) {
    activeList = desc;
  } else if (activeList->size_ >= descSize) {
    // Enter desc at head.
    assert(activeList->prevFree_ == nullptr);
    activeList->prevFree_ = desc;
    desc->nextFree_ = activeList;
    activeList = desc;
  } else {
    assert(activeList != nullptr);
    MemoryManager::Desc *prevDesc = activeList;
    MemoryManager::Desc *nextDesc = activeList->nextFree_;
    while ((nextDesc != nullptr) && (nextDesc->size_ < descSize)) {
      prevDesc = nextDesc;
      nextDesc = prevDesc->nextFree_;
    }
    prevDesc->nextFree_ = desc;
    desc->prevFree_ = prevDesc;
    desc->nextFree_ = nextDesc;
    if (nextDesc != nullptr) {
      nextDesc->prevFree_ = desc;
    }
  }

  memoryManager_.freeSize_ += desc->size_.load(std::memory_order_relaxed);
  memoryManager_.freeCount_++;
}

void MemoryManagerFreeList::removeFromFreeList(MemoryManager::Desc *desc) {
  assert(desc->isFree());

  memoryManager_.freeSize_ -= desc->size_.load(std::memory_order_relaxed);
  memoryManager_.freeCount_--;

  if (desc->prevFree_ == nullptr) {
    // desc is at the head of a freeBuffers_ list. Identify which list and
    // unhook.
    int i = 0;
    for (; i < kNumFreeBufferLists; ++i) {
      if (freeBuffers_[i] == desc) {
        freeBuffers_[i] = desc->nextFree_;
        if (freeBuffers_[i] != nullptr) {
          freeBuffers_[i]->prevFree_ = nullptr;
        }
        break; // Found the entry, no need to iterate further.
      }
    }
    assert(i < kNumFreeBufferLists);
  } else {
    desc->prevFree_->nextFree_ = desc->nextFree_;
    if (desc->nextFree_ != nullptr) {
      desc->nextFree_->prevFree_ = desc->prevFree_;
    }
  }
  desc->nextFree_ = nullptr;
  desc->prevFree_ = nullptr;
  desc->setInUse();
}

void MemoryManagerFreeList::selfCheck() const {
  if (!memoryManager_.doSelfCheck_ || memoryManager_.chunks_ == nullptr)
    return;

#define SELF_CHECK(cond)                                                       \
  if (!(cond)) {                                                               \
    memoryManager_.logErr("!!! selfCheck !!! - " #cond);                       \
    abort();                                                                   \
  }

  // Make sure all lists are checked.
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    // Check free markers, no cycle, increasing sizes.
    MemoryManager::Desc *curDesc = freeBuffers_[i];
    MemoryManager::Desc *aheadDesc = curDesc;
    if (curDesc != nullptr) {
      SELF_CHECK(curDesc->prevFree_ == nullptr);
    }
    size_t prevSize = 0;
    while (curDesc != nullptr) {
      MemoryManager::Desc *next = curDesc->nextFree_;
      SELF_CHECK(curDesc->isFree());
      SELF_CHECK(curDesc->owner() == this);
      SELF_CHECK(prevSize <= curDesc->size_);
      prevSize = curDesc->size_;
      SELF_CHECK((next == nullptr) || (next->prevFree_ == curDesc));
      if ((aheadDesc != nullptr) && (aheadDesc->nextFree_ != nullptr)) {
        aheadDesc = aheadDesc->nextFree_->nextFree_;
        if (aheadDesc == curDesc) {
          memoryManager_.logErr("!!! selfCheck !!! cycle - free buffers");
          abort();
        }
      }
      curDesc = next;
    }
  }

  if (parent_ != nullptr) {
    // Check all levels of the hierarchical free lists.
    parent_->selfCheck();
  } else {
    // Reached the bottom of the hierarchical free lists, so we now check the
    // associated memory manager.
    memoryManager_.selfCheck();
  }

#undef SELF_CHECK
}

bool MemoryManagerFreeList::contains(MemoryManager::Desc *desc) const {
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    MemoryManager::Desc *curDesc = freeBuffers_[i];
    while (curDesc != nullptr) {
      if (curDesc == desc)
        return true;
      curDesc = curDesc->nextFree_;
    }
  }
  return false;
}

std::string &MemoryManager::getDeviceDescription() const {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  if (deviceDescription_.empty()) {
    deviceDescription_ = (device().api() + "[" + llvm::Twine(device().id()) +
                          (isHost() ? " - host]" : "]"))
                             .str();
    // Print the initial information about this MemoryManager.
    llvm::errs() << "##(" << deviceDescription_ << ":" << this
                 << ")##: memory_manager(chunk_size: " << chunkSize_
                 << " ; max_cache_size: " << maxCacheSize_
                 << " ; free_list: " << freeList_.get() << ")\n";
  }
  return deviceDescription_;
}

void MemoryManager::logErr(llvm::Twine msg) const {
  logErr(msg, 0, 0, /* printInfoSize = */ false);
}

void MemoryManager::logErr(llvm::Twine msg, size_t info, int64_t size,
                           bool printInfoSize) const {
  if (chunkSize_ == 0)
    return;

  // Using a shared mutex to avoid interleaved output.
  static std::mutex mu;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (printInfoSize) {
      llvm::errs() << "##(" << getDeviceDescription() << ")##: " << msg
                   << " - , " << cacheSize_ << ", " << chunkCount_ << ", "
                   << freeSize_ << ", " << freeCount_ << ", " << info << ", "
                   << size << "\n";
    } else {
      if (!msg.isTriviallyEmpty())
        llvm::errs() << "##(" << getDeviceDescription() << ")##: " << msg
                     << "\n";
    }
  }
}

MemoryManager::MemoryManager(const Device &device, size_t maxSize,
                             size_t chunkSize, size_t alignment, bool isHost)
    : device_(device), isHost_(isHost), maxCacheSize_(maxSize),
      chunkSize_(chunkSize),
      // A minimum of 64 byte alignment for allocations from the memory manager.
      alignmentOffset_(std::max(64UL, alignment) - 1),
      alignmentMask_(~(std::max(64UL, alignment) - 1)),
      allocatedMap_(chunkSize_ == 0 ? 0 : 10) {
  // chunks_ will be lazily initialized on the first allocation request.
  // Alignment masking only works with powers of 2.
  assert(llvm::isPowerOf2_64(alignment));

  // Allocate a freelist that maintains free memory blocks not associated with a
  // DeviceContext (aka stream).
  freeList_ = std::make_unique<MemoryManagerFreeList>(nullptr, *this, nullptr);

  // Determine state of debugging flags.
  doLog_ = DeviceContext::configAsBool("memory_manager_log", false);
  doSelfCheck_ =
      DeviceContext::configAsBool("memory_manager_self_check", false);
  if (doLog_) {
    logErr(llvm::Twine()); // Print initial config of memory manager.
  }
}

MemoryManager::~MemoryManager() {
  if (doLog_) {
    logErr("destructor");
  }
  assert(chunks_ == nullptr);
  assert(freeList_->empty());
  assert(unusedDescs_ == nullptr);
}

size_t MemoryManager::allocationAlignment() const {
  return alignmentOffset_ + 1;
}

bool MemoryManager::isHost() const { return isHost_; }

Device::MemInfo MemoryManager::getMemoryInfo() const {
  Device::MemInfo info;
  std::lock_guard<std::mutex> lock(mu_);

  info.total = maxCacheSize_;
  info.free = maxCacheSize_ >= cacheSize_
                  ? (maxCacheSize_ - cacheSize_) + freeSize_
                  : freeSize_;

  return info;
}

bool MemoryManager::isEnabled() const { return maxCacheSize_ > 0; }

bool MemoryManager::allocateChunk() {
  if (shouldUseMaxSize()) {
    // Ensure that we have grown chunkSize_ to the maximum.
    growChunkSizeTo(maxCacheSize_);
  }

  // Avoid allocating chunks beyond the maxCacheSize_ or available memory.
  size_t maxSize = maxCacheSize_;
  if (!isHost()) {
    // Limit to available GPU memory.
    ErrorOr<Device::MemInfo> memInfo = device().getMemoryInfo();
    maxSize = memInfo.isError() ? maxCacheSize_
                                : std::min((*memInfo).free, maxCacheSize_);
  }

  const size_t allocSize =
      std::min(chunkSize_, maxSize -
                               // Avoid overflowing the subtraction by guarding
                               // against cacheSize_ being larger than maxSize.
                               std::min(maxSize, cacheSize_));

  const size_t minChunkSize = std::min(128UL * 1024, (chunkSize_ * 25) / 100);
  if (allocSize < minChunkSize) {
    return false;
  }
  void *ptr = nullptr;
  ErrorOrSuccess err = device().chunkAlloc(&ptr, allocSize, isHost());
  if (err.isError()) {
    if (doLog_) {
      logErr("allocate_chunk(" + TWINE(allocSize) + ") - failed with " +
             err.getError());
    }
    // Keep chunks_ unmodified.
    return false;
  }
  Chunk *chunk = new Chunk(*this, ptr, allocSize);
  chunk->next_ = chunks_;
  chunks_ = chunk;
  cacheSize_ += allocSize;
  ++chunkCount_;
  if (doLog_) {
    logErr("allocate_chunk(" + TWINE_PTR(ptr) + " ; " + TWINE(allocSize) + ")",
           0, 0);
  }
  return true;
}

const Device &MemoryManager::device() const { return device_; }

void MemoryManager::selfCheck() const {
  if (!doSelfCheck_ || chunks_ == nullptr) {
    return;
  }

  Chunk *curChunk = chunks_;
  while (curChunk != nullptr) {
    curChunk->selfCheck();
    curChunk = curChunk->next_;
  }

  if (doLog_) {
    logErr("selfCheck OK");
  }
}

MemoryManager::Desc *MemoryManager::popDesc() {
  if (unusedDescs_ == nullptr) {
    return new Desc();
  }
  Desc *result = unusedDescs_;
  unusedDescs_ = result->next_;
  // Make sure all values are reset before using.
  result->reset();
  return result;
}

void MemoryManager::pushDesc(Desc *desc) {
  // Do not preemptively clear the desc. The reset work could be unnecessary
  // and reset before writing will load the memory locations into the CPU cache.
  desc->next_ = unusedDescs_;
  unusedDescs_ = desc;
}

bool MemoryManager::shouldUseMaxSize() {
  // The state whether this process has determined to use the full memory
  // manager in as few chunks as possible is kept in an atomic integer. A value
  // of zero is interpreted as false, while a value greater other than zero is
  // interpreted as true.
  // We use a relaxed load since we do not require strong synchronization. If
  // another thread races to set the value to true we potentially miss
  // allocating one chunk at maximum size. However even with proper locking
  // this race would exist.
  return useMaxSize_.load(std::memory_order_relaxed) != 0;
}

void MemoryManager::enableUseMaxSize() {
  // We use a relaxed store since we do not require strong synchronization. If
  // another thread races to set the value to "true" we potentially miss
  // allocating one chunk at maximum size. However even with proper locking
  // this race would exist.
  useMaxSize_.store(1, std::memory_order_relaxed);
}

void MemoryManager::growChunkSizeTo(size_t newChunkSize) {
  // Limit newChunkSize to fit overall cache size within maxCacheSize_.
  newChunkSize = std::min(
      newChunkSize, maxCacheSize_ -
                        // Avoid overflowing the subtraction by guarding
                        // against cacheSize_ being larger than maxCacheSize_.
                        std::min(maxCacheSize_, cacheSize_));

  // Ensure to not fall below current chunkSize_.
  size_t oldChunkSize = chunkSize_;
  chunkSize_ = std::max(newChunkSize, chunkSize_);
  if (doLog_ && (chunkSize_ != oldChunkSize)) {
    if (isHost()) {
      logErr("growing_chunk_size_to(" + TWINE(chunkSize_) + ")");
    } else {
      ErrorOr<Device::MemInfo> memInfo = device().getMemoryInfo();
      logErr("growing_chunk_size_to(" + TWINE(chunkSize_) +
             ", free: " + TWINE((*memInfo).free) +
             ", total: " + TWINE((*memInfo).total) + ")");
    }
  }
}

std::atomic<int32_t> MemoryManager::useMaxSize_ = {0};

void MemoryManager::clear() {
  if (chunks_ == nullptr) {
    return;
  }

  selfCheck();

  Desc *desc = freeList_->freeBuffers_[freeList_->primaryFreeBuffers_];
  while (desc != nullptr) {
    // Ensure each free desc is spanning the whole chunk area. We should have
    // one desc per chunk at this point.
    assert(desc->ptr_ == desc->chunk_->base_);
    assert(desc->size_ == desc->chunk_->size_);
    desc = desc->nextFree_;
  }

  Chunk *chunk = chunks_;
  while (chunk != nullptr) {
    Chunk *nextChunk = chunk->next_;
    Desc *desc = chunk->descs_;
    // Ensure each chunk only contains one free desc.
    assert(desc->ptr_ == chunk->base_);
    assert(desc->size_ == chunk->size_);
    assert(desc->prev_ == nullptr);
    assert(desc->next_ == nullptr);
    assert(desc->isFree());

    freeList_->removeFromFreeList(desc);
    delete desc;
    // Release the chunk memory.
    if (doLog_) {
      logErr("free_chunk(" + TWINE_PTR(chunk->base_) + ", " +
             TWINE(chunk->size_) + ")");
    }
    ErrorOrSuccess err = device().chunkFree(chunk->base_, isHost());
    if (err.isError()) {
      logErr(err.getError());
      abort();
    }
    assert(!err.isError());
    delete chunk;

    chunk = nextChunk;
  }
  chunks_ = nullptr;
  for (int i = 0; i < MemoryManagerFreeList::kNumFreeBufferLists; ++i) {
    assert(freeList_->freeBuffers_[i] == nullptr);
  }

  Desc *cur = unusedDescs_;
  while (cur != nullptr) {
    Desc *next = cur->next_;
    delete cur;
    cur = next;
  }
  unusedDescs_ = nullptr;
}

MemoryManagerFreeList *MemoryManager::freeList() const {
  return freeList_.get();
}

ErrorOr<void *> MemoryManagerFreeList::allocate(size_t size,
                                                Device::AllocMode mode) {
  assert(size != 0);

  if (memoryManager_.chunkSize_ != 0) {
    size = (size + memoryManager_.alignmentOffset_) &
           memoryManager_.alignmentMask_;

    MemoryManager::Desc *desc = nullptr;
    {
      std::lock_guard<std::mutex> lock(memoryManager_.mu_);
      if (size > memoryManager_.chunkSize_) {
        // Check heuristics if chunk size should be updated.

        if (memoryManager_.chunks_ == nullptr) {
          // Heuristic: If the first allocation does overrun the chunk size,
          // then we override the "memory_manager_chunk_percent" setting and
          // adjust chunk size to max cache size. We take this situation as an
          // indication that a large amount of memory will be necessary during
          // the execution of a model.
          //
          // Avoid aggressive growth if the allocation will not fit within the
          // maximum memory manager size to start.
          if (size <= memoryManager_.maxCacheSize_) {
            assert(memoryManager_.cacheSize_ == 0);
            MemoryManager::enableUseMaxSize();
          } else {
            if (memoryManager_.doLog_) {
              memoryManager_.logErr("allocate(" + TWINE(size) +
                                    ") - skip growChunkSizeTo()");
            }
          }
        }

        if (MemoryManager::shouldUseMaxSize()) {
          // Heuristic: The code above or another memory manager has determined
          // that we should grow to the largest possible chunk size. Update
          // the chunk size to all remaining memory for future chunk
          // allocations.
          memoryManager_.growChunkSizeTo(memoryManager_.maxCacheSize_);
        }
      }

      if (size <= memoryManager_.chunkSize_) {
        selfCheck();
        desc = allocateDesc(size);
      }
    }
    if (desc != nullptr) {
      // Allocation from memory manager was successful, return the value.
      return desc->ptr_;
    }

    // If only allowing allocations from the memory manager, raise an error at
    // this point, printing as much debugging information as needed.
    static bool useOnlyMemoryManager =
        DeviceContext::configAsBool("memory_manager_only", false);
    if (useOnlyMemoryManager) {
      if (memoryManager_.doLog_) {
        memoryManager_.logErr(
            "allocate(" + TWINE(size) + ") - memory_manager_only OOM", 0, 0);
        // Print more debugging information about the largest available blocks.
        MemoryManagerFreeList *cur = this;
        while (cur != nullptr) {
          for (int i = 0; i < kNumFreeBufferLists; ++i) {
            if (cur->freeBuffers_[i] != nullptr) {
              memoryManager_.logErr(
                  "max block on " + TWINE_PTR(cur) + " id: " + TWINE(i) +
                  ": (" + TWINE_PTR(cur->freeBuffers_[i]->ptr_) + " ; " +
                  TWINE(cur->freeBuffers_[i]->size_) + ")");
            }
          }
          cur = cur->parent_;
        }
      }
      return Error(
          "[Use only memory manager mode]: No room left in memory manager: " +
          memoryManager_.getDeviceDescription() + " on " + TWINE_PTR(this) +
          " (size: " + humanMemorySize(size) +
          " ; free: " + humanMemorySize(memoryManager_.freeSize_) +
          " ; cache_size: " + humanMemorySize(memoryManager_.cacheSize_) +
          " ; max_cache_size: " +
          humanMemorySize(memoryManager_.maxCacheSize_) + ")");
    }
  }

  // Allocate straight from device driver if cache was not able to fulfill
  // the request. If the cache is disabled, the we pass on the allocation
  // mode.
  void *ptr = nullptr;
  if (ctx_ == nullptr) {
    // In the case of the bottom most freelist being consulted for allocation
    // we might not have a DeviceContext associated, so we fail the allocation
    // with an error.
    if (memoryManager_.doLog_) {
      memoryManager_.logErr("allocate_buffer(" + TWINE(size) +
                            ") - failed due to missing context");
    }
    return Error("memAlloc failed due to missing context");
  }
  ErrorOrSuccess err = ctx_->memAlloc(
      &ptr, size,
      // If allocating for host just forward that.
      (mode == Device::AllocMode::HOST) ? Device::AllocMode::HOST
      // Ensure that all allocations are sync when using a memory manager.
      : (memoryManager_.chunkSize_ != 0) ? Device::AllocMode::SYNC
                                         : mode);
  if (err.isError()) {
    if (memoryManager_.doLog_) {
      memoryManager_.logErr("allocate_buffer(" + TWINE(size) +
                            ") - failed with " + err.getError());
    }
    return err.takeError();
  }
  if (memoryManager_.doLog_) {
    memoryManager_.logErr(
        "allocate_buffer(" + TWINE_PTR(ptr) + " ; " + TWINE(size) + ")", 0, 0);
  }
  assert(ptr != nullptr);
  return ptr;
}

MemoryManager::Desc *MemoryManagerFreeList::allocateDesc(size_t size) {
  assert((size & memoryManager_.alignmentMask_) == size);
  if (memoryManager_.doLog_) {
    memoryManager_.logErr(
        "In allocateDesc with size " + TWINE(size) +
        " and primary free buffer id: " + TWINE(primaryFreeBuffers_));
  }
  int iterations = 0;
  // Try allocating from all free buffer lists in this MemoryManagerFreeList,
  // starting with the primary free buffer list.
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    int freeListIdToUse = (primaryFreeBuffers_ + i) % kNumFreeBufferLists;
    // Find a fit in the free list.
    MemoryManager::Desc *desc = freeBuffers_[freeListIdToUse];
    while ((desc != nullptr) && (desc->size_ < size)) {
      desc = desc->nextFree_;
      ++iterations;
    }
    if (desc != nullptr) {
      assert(desc->isFree());
      if (desc->size_ == size) {
        removeFromFreeList(desc);
      } else {
        desc = desc->chunk_->splitFreeDesc(desc, size);
      }
      assert(desc->nextFree_ == nullptr);
      assert(desc->prevFree_ == nullptr);
      assert(desc->isInUse());

      memoryManager_.allocatedMap_.add(desc);

      if (memoryManager_.doLog_) {
        memoryManager_.logErr(
            "allocate on " + TWINE_PTR(this) +
                " from free buffer at id: " + TWINE(freeListIdToUse) + " (" +
                TWINE_PTR(desc->ptr_) + " ; " + TWINE(size) + ")",
            iterations, -size);
      }

      return desc;
    }
  }

  // Delegate to parent freelist if available.
  if (parent_ != nullptr) {
    return parent_->allocateDesc(size);
  }

  // We only ever allocate chunks in the bottom most free list which is
  // associated with the MemoryManager directly.
  assert(parent_ == nullptr);
  if (!memoryManager_.allocateChunk()) {
    if (memoryManager_.doLog_) {
      memoryManager_.logErr("allocate(" + TWINE(size) + ") - nullptr (" +
                            TWINE(iterations) + ")");
    }
    // No success allocating a new Chunk, so we bail from allocating in the
    // cache. Instead we will try to allocate directly from the device and if
    // that fails, then we propagate the failure reported by the device.
    return nullptr;
  }
  // Allocated a new chunk, try again.
  return allocateDesc(size);
}

ErrorOrSuccess MemoryManagerFreeList::free(void *ptr) {
  if (ptr == nullptr)
    return success();

  if (memoryManager_.chunks_ != nullptr) {
    if (memoryManager_.doLog_) {
      memoryManager_.logErr("Starting free of " + TWINE_PTR(ptr) +
                            " on free list " + TWINE_PTR(this) +
                            " with free id: " + TWINE(primaryFreeBuffers_));
    }
    std::lock_guard<std::mutex> lock(memoryManager_.mu_);
    selfCheck();

    MemoryManager::Desc *desc = memoryManager_.allocatedMap_.remove(ptr);
    if (desc != nullptr) {
      freeDesc(desc);
      if ((ctx_ != nullptr) && !handlerPending_) {
        // Free the list preceeding the next primary so that we can keep buffers
        // cached for an extended period.
        // primaryFreeBuffers + 0: current primary
        // primaryFreeBuffers + 1: next primary to be used, should contain null
        // now
        // primaryFreeBuffers + 2: oldest list of free buffers, to be released
        // to parent (for the default of 2 lists, this is equivalent to the
        // current primary)
        int toFreeId = (primaryFreeBuffers_ + 2) & (kNumFreeBufferLists - 1);
        primaryFreeBuffers_ =
            (primaryFreeBuffers_ + 1) & (kNumFreeBufferLists - 1);
        assert(freeBuffers_[primaryFreeBuffers_] == nullptr);
        handlerPending_ = true;
        ErrorOrSuccess err = ctx_->enqueueEventHandler(
            // We need to ensure that ctx_ survives while this handler is
            // pending, so we capture the DeviceContextRef.
            [contextRef = DeviceContextRef::copy(ctx_), freelist = this,
             id = toFreeId]() {
              std::lock_guard<std::mutex> lock(freelist->memoryManager_.mu_);
              if (freelist->memoryManager_.doLog_) {
                freelist->memoryManager_.logErr("Running flush handler for " +
                                                TWINE_PTR(freelist) +
                                                " id: " + TWINE(id));
              }
              assert(freelist->handlerPending_);
              freelist->flushFreeBuffersToParent(id);
              freelist->handlerPending_ = false;
            });
        if (err.isError()) {
          return err.takeError();
        }
        if (memoryManager_.doLog_) {
          memoryManager_.logErr("Enqueued flush handler for " +
                                TWINE_PTR(this) + " id: " + TWINE(toFreeId));
        }
      }
      if (memoryManager_.doLog_) {
        memoryManager_.logErr("Completed free of " + TWINE_PTR(ptr) +
                              " after queuing flush handler");
      }
      return success();
    }
  }

  if (memoryManager_.doLog_) {
    memoryManager_.logErr("free_buffer(" + TWINE_PTR(ptr) + ")");
  }
  return ctx_->memFree(ptr, memoryManager_.isHost());
}

void MemoryManagerFreeList::freeDesc(MemoryManager::Desc *desc) {
  void *ptr = desc->ptr_;
  size_t bufSize = desc->size_;
  desc->chunk_->mergeFreeBlockDesc(this, desc);
  if (memoryManager_.doLog_) {
    memoryManager_.logErr("free on " + TWINE_PTR(this) + " (" + TWINE_PTR(ptr) +
                              " ; " + TWINE(bufSize) + ")",
                          0, bufSize);
  }
}

MemoryManager *MemoryManager::onDevice(const Device &device,
                                       Device::MemInfo memInfo,
                                       size_t alignment, int64_t memoryLimit) {
  // Do not use memory manager if memory manager has been disabled by the
  // calling code (empty memInfo).
  // This is for example the case for CPU Device: We are relying on malloc
  // optimizations instead.
  if (memInfo.free == 0 || memoryLimit == 0)
    return new MemoryManager(device, 0, 0, alignment, false);

  // Determine cache size
  bool isDefaultCacheSize = false;
  size_t maxCacheSize;
  if (memoryLimit > 0) // Try to use parameter value first
    maxCacheSize = static_cast<size_t>(memoryLimit);
  else // Otherwise look in the config
    maxCacheSize = static_cast<size_t>(DeviceContext::configAsInt(
        "memory_manager_size", 0, &isDefaultCacheSize));

  // Disable the memory manager if the requested cache size is 0
  if (maxCacheSize == 0 && !isDefaultCacheSize)
    return new MemoryManager(device, 0, 0, alignment, false);

  // If there's no provided size or if the provided size is larger than
  // the free memory, use a percentage of the available memory
  if (isDefaultCacheSize || maxCacheSize > memInfo.free) {
    // By default use 90% of the available memory
    int64_t percentMemoryLimit =
        DeviceContext::configAsInt("memory_manager_size_percent", 90);

    if (maxCacheSize > memInfo.free)
      llvm::errs() << "DeviceContext.MemoryManager: Requested maximum memory "
                      "manager size ("
                   << humanMemorySize(maxCacheSize)
                   << ") does not fit into available free memory ("
                   << humanMemorySize(memInfo.free) << "), using "
                   << percentMemoryLimit << "% of all free memory instead.\n";

    maxCacheSize = (memInfo.free * percentMemoryLimit) / 100;
  }

  // Ensure the cache size is aligned
  maxCacheSize = llvm::alignToPowerOf2(maxCacheSize, 256 * KB);

  // Get chunk size as percentage of max cache size. Limit to between 0 and 100.
  size_t chunkSizePercent =
      std::max(static_cast<size_t>(0),
               std::min(static_cast<size_t>(100),
                        static_cast<size_t>(DeviceContext::configAsInt(
                            "memory_manager_chunk_percent", 2))));

  size_t chunkSize = std::min(
      maxCacheSize,
      llvm::alignToPowerOf2((maxCacheSize * chunkSizePercent) / 100, 256 * KB));

  auto result = new MemoryManager(device, maxCacheSize, chunkSize, alignment,
                                  /* isHost = */ false);
  if (result->doLog_) {
    result->logErr("initial memInfo(free: " + humanMemorySize(memInfo.free) +
                   ", total: " + humanMemorySize(memInfo.total) + ")");
  }
  return result;
}
MemoryManager *MemoryManager::onHost(const Device &device, size_t alignment) {

  size_t maxCacheSize = llvm::alignToPowerOf2(
      DeviceContext::configAsInt("host_memory_manager_size", 16 * GB),
      256 * KB);

  // Disable host memory manager for Metal devices due to pointer tracking
  // issues. Metal tracks buffers by exact pointer match, but the memory manager
  // uses sub-allocation with offset pointers that aren't in the tracking map.
  // TODO: GEX-2688 - Enable host memory manager for Metal.
  if (device.api() == Device::metalAPI) {
    maxCacheSize = 0;
  }

  size_t chunkSize = std::min(
      maxCacheSize,
      static_cast<size_t>(llvm::alignToPowerOf2(
          (maxCacheSize *
           DeviceContext::configAsInt("host_memory_manager_chunk_percent", 2)) /
              100,
          256 * KB)));

  return new MemoryManager(device, maxCacheSize, chunkSize, alignment,
                           /* isHost = */ true);
}

} // namespace M::AsyncRT
