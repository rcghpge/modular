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
#include "Support/internal/Tracy/Tracy.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>

// Helpers for logging.
#define TWINE(val) (llvm::Twine(val))

#define TWINE_PTR(ptr)                                                         \
  (llvm::Twine("0x") + llvm::Twine::utohexstr(reinterpret_cast<uintptr_t>(ptr)))

// Guard a logErr() call inside MemoryManager methods.
#define LOG_ERR(...)                                                           \
  if (doLog_)                                                                  \
  logErr(__VA_ARGS__)

// Guard a logMsg() call inside MemoryManagerFreeList methods.
#define LOG_MSG(...)                                                           \
  if (memoryManager_.doLog_)                                                   \
  logMsg(__VA_ARGS__)

static constexpr int KB = 1024;                  // NOLINT
static constexpr size_t GB = 1024 * 1024 * 1024; // NOLINT

namespace M::Driver {

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
  std::unique_ptr<Chunk> next_ = nullptr;
  Desc *descs_ = nullptr; // Ordered list of all memory regions in this chunk.

  // VMM (CUDA Virtual Memory Management) backing. When `vmmBacked_` is true,
  // `base_` is a virtual address reservation and the physical memory is a
  // vector of granule handles (one per `pageSize_` slice of [base_,
  // base_+size_)). Plain (`cuMemAlloc`) chunks leave these empty. See
  // MemoryManager VMM path.
  bool vmmBacked_ = false;
  size_t pageSize_ = 0;
  std::vector<uint64_t> vmmHandles_;
  // Single-arena lazy backing: one bit per granule slice of [base_,
  // base_+size_) (same length as vmmHandles_ for VMM chunks; empty otherwise).
  // A bit is clear when that granule has not yet been physically backed (lazy)
  // or has been unmapped; `vmmHandles_[i]` is only valid when
  // `granuleBacked_[i]`.
  llvm::BitVector granuleBacked_;
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
    allocatedBy_ = nullptr;
  }

  MemoryManagerFreeList *owner() const { return owner_; }
  bool isFree() const { return owner_ != nullptr; }
  bool isInUse() const { return !isFree(); }

  void setInUse() { owner_ = nullptr; }
  void setFree(MemoryManagerFreeList *owner) {
    assert(owner != nullptr);
    owner_ = owner;
    allocatedBy_ = nullptr;
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
  // through descriptors checking for ownership. The value is only modified by
  // the freelist owning the descriptor.
  std::atomic<MemoryManagerFreeList *> owner_;

  // Fields owned by MemoryManager::DescMap.
  Desc *nextMap_;

  // Tracks which MemoryManagerFreeList originally allocated this descriptor
  // (set in allocateDesc(), cleared in reset()). Used by free() to detect when
  // a stale DeviceBuffer frees an address now owned by a different free list.
  MemoryManagerFreeList *allocatedBy_;
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

MemoryManager::Desc *MemoryManager::DescMap::find(void *ptr) const {
  uint32_t h = hash(ptr);
  Desc *result = buckets_[h & bucketMask_];
  while ((result != nullptr) && (result->ptr_ != ptr))
    result = result->nextMap_;
  return result;
}

void MemoryManager::DescMap::redirectAllocatedBy(MemoryManagerFreeList *from,
                                                 MemoryManagerFreeList *to) {
  for (size_t i = 0; i < bucketCount_; ++i) {
    for (Desc *desc = buckets_[i]; desc != nullptr; desc = desc->nextMap_) {
      if (desc->allocatedBy_ == from)
        desc->allocatedBy_ = to;
    }
  }
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

MemoryManagerFreeList::MemoryManagerFreeList(DeviceContext *context,
                                             MemoryManager &memoryManager,
                                             MemoryManagerFreeList *parent,
                                             bool disableFlush,
                                             std::string label)
    : ctx_(context), memoryManager_(memoryManager), parent_(parent),
      disableFlush_(disableFlush), label_(std::move(label)) {
  // Either parent and context are both null for the free list associated with
  // the MemoryManager or both are set for free lists associated with a
  // DeviceContext.
  assert(((ctx_ == nullptr) && (parent_ == nullptr)) ||
         ((ctx_ != nullptr) && (parent_ != nullptr)));
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    freeBuffers_[i] = nullptr;
  }

  if (ctx_ && ctx_->isDebugAllocatorEnabled("out-of-bounds")) {
    if (context == nullptr) {
      memoryManager_.logErr("[" + label_ +
                            "] redzone checks disabled due to "
                            "null context in MemoryManagerFreeList");
    } else {
      enableRedzones();
    }
  }

  // `poison-all` is the allocator-layer counterpart to the graph-driver's
  // type-aware `uninitialized-poison` (see Device::createDeviceMemory): it has
  // no dtype information, so it fills *every* allocation body with a raw byte
  // pattern rather than a type-specific sentinel. See the MemoryManagerOverview
  // doc (`MODULAR_DEBUG_DEVICE_ALLOCATOR`) for how the two tiers relate.
  if (ctx_ && ctx_->isDebugAllocatorEnabled("poison-all")) {
    if (context == nullptr) {
      memoryManager_.logErr("[" + label_ +
                            "] poison fill disabled due to "
                            "null context in MemoryManagerFreeList");
    } else {
      enablePoison();
    }
  }

  // Register per-stream (child) free-lists so a VMM allocation miss can flush
  // sibling streams' completed frees to the root (see flushSiblingFreeLists).
  if (ctx_ != nullptr && parent_ != nullptr)
    memoryManager_.registerChildFreeList(this);
}

DeviceContext *MemoryManagerFreeList::context() { return ctx_; }

const DeviceContext *MemoryManagerFreeList::context() const { return ctx_; }

MemoryManagerFreeList::~MemoryManagerFreeList() {
  if (ctx_ != nullptr && parent_ != nullptr)
    memoryManager_.unregisterChildFreeList(this);
  flushToParent();
}

void MemoryManager::registerChildFreeList(MemoryManagerFreeList *fl) {
  std::lock_guard<std::mutex> lock(childFreeListsMu_);
  childFreeLists_.push_back(fl);
}

void MemoryManager::unregisterChildFreeList(MemoryManagerFreeList *fl) {
  std::lock_guard<std::mutex> lock(childFreeListsMu_);
  auto it = std::find(childFreeLists_.begin(), childFreeLists_.end(), fl);
  if (it != childFreeLists_.end())
    childFreeLists_.erase(it);
}

void MemoryManager::flushSiblingFreeLists(const DeviceContext *except) {
  // Poll pending events on every registered child context so their completed
  // frees migrate from per-stream free buffers to the root free-list (where
  // allocateDesc and defrag can use them). Safe: pollPendingEvents only runs
  // handlers whose device work has completed, and it releases its own lock
  // before running each handler, so there is no lock-order inversion with the
  // free() path (mu_ -> pendingWorkMutex_). The registry lock is held across
  // the loop so entries cannot be removed mid-iteration; this relies on child
  // contexts being stable during allocation (created at startup, destroyed at
  // shutdown), which holds for MAX Serve.
  std::lock_guard<std::mutex> lock(childFreeListsMu_);
  const DeviceContext *lastPolled = nullptr;
  for (MemoryManagerFreeList *fl : childFreeLists_) {
    DeviceContext *c = fl->ctx_;
    // Skip the caller's context (already polled) and cheaply de-dup the
    // adjacent freeList_/graphFreeList_ pair of the same context.
    if (c == nullptr || c == except || c == lastPolled)
      continue;
    lastPolled = c;
    (void)c->pollPendingEvents();
  }
}

void MemoryManagerFreeList::logMsg(llvm::Twine msg) const {
  memoryManager_.logErr("[" + label_ + "] " + msg);
}

void MemoryManagerFreeList::logMsg(llvm::Twine msg, size_t info,
                                   int64_t size) const {
  memoryManager_.logErr("[" + label_ + "] " + msg, info, size);
}

void MemoryManagerFreeList::flushToParent() {
  std::lock_guard<std::mutex> lock(memoryManager_.mu_);
  LOG_MSG("Flush all to parent");
  for (int i = 0; i < kNumFreeBufferLists; ++i) {
    flushFreeBuffersToParent(i);
  }
  // Reassign the parent freelist as the allocator-of-record for any descriptor
  // still in use that was originally allocated through this freelist. Without
  // this, `desc->allocatedBy_` becomes a dangling pointer once this freelist
  // is destroyed: a later `MemoryManagerFreeList::free` call on a sibling
  // freelist dispatches to `desc->allocatedBy_->freeAllocatedDesc(...)` and
  // dereferences freed memory. This is reachable today when a `DeviceBuffer`
  // is reassigned across `DeviceContext`s and its allocating context is torn
  // down before the buffer is freed (DRIV-193).
  if (parent_ != nullptr)
    memoryManager_.allocatedMap_.redirectAllocatedBy(this, parent_);
}

void MemoryManagerFreeList::flushFreeBuffersToParent(int id) {
  if (parent_ == nullptr)
    return;

  // Early return if no chunks have been allocated yet.
  if (memoryManager_.chunks_ == nullptr)
    return;

  LOG_MSG("flushing id: " + TWINE(id) + " to parent " + TWINE_PTR(parent_));
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

  LOG_MSG("flushing id: " + TWINE(id) + " to parent " + TWINE_PTR(parent_) +
              " : Done",
          0, bufSize);
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
  // list of free buffers is ordered non-descending by size.
  MemoryManager::Desc *&activeList = freeBuffers_[primaryFreeBuffers_];
  LOG_MSG("In addToFreeList using free buffer id: " +
          TWINE(primaryFreeBuffers_) + " address " + TWINE_PTR(desc->ptr_));
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
    memoryManager_.logErr("[" + label_ + "] !!! selfCheck !!! - " #cond);      \
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
          memoryManager_.logErr("[" + label_ +
                                "] !!! selfCheck !!! cycle - "
                                "free buffers");
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

MemoryManager::MemoryManager(Device &device, size_t maxSize, size_t chunkSize,
                             size_t alignment, bool isHost,
                             bool useOnlyMemoryManagerMode, bool vmmEnabled,
                             size_t vmmPageSize)
    : device_(device), isHost_(isHost),
      useOnlyMemoryManagerMode_(useOnlyMemoryManagerMode),
      vmmEnabled_(vmmEnabled), pageSize_(vmmPageSize), maxCacheSize_(maxSize),
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
  freeList_ = std::make_unique<MemoryManagerFreeList>(nullptr, *this, nullptr,
                                                      /*disableFlush=*/false,
                                                      "mm:root");

  // Determine state of debugging flags.
  doLog_ = DeviceContext::configAsBool("memory_manager_log", false);
  doSelfCheck_ =
      DeviceContext::configAsBool("memory_manager_self_check", false);
  // Print initial config of memory manager.
  LOG_ERR(llvm::Twine("Memory manager is ") +
          (this->isEnabled() ? "enabled" : "disabled"));
}

MemoryManager::~MemoryManager() {
  LOG_ERR("destructor");
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
  if (vmmEnabled_) {
    info.free = vmmAllocatableBytes();
  } else {
    info.free = maxCacheSize_ >= cacheSize_
                    ? (maxCacheSize_ - cacheSize_) + freeSize_
                    : freeSize_;
  }

  return info;
}

size_t MemoryManager::vmmAllocatableBytes() const {
  // The reserved VA exceeds the physical budget, so the amount still
  // allocatable is bounded by physical, not by free VA: the budget minus the
  // physical committed under in-use allocations. The rest of the budget is
  // either uncommitted or mapped-but-reclaimable (remappable on demand).
  const size_t inUse = inUsePhysicalBytes();
  return maxCacheSize_ >= inUse ? maxCacheSize_ - inUse : 0;
}

bool MemoryManager::isEnabled() const { return maxCacheSize_ > 0; }

bool MemoryManager::isVmmEnabled() const { return vmmEnabled_; }

size_t MemoryManager::allocatedSize(void *ptr) const {
  if (ptr == nullptr)
    return 0;
  std::lock_guard<std::mutex> lock(mu_);
  const Desc *desc = allocatedMap_.find(ptr);
  return desc != nullptr ? desc->size_.load() : size_t{0};
}

Error MemoryManager::allocationMissError(const MemoryManagerFreeList *freeList,
                                         size_t wrappedSize) const {
  size_t availableCacheSize = 0;
  size_t chunkSize = 0;
  size_t cacheSize = 0;
  size_t maxCacheSize = 0;
  size_t freeSize = 0;

  size_t freeCount = 0;
  size_t chunkCount = 0;

  std::string lastChunkAllocFailureDetail;

  {
    std::lock_guard<std::mutex> lock(mu_);
    availableCacheSize =
        maxCacheSize_ >= cacheSize_ ? maxCacheSize_ - cacheSize_ : 0;
    chunkSize = chunkSize_;
    cacheSize = cacheSize_;
    maxCacheSize = maxCacheSize_;
    freeSize = freeSize_;

    freeCount = freeCount_;
    chunkCount = chunkCount_;

    lastChunkAllocFailureDetail = lastChunkAllocFailureDetail_;
  }

  std::string missReason =
      "memory manager free lists could not satisfy request";
  if (chunkSize == 0) {
    missReason = "memory manager chunk size is zero";
  } else if (wrappedSize > chunkSize) {
    missReason = "request exceeds memory manager chunk size";
  } else if (availableCacheSize < wrappedSize) {
    missReason = "remaining memory manager capacity is insufficient";
  }

  freeList->logLargestBlocksInfo(wrappedSize, missReason);
  freeList->dumpFragmentationDiagnostic(wrappedSize, missReason);

  std::string message;
  llvm::raw_string_ostream os(message);
  os << "Memory manager cannot satisfy allocation (" << missReason
     << "): " << getDeviceDescription() << " on "
     << "[" << freeList->label_ << "]@" << TWINE_PTR(freeList)
     << " (request: " << humanMemorySize(wrappedSize)
     << " ; free: " << humanMemorySize(freeSize)
     << " ; free_blocks: " << freeCount
     << " ; available: " << humanMemorySize(availableCacheSize)
     << " ; chunk_size: " << humanMemorySize(chunkSize)
     << " ; chunk_count: " << chunkCount
     << " ; cache_size: " << humanMemorySize(cacheSize)
     << " ; max_cache_size: " << humanMemorySize(maxCacheSize);
  if (!lastChunkAllocFailureDetail.empty())
    os << " ; last_chunk_alloc_failure: " << lastChunkAllocFailureDetail;
  os << ")";
  return Error(os.str());
}

void MemoryManagerFreeList::logLargestBlocksInfo(size_t wrappedSize,
                                                 std::string missReason) const {
  if (memoryManager_.doLog_) {
    memoryManager_.logErr("allocate(" + TWINE(wrappedSize) + ") - reject (" +
                              missReason + ")",
                          0, 0);
    // Print more debugging information about the largest available blocks.
    const MemoryManagerFreeList *cur = this;
    while (cur != nullptr) {
      for (int i = 0; i < kNumFreeBufferLists; ++i) {
        if (cur->freeBuffers_[i] != nullptr) {
          cur->logMsg("max block id: " + TWINE(i) + ": (" +
                      TWINE_PTR(cur->freeBuffers_[i]->ptr_) + " ; " +
                      TWINE(cur->freeBuffers_[i]->size_) + ")");
        }
      }
      cur = cur->parent_;
    }
  }
}

void MemoryManagerFreeList::dumpFragmentationDiagnostic(
    size_t requestedSize, llvm::Twine reason) const {
  // Free list buffers_[i] is sorted non-descending by size: head is smallest,
  // tail is largest. We rely on this to capture the top-N largest blocks in
  // a single forward walk using a ring buffer of size N.
  static constexpr int kHistBuckets = 64;
  static constexpr int kTopN = 10;

  // Acquire a coherent snapshot of the memory manager state.
  std::lock_guard<std::mutex> lock(memoryManager_.mu_);

  auto logLine = [this](llvm::Twine msg) { memoryManager_.logErr(msg); };

  logLine("oom log ===== begin =====");
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "oom log request=" << humanMemorySize(requestedSize) << " reason=\""
       << reason.str() << "\"";
    logLine(os.str());
  }
  {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "oom log mm max_cache="
       << humanMemorySize(memoryManager_.maxCacheSize_)
       << " cache_used=" << humanMemorySize(memoryManager_.cacheSize_)
       << " free=" << humanMemorySize(memoryManager_.freeSize_)
       << " free_blocks=" << memoryManager_.freeCount_
       << " chunk_size=" << humanMemorySize(memoryManager_.chunkSize_)
       << " chunks=" << memoryManager_.chunkCount_;
    logLine(os.str());
  }

  // Walk the free list chain (this + parents).
  const MemoryManagerFreeList *cur = this;
  while (cur != nullptr) {
    {
      std::string s;
      llvm::raw_string_ostream os(s);
      os << "oom log freelist=[" << cur->label_ << "]@"
         << static_cast<const void *>(cur)
         << " primary=" << cur->primaryFreeBuffers_;
      logLine(os.str());
    }
    for (int i = 0; i < kNumFreeBufferLists; ++i) {
      MemoryManager::Desc *first = cur->freeBuffers_[i];
      if (first == nullptr) {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "oom log   bucket=" << i << " empty";
        logLine(os.str());
        continue;
      }

      uint64_t hist[kHistBuckets] = {0};
      size_t topRing[kTopN] = {0};
      size_t count = 0;
      size_t total = 0;
      size_t smallest = ~size_t{0};
      size_t largest = 0;
      for (MemoryManager::Desc *d = first; d != nullptr; d = d->nextFree_) {
        size_t sz = d->size_.load(std::memory_order_relaxed);
        ++count;
        total += sz;
        if (sz < smallest)
          smallest = sz;
        if (sz > largest)
          largest = sz;
        int bucket = (sz == 0) ? 0
                               : std::min(kHistBuckets - 1,
                                          static_cast<int>(llvm::Log2_64(sz)));
        ++hist[bucket];
        topRing[(count - 1) % kTopN] = sz;
      }

      {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "oom log   bucket=" << i << " count=" << count
           << " total=" << humanMemorySize(total)
           << " smallest=" << humanMemorySize(smallest)
           << " largest=" << humanMemorySize(largest);
        logLine(os.str());
      }
      {
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "oom log   bucket=" << i << " hist";
        for (int b = 0; b < kHistBuckets; ++b) {
          if (hist[b] > 0)
            os << " 2^" << b << "=" << hist[b];
        }
        logLine(os.str());
      }
      {
        // Ring buffer holds the last min(count, kTopN) sizes visited, which
        // are the largest because the list is sorted ascending. Output them
        // in descending order.
        std::string s;
        llvm::raw_string_ostream os(s);
        int topCount = static_cast<int>(std::min<size_t>(count, kTopN));
        os << "oom log   bucket=" << i << " top" << topCount << "=";
        for (int j = 0; j < topCount; ++j) {
          if (j > 0)
            os << ",";
          int idx = static_cast<int>((count - 1 - j + kTopN) % kTopN);
          os << humanMemorySize(topRing[idx]);
        }
        logLine(os.str());
      }
    }
    cur = cur->parent_;
  }

  // Per-chunk layout. The killer fragmentation signal: if largest_free per
  // chunk is tiny while free bytes are large, the chunk is shredded.
  int chunkIdx = 0;
  for (MemoryManager::Chunk *chunk = memoryManager_.chunks_.get();
       chunk != nullptr; chunk = chunk->next_.get(), ++chunkIdx) {
    size_t freeBlocks = 0;
    size_t inUseBlocks = 0;
    size_t freeBytes = 0;
    size_t inUseBytes = 0;
    size_t largestFree = 0;
    for (MemoryManager::Desc *d = chunk->descs_; d != nullptr; d = d->next_) {
      size_t sz = d->size_.load(std::memory_order_relaxed);
      if (d->isFree()) {
        ++freeBlocks;
        freeBytes += sz;
        if (sz > largestFree)
          largestFree = sz;
      } else {
        ++inUseBlocks;
        inUseBytes += sz;
      }
    }
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "oom log chunk=" << chunkIdx << " base=" << chunk->base_
       << " size=" << humanMemorySize(chunk->size_)
       << " in_use_blocks=" << inUseBlocks
       << " in_use=" << humanMemorySize(inUseBytes)
       << " free_blocks=" << freeBlocks
       << " free=" << humanMemorySize(freeBytes)
       << " largest_free=" << humanMemorySize(largestFree);
    logLine(os.str());
  }

  logLine("oom log ===== end =====");
}

bool MemoryManager::allocateVmmArena() {
  // Single lazily-backed VMM arena: reserve one contiguous VA range on first
  // use and map physical granules on demand (ensureRegionBacked). The VA
  // reservation is deliberately LARGER than the physical budget so a large
  // allocation that does not fit an existing free hole can always carve fresh
  // contiguous address space at the frontier; the physical it needs is
  // reclaimed from free-but-mapped granules elsewhere and remapped here (the
  // expandable-segments model). Physical stays capped at maxCacheSize_. Never
  // create a second arena.
  if (chunks_ != nullptr)
    return false;
  assert(pageSize_ != 0 && "VMM requires a non-zero granule size");
  const size_t budget = (maxCacheSize_ / pageSize_) * pageSize_;
  if (budget == 0)
    return false;
  const int64_t vaFactor = std::max<int64_t>(
      1, DeviceContext::configAsInt("memory_manager_vmm_va_factor", 4));
  size_t vaSize = budget * static_cast<size_t>(vaFactor);
  ErrorOr<void *> vaOr = device().vmmAddressReserve(vaSize);
  if (vaOr.isError()) {
    // Oversized reservation failed (e.g. VA exhaustion); fall back to a
    // budget-sized arena (loses frontier headroom but still functions).
    LOG_ERR(llvm::Twine("vmmAddressReserve(") + humanMemorySize(vaSize) +
            ") failed, retrying at budget size: " + vaOr.getError());
    vaSize = budget;
    vaOr = device().vmmAddressReserve(vaSize);
    if (vaOr.isError()) {
      lastChunkAllocFailureDetail_ = "vmmAddressReserve(" +
                                     humanMemorySize(vaSize) +
                                     ") failed with " + vaOr.getError();
      return false;
    }
  }
  const size_t numGranules = vaSize / pageSize_;
  auto chunk = std::make_unique<Chunk>(*this, *vaOr, vaSize);
  chunk->vmmBacked_ = true;
  chunk->pageSize_ = pageSize_;
  chunk->vmmHandles_.assign(numGranules, 0);
  chunk->granuleBacked_.resize(numGranules, false);
  chunk->next_ = std::move(chunks_);
  chunks_ = std::move(chunk);
  ++chunkCount_;
  // cacheSize_ tracks mapped physical, which is zero until granules are
  // mapped lazily; the VA reservation itself consumes no physical memory.
  LOG_ERR("vmm_arena_reserve(" + TWINE_PTR(*vaOr) + " ; va=" + TWINE(vaSize) +
              " ; budget=" + TWINE(budget) + ")",
          0, 0);
  return true;
}

bool MemoryManager::allocateChunk() {
  // VMM uses a single oversized arena with lazy physical backing, an entirely
  // separate implementation from the chunked allocator below.
  if (vmmEnabled_)
    return allocateVmmArena();

  if (shouldUseMaxSize()) {
    // Ensure that we have grown chunkSize_ to the maximum.
    growChunkSizeTo(maxCacheSize_);
  }

  // Cap the chunk to the remaining memory manager budget and, for GPU
  // devices, to how much free memory the device actually has right now.
  assert(cacheSize_ <= maxCacheSize_ && "cache size <= max size precondition");
  lastChunkAllocFailureDetail_.clear();
  size_t remainingBudget = maxCacheSize_ - cacheSize_;
  std::string currentDeviceFree = "n/a";
  std::string currentDeviceTotal = "n/a";
  if (!isHost()) {
    // Limit to available GPU memory.
    ErrorOr<Device::MemInfo> memInfo = device().getMemoryInfo();
    if (!memInfo.isError()) {
      remainingBudget = std::min((*memInfo).free, remainingBudget);
      currentDeviceFree = humanMemorySize((*memInfo).free);
      currentDeviceTotal = humanMemorySize((*memInfo).total);
    }
  }

  size_t allocSize = std::min(chunkSize_, remainingBudget);

  const size_t minChunkSize = std::min(128UL * 1024, (chunkSize_ * 25) / 100);
  if (allocSize < minChunkSize) {
    llvm::raw_string_ostream os(lastChunkAllocFailureDetail_);
    os << "alloc_size " << humanMemorySize(allocSize)
       << " below min_chunk_size " << humanMemorySize(minChunkSize)
       << " (remaining_budget: " << humanMemorySize(remainingBudget)
       << " ; device_free: " << currentDeviceFree
       << " ; device_total: " << currentDeviceTotal << ")";
    return false;
  }

  void *ptr = nullptr;
  std::vector<uint64_t> handles;
  size_t chunkPageSize = 0;
  if (!acquireChunkBacking(allocSize, &ptr, &handles, &chunkPageSize)) {
    // acquireChunkBacking records lastChunkAllocFailureDetail_ and frees any
    // partial backing on failure.
    return false;
  }
  auto chunk = std::make_unique<Chunk>(*this, ptr, allocSize);
  chunk->next_ = std::move(chunks_);
  chunks_ = std::move(chunk);

  cacheSize_ += allocSize;
  ++chunkCount_;
  LOG_ERR("allocate_chunk(" + TWINE_PTR(ptr) + " ; " + TWINE(allocSize) + ")",
          0, 0);
  return true;
}

// Acquire physical+virtual backing for a chunk. Plain path delegates to
// chunkAlloc; VMM path reserves contiguous VA and maps `allocSize/pageSize_`
// granules (reusing the free-granule pool before creating new physical memory).
bool MemoryManager::acquireChunkBacking(size_t allocSize, void **base,
                                        std::vector<uint64_t> *handles,
                                        size_t *pageSize) {
  *pageSize = 0;
  if (!vmmEnabled_) {
    ErrorOrSuccess err = device().chunkAlloc(base, allocSize, isHost());
    if (err.isError()) {
      llvm::raw_string_ostream os(lastChunkAllocFailureDetail_);
      os << "chunkAlloc(" << humanMemorySize(allocSize) << ") failed with "
         << err.getError();
      LOG_ERR("allocate_chunk(" + TWINE(allocSize) + ") - failed with " +
              err.getError());
      return false;
    }
    return true;
  }

  assert(pageSize_ != 0 && (allocSize % pageSize_) == 0 &&
         "VMM allocSize must be a whole number of granules");
  *pageSize = pageSize_;
  const size_t numPages = allocSize / pageSize_;

  ErrorOr<void *> vaOr = device().vmmAddressReserve(allocSize);
  if (vaOr.isError()) {
    lastChunkAllocFailureDetail_ = "vmmAddressReserve(" +
                                   humanMemorySize(allocSize) +
                                   ") failed with " + vaOr.getError();
    return false;
  }
  void *va = *vaOr;

  std::vector<uint64_t> acquired;
  acquired.reserve(numPages);
  // Roll back: unmap mapped pages, return their (still-valid) handles to the
  // pool, free the VA reservation. Records the failure reason.
  auto rollback = [&](const std::string &why) {
    for (size_t i = 0; i < acquired.size(); ++i) {
      void *pageVa = static_cast<char *>(va) + i * pageSize_;
      ErrorOrSuccess e = device().vmmUnmap(pageVa, pageSize_);
      if (e.isError())
        LOG_ERR(llvm::Twine("vmmUnmap during rollback failed: ") +
                e.getError());
      freeGranulePool_.push_back(acquired[i]);
    }
    ErrorOrSuccess e = device().vmmAddressFree(va, allocSize);
    if (e.isError())
      LOG_ERR(llvm::Twine("vmmAddressFree during rollback failed: ") +
              e.getError());
    lastChunkAllocFailureDetail_ = why;
  };

  for (size_t i = 0; i < numPages; ++i) {
    uint64_t handle = 0;
    if (!freeGranulePool_.empty()) {
      handle = freeGranulePool_.back();
      freeGranulePool_.pop_back();
    } else {
      ErrorOr<uint64_t> h = device().vmmCreate(pageSize_);
      if (h.isError()) {
        rollback("vmmCreate(" + humanMemorySize(pageSize_) + ") failed with " +
                 h.getError());
        return false;
      }
      handle = *h;
    }
    void *pageVa = static_cast<char *>(va) + i * pageSize_;
    ErrorOrSuccess mapErr =
        device().vmmMapAndSetAccess(pageVa, pageSize_, handle);
    if (mapErr.isError()) {
      // The handle is valid but unmapped; keep it in the pool for reuse.
      freeGranulePool_.push_back(handle);
      rollback(std::string("vmmMapAndSetAccess failed with ") +
               mapErr.getError());
      return false;
    }
    acquired.push_back(handle);
  }

  *base = va;
  *handles = std::move(acquired);
  return true;
}

// Tear down a chunk's backing. VMM chunks unmap each granule and either release
// it to the driver or return it to the free-granule pool for reuse, then free
// the VA reservation. Plain chunks call chunkFree.
void MemoryManager::releaseChunkBacking(Chunk *chunk, bool releaseToDriver) {
  if (!chunk->vmmBacked_) {
    ErrorOrSuccess err = device().chunkFree(chunk->base_, isHost());
    if (err.isError()) {
      logErr(err.getError());
      abort();
    }
    return;
  }
  const size_t ps = chunk->pageSize_;
  for (size_t i = 0; i < chunk->vmmHandles_.size(); ++i) {
    // Lazily-unbacked granule (single-arena VMM): no physical mapped, nothing
    // to unmap or release.
    if (i < chunk->granuleBacked_.size() && !chunk->granuleBacked_[i])
      continue;
    void *pageVa = static_cast<char *>(chunk->base_) + i * ps;
    ErrorOrSuccess e = device().vmmUnmap(pageVa, ps);
    if (e.isError())
      LOG_ERR(llvm::Twine("vmmUnmap failed: ") + e.getError());
    if (releaseToDriver) {
      ErrorOrSuccess r = device().vmmRelease(chunk->vmmHandles_[i]);
      if (r.isError())
        LOG_ERR(llvm::Twine("vmmRelease failed: ") + r.getError());
    } else {
      freeGranulePool_.push_back(chunk->vmmHandles_[i]);
    }
  }
  chunk->vmmHandles_.clear();
  chunk->granuleBacked_.clear();
  ErrorOrSuccess e = device().vmmAddressFree(chunk->base_, chunk->size_);
  if (e.isError())
    LOG_ERR(llvm::Twine("vmmAddressFree failed: ") + e.getError());
}

bool MemoryManager::descHasUnbackedRegion(const Desc *desc) const {
  const Chunk *c = desc->chunk_;
  if (!c->vmmBacked_ || c->granuleBacked_.empty())
    return false;
  // Map the desc's byte range to the inclusive granule index range it spans,
  // then ask the bitset whether any granule in [first, end) is still unmapped.
  const char *base = static_cast<char *>(c->base_);
  const size_t ps = c->pageSize_;
  const size_t first = (static_cast<char *>(desc->ptr_) - base) / ps;
  const size_t last =
      (static_cast<char *>(desc->ptr_) + desc->size_ - 1 - base) / ps;
  const size_t end = std::min<size_t>(last + 1, c->granuleBacked_.size());
  if (first >= end)
    return false;
  return c->granuleBacked_.find_first_unset_in(first, end) != -1;
}

void MemoryManager::computeInUseGranules(const Chunk *c,
                                         llvm::BitVector &out) const {
  const size_t n = c->vmmBacked_ ? c->vmmHandles_.size() : 0;
  out.clear();
  out.resize(n, false);
  if (n == 0)
    return;
  const char *base = static_cast<char *>(c->base_);
  const size_t ps = c->pageSize_;
  for (const Desc *d = c->descs_; d != nullptr; d = d->next_) {
    if (d->isFree())
      continue;
    const size_t first = (static_cast<char *>(d->ptr_) - base) / ps;
    const size_t last =
        (static_cast<char *>(d->ptr_) + d->size_ - 1 - base) / ps;
    const size_t end = std::min(last + 1, n);
    if (first < end)
      out.set(first, end);
  }
}

void MemoryManager::computeUnreclaimableGranules(const Chunk *c,
                                                 llvm::BitVector &out) const {
  const size_t n = c->vmmBacked_ ? c->vmmHandles_.size() : 0;
  out.clear();
  out.resize(n, false);
  if (n == 0)
    return;
  const char *base = static_cast<char *>(c->base_);
  const size_t ps = c->pageSize_;
  for (const Desc *d = c->descs_; d != nullptr; d = d->next_) {
    // A granule is safe to reclaim only when every desc covering it is a
    // root-owned free desc. A desc that is in-use, or free but still owned by a
    // per-stream (child) free-list, may have in-flight GPU work referencing its
    // memory (a per-stream free only migrates to the root free-list once its
    // stream events fire), so unmapping it could fault a live kernel.
    if (d->isFree() && d->owner() == freeList_.get())
      continue;
    const size_t first = (static_cast<char *>(d->ptr_) - base) / ps;
    const size_t last =
        (static_cast<char *>(d->ptr_) + d->size_ - 1 - base) / ps;
    const size_t end = std::min(last + 1, n);
    if (first < end)
      out.set(first, end);
  }
}

size_t MemoryManager::inUsePhysicalBytes() const {
  if (chunks_ == nullptr || !chunks_->vmmBacked_)
    return 0;
  llvm::BitVector inUse;
  computeInUseGranules(chunks_.get(), inUse);
  return inUse.count() * chunks_->pageSize_;
}

size_t MemoryManager::reclaimMappedGranules(Chunk *c, size_t count) {
  if (count == 0 || !c->vmmBacked_)
    return 0;
  llvm::BitVector unreclaimable;
  computeUnreclaimableGranules(c, unreclaimable);
  const size_t ps = c->pageSize_;
  char *base = static_cast<char *>(c->base_);
  size_t reclaimed = 0;
  for (size_t i = 0; i < c->granuleBacked_.size() && reclaimed < count; ++i) {
    // Reclaim only granules that are physically mapped AND safe to unmap (not
    // overlapped by an in-use desc or a non-root-owned free desc whose memory
    // may still be referenced by in-flight GPU work).
    if (!c->granuleBacked_[i] || unreclaimable[i])
      continue;
    void *pageVa = base + i * ps;
    ErrorOrSuccess e = device().vmmUnmap(pageVa, ps);
    if (e.isError()) {
      LOG_ERR(llvm::Twine("reclaimMappedGranules vmmUnmap failed: ") +
              e.getError());
      continue;
    }
    freeGranulePool_.push_back(c->vmmHandles_[i]);
    c->vmmHandles_[i] = 0;
    c->granuleBacked_.reset(i);
    cacheSize_ -= ps;
    ++reclaimed;
  }
  return reclaimed;
}

ErrorOrSuccess MemoryManager::ensureRegionBacked(Desc *desc) {
  Chunk *c = desc->chunk_;
  if (!c->vmmBacked_ || c->granuleBacked_.empty())
    return success();
  char *base = static_cast<char *>(c->base_);
  const size_t ps = c->pageSize_;
  const size_t first = (static_cast<char *>(desc->ptr_) - base) / ps;
  const size_t last =
      (static_cast<char *>(desc->ptr_) + desc->size_ - 1 - base) / ps;

  // Count granules in this region that still need physical backing. Callers
  // gate on descHasUnbackedRegion(), so this is normally non-zero; the early
  // return just keeps the function safe if called directly. (Mapping during
  // graph capture is illegal and will fail below with the driver's native
  // vmmMap error -- no explicit capture check is needed.)
  size_t needMap = 0;
  for (size_t i = first; i <= last && i < c->granuleBacked_.size(); ++i)
    if (!c->granuleBacked_[i])
      ++needMap;
  if (needMap == 0)
    return success();

  // The VA arena is larger than the physical budget, so backing these granules
  // may require reclaiming physical from free-but-mapped granules elsewhere.
  // The desc is already in use, so computeUnreclaimableGranules excludes this
  // region (reclaim never touches the granules we are about to back).
  const size_t poolAvail = freeGranulePool_.size();
  // Physical already created (mapped granules + pooled unmapped handles) counts
  // against the budget; new physical may only be created up to the remainder.
  const size_t createdPhysical = cacheSize_ + poolAvail * ps;
  const size_t canCreate = maxCacheSize_ > createdPhysical
                               ? (maxCacheSize_ - createdPhysical) / ps
                               : 0;
  if (needMap > poolAvail + canCreate) {
    const size_t toReclaim = needMap - poolAvail - canCreate;
    const size_t got = reclaimMappedGranules(c, toReclaim);
    if (got < toReclaim)
      return Error("ensureRegionBacked: out of physical budget (need " +
                   humanMemorySize(needMap * ps) + ", reclaimed " +
                   humanMemorySize(got * ps) + ")");
  }

  // Granules mapped in this call, for rollback on a later partial failure.
  llvm::SmallVector<size_t, 8> mappedNow;
  auto rollback = [&]() {
    for (size_t idx : mappedNow) {
      void *pageVa = base + idx * ps;
      ErrorOrSuccess ue = device().vmmUnmap(pageVa, ps);
      if (ue.isError())
        LOG_ERR(llvm::Twine("ensureRegionBacked rollback vmmUnmap failed: ") +
                ue.getError());
      freeGranulePool_.push_back(c->vmmHandles_[idx]);
      c->vmmHandles_[idx] = 0;
      c->granuleBacked_.reset(idx);
      cacheSize_ -= ps;
    }
  };

  for (size_t i = first; i <= last && i < c->granuleBacked_.size(); ++i) {
    if (c->granuleBacked_[i])
      continue;
    // Physical is guaranteed available now: pool (incl. just-reclaimed handles)
    // first, else create within the remaining budget.
    uint64_t handle = 0;
    if (!freeGranulePool_.empty()) {
      handle = freeGranulePool_.back();
      freeGranulePool_.pop_back();
    } else {
      ErrorOr<uint64_t> h = device().vmmCreate(ps);
      if (h.isError()) {
        rollback();
        return Error(std::string("ensureRegionBacked: vmmCreate failed: ") +
                     h.getError());
      }
      handle = *h;
    }
    void *pageVa = base + i * ps;
    ErrorOrSuccess me = device().vmmMapAndSetAccess(pageVa, ps, handle);
    if (me.isError()) {
      freeGranulePool_.push_back(handle); // valid handle, just unmapped
      rollback();
      return Error(
          std::string("ensureRegionBacked: vmmMapAndSetAccess failed: ") +
          me.getError());
    }
    c->vmmHandles_[i] = handle;
    c->granuleBacked_.set(i);
    cacheSize_ += ps;
    mappedNow.push_back(i);
  }
  return success();
}

ErrorOrSuccess MemoryManager::preBackForCapture() {
  if (!vmmEnabled_)
    return success();
  std::lock_guard<std::mutex> lock(mu_);
  if (chunks_ == nullptr || !chunks_->vmmBacked_)
    return success();
  Chunk *c = chunks_.get();
  const size_t ps = c->pageSize_;
  char *base = static_cast<char *>(c->base_);
  // Map free-but-unbacked granules (low address first, where allocations land)
  // until the physical budget is fully committed. In-use granules are already
  // backed (skipped). Running out of budget to commit is a normal stop, but a
  // genuine driver failure (create/map) is propagated so capture setup fails
  // fast with the real cause, rather than proceeding into a guaranteed -- and
  // more cryptic -- mid-capture mapping failure.
  for (size_t i = 0; i < c->granuleBacked_.size(); ++i) {
    if (cacheSize_ >= maxCacheSize_)
      break;
    if (c->granuleBacked_[i])
      continue;
    uint64_t handle = 0;
    if (!freeGranulePool_.empty()) {
      handle = freeGranulePool_.back();
      freeGranulePool_.pop_back();
    } else {
      // Respect the budget: created physical == mapped + pooled handles.
      if (cacheSize_ + freeGranulePool_.size() * ps + ps > maxCacheSize_)
        break;
      ErrorOr<uint64_t> h = device().vmmCreate(ps);
      if (h.isError())
        return Error(std::string("preBackForCapture vmmCreate failed: ") +
                     h.getError());
      handle = *h;
    }
    void *pageVa = base + i * ps;
    ErrorOrSuccess me = device().vmmMapAndSetAccess(pageVa, ps, handle);
    if (me.isError()) {
      freeGranulePool_.push_back(handle);
      return Error(
          std::string("preBackForCapture vmmMapAndSetAccess failed: ") +
          me.getError());
    }
    c->vmmHandles_[i] = handle;
    c->granuleBacked_.set(i);
    cacheSize_ += ps;
  }
  return success();
}

Device &MemoryManager::device() const { return device_; }

void MemoryManager::selfCheck() const {
  if (!doSelfCheck_ || chunks_ == nullptr) {
    return;
  }

  size_t mappedBytes = 0;
  Chunk *curChunk = chunks_.get();
  while (curChunk != nullptr) {
    curChunk->selfCheck();
    if (curChunk->vmmBacked_) {
      // Single-arena VMM accounting: every slot's mapped state must agree with
      // its handle (a mapped slot holds a non-zero physical handle; an unmapped
      // slot holds 0), and the running mapped-physical counter `cacheSize_`
      // must equal the number of backed slots times the granule size. Catches
      // any path that updates granuleBacked_/vmmHandles_/cacheSize_
      // inconsistently.
      const size_t ps = curChunk->pageSize_;
      size_t backed = 0;
      for (size_t i = 0; i < curChunk->granuleBacked_.size(); ++i) {
        const bool isBacked = curChunk->granuleBacked_[i];
        const bool hasHandle = curChunk->vmmHandles_[i] != 0;
        if (isBacked != hasHandle) {
          logErr("selfCheck failed: VMM slot " + TWINE(i) +
                 " backed-bit/handle mismatch (handle=" +
                 TWINE(curChunk->vmmHandles_[i]) + ")");
          abort();
        }
        if (isBacked)
          ++backed;
      }
      mappedBytes += backed * ps;
    }
    curChunk = curChunk->next_.get();
  }
  if (vmmEnabled_ && mappedBytes != cacheSize_) {
    logErr("selfCheck failed: cacheSize_ (" + TWINE(cacheSize_) +
           ") does not match mapped VMM physical (" + TWINE(mappedBytes) + ")");
    abort();
  }

  LOG_ERR("selfCheck OK");
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

  std::unique_ptr<Chunk> chunk = std::move(chunks_);
  while (chunk != nullptr) {
    std::unique_ptr<Chunk> nextChunk = std::move(chunk->next_);
    Desc *desc = chunk->descs_;
    // Ensure each chunk only contains one free desc.
    assert(desc->ptr_ == chunk->base_);
    assert(desc->size_ == chunk->size_);
    assert(desc->prev_ == nullptr);
    assert(desc->next_ == nullptr);
    assert(desc->isFree());

    freeList_->removeFromFreeList(desc);
    delete desc;
    // Release the chunk memory (VMM-aware: unmaps granules, releases handles,
    // frees the VA reservation; plain chunks call chunkFree).
    LOG_ERR("free_chunk(" + TWINE_PTR(chunk->base_) + ", " +
            TWINE(chunk->size_) + ")");
    releaseChunkBacking(chunk.get(), /*releaseToDriver=*/true);

    chunk = std::move(nextChunk);
  }
  chunks_ = nullptr;
  // All physical backing has been released; reset mapped-physical accounting so
  // a reused manager (re-creating the arena) starts from a clean state.
  cacheSize_ = 0;
  chunkCount_ = 0;

  // Release any granules sitting in the free-granule pool back to the driver.
  for (uint64_t handle : freeGranulePool_) {
    ErrorOrSuccess err = device().vmmRelease(handle);
    if (err.isError()) {
      logErr(err.getError());
    }
  }
  freeGranulePool_.clear();
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

void MemoryManagerFreeList::enableRedzones() {
  redzoneEnable_ = true;
  size_t alignment = memoryManager_.allocationAlignment();
  redzoneSize_ =
      DeviceContext::configAsInt("memory_manager_redzone_size", alignment);
  size_t remainder = redzoneSize_ % alignment;
  if (remainder != 0) {
    redzoneSize_ += (alignment - remainder);
    LOG_MSG("memory_manager_redzone_size must be a multiple of " +
            Twine(alignment) + " rounding up to " + Twine(redzoneSize_));
  }
  redzonePattern_ = static_cast<uint8_t>(
      DeviceContext::configAsInt("memory_manager_redzone_pattern", 0xBA));
  LOG_MSG("redzone checks enabled");
}

void MemoryManagerFreeList::enablePoison() {
  poisonEnable_ = true;
  // Default 0xFF: as fp32/bf16 this is a (negative) NaN, so an uninitialized
  // read propagates NaN into the output and trips the differential test. This
  // is the opposite choice from the graph-driver's `uninitialized-poison`,
  // which deliberately uses a *non-NaN* sentinel to avoid conflicting with the
  // nan-check pass; the cost here is that the NaN can also surface on
  // legitimately-uninitialized allocation padding, so `poison-all` is a manual,
  // opt-in aid rather than a CI default.
  poisonPattern_ = static_cast<uint8_t>(
      DeviceContext::configAsInt("memory_manager_poison_pattern", 0xFF));
  // Poison is independent of redzones: it only fills the allocation body on
  // alloc, with no padding and no per-free validation (cheap). Combine with the
  // `out-of-bounds` mode (`debug.device_allocator=poison-all,out-of-bounds`) to
  // also get OOB-write detection.
  LOG_MSG("poison fill enabled (pattern 0x" + Twine::utohexstr(poisonPattern_) +
          ")");
}

ErrorOr<void *> MemoryManagerFreeList::finalizeDebugAlloc(void *unwrappedPtr,
                                                          size_t size) {
  // With redzones the block is [underflow | body | overflow] and the body
  // starts `redzoneSize_` in; without redzones the body starts at the base.
  // Redzones and poison are independent and compose.
  uint8_t *base = static_cast<uint8_t *>(unwrappedPtr);
  uint8_t *wrappedPtr = base + (redzoneEnable_ ? redzoneSize_ : 0);
  uint8_t *overflowRedzone = wrappedPtr + size;

  if (redzoneEnable_)
    LOG_MSG("redzone(address 0x" + Twine::utohexstr((uintptr_t)wrappedPtr) +
            " ; size: " + Twine(redzoneSize_) +
            " ; pattern: " + Twine::utohexstr(redzonePattern_) +
            " ; underflow: 0x" + Twine::utohexstr((uintptr_t)base) +
            " ; overflow: 0x" + Twine::utohexstr((uintptr_t)overflowRedzone));

  if (memoryManager_.isHost()) {
    if (redzoneEnable_) {
      std::memset(base, redzonePattern_, redzoneSize_);
      std::memset(overflowRedzone, redzonePattern_, redzoneSize_);
    }
    // Poison the body so uninitialized reads surface as NaN downstream.
    if (poisonEnable_)
      std::memset(wrappedPtr, poisonPattern_, size);
  } else {
    if (redzoneEnable_) {
      if (auto result =
              ctx_->enqueueSetMemory(base, redzoneSize_, redzonePattern_, 1);
          result.isError())
        return Error("failed writing underflow redzone pattern: " +
                     Twine(result.getError()));
      if (auto result = ctx_->enqueueSetMemory(overflowRedzone, redzoneSize_,
                                               redzonePattern_, 1);
          result.isError())
        return Error("failed writing overflow redzone pattern: " +
                     Twine(result.getError()));
    }
    // Poison the body so uninitialized reads surface as NaN downstream.
    if (poisonEnable_)
      if (auto result =
              ctx_->enqueueSetMemory(wrappedPtr, size, poisonPattern_, 1);
          result.isError())
        return Error("failed writing poison pattern: " +
                     Twine(result.getError()));

    if (auto result = ctx_->synchronize(); result.isError())
      return Error("failed writing debug-allocator pattern: " +
                   Twine(result.getError()));
  }

  // Only redzones are validated at free, so only track redzoned allocations.
  if (redzoneEnable_) {
    // Store stack trace to print when an out-of-bounds access is detected.
    std::string stackTrace;
    llvm::raw_string_ostream os(stackTrace);
    llvm::sys::PrintStackTrace(os);
    redzoneDeviceAllocSizeMap_[wrappedPtr] = {std::move(stackTrace), size};
  }

  return wrappedPtr;
}

ErrorOr<void *> MemoryManagerFreeList::allocate(size_t size,
                                                Device::AllocMode mode) {
  assert(size != 0);
  TRACY_ZONE_SCOPED_NCT("MemoryManagerFreeList::allocate", TRACY_COLOR_PURPLE,
                        "size: " + std::to_string(size));

  size_t wrappedSize = size;
  if (redzoneEnable_) {
    // Add redzone padding to the allocation size
    wrappedSize += redzoneSize_ * 2;
  }

  if (memoryManager_.isEnabled() && memoryManager_.chunkSize_ != 0) {
    wrappedSize = (wrappedSize + memoryManager_.alignmentOffset_) &
                  memoryManager_.alignmentMask_;

    MemoryManager::Desc *desc = nullptr;
    {
      std::lock_guard<std::mutex> lock(memoryManager_.mu_);
      // The chunk-growth heuristic below is for the non-VMM allocator only.
      // The single-arena VMM path reserves one oversized arena up front
      // (allocateVmmArena) with lazily-backed pages, so there is no per-chunk
      // growth to manage and large requests are served from the arena frontier.
      if (wrappedSize > memoryManager_.chunkSize_ &&
          !memoryManager_.vmmEnabled_) {
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
          if (wrappedSize <= memoryManager_.maxCacheSize_) {
            assert(memoryManager_.cacheSize_ == 0);
            MemoryManager::enableUseMaxSize();
          } else {
            LOG_MSG("allocate(" + TWINE(wrappedSize) +
                    ") - skip growChunkSizeTo()");
          }
        }

        if (MemoryManager::shouldUseMaxSize()) {
          // Heuristic: The code above or another memory manager has determined
          // that we should grow to the largest possible chunk size. Update
          // the chunk size to all remaining memory for future chunk
          // allocations.
          memoryManager_.growChunkSizeTo(memoryManager_.maxCacheSize_);
        } else {
          memoryManager_.growChunkSizeTo(
              std::min(memoryManager_.maxCacheSize_, wrappedSize));
        }
      }

      if (wrappedSize <= memoryManager_.chunkSize_) {
        selfCheck();
        desc = allocateDesc(wrappedSize);
      }
    }
    if (desc != nullptr) {
      // Allocation from memory manager was successful, return the value.
      if (redzoneEnable_ || poisonEnable_) {
        return finalizeDebugAlloc(desc->ptr_, size);
      }
      return desc->ptr_;
    }

    // On a miss under VMM, this stream's already-completed frees may still be
    // sitting in its per-stream free buffers: a freed block only migrates to
    // the root free-list when its enqueued event handler fires. Poll pending
    // events to fire those handlers -- safe because pollPendingEvents only runs
    // handlers whose device work has completed -- then retry. This both lets a
    // plain retry succeed and, by moving frees to the root free-list, gives the
    // arena's reclaim path more root-owned (safe-to-remap) memory. Must run
    // OUTSIDE the memory-manager lock (the flush handlers take it). Gated on
    // VMM so the default allocator path is unchanged.
    if (memoryManager_.vmmEnabled_ && ctx_ != nullptr &&
        !ctx_->isGraphCapturing()) {
      ErrorOrSuccess polled = ctx_->pollPendingEvents();
      if (!polled.isError() && wrappedSize <= memoryManager_.chunkSize_) {
        std::lock_guard<std::mutex> lock(memoryManager_.mu_);
        desc = allocateDesc(wrappedSize);
      }
      // Still nothing: the free memory may be stranded in *sibling* streams'
      // per-stream free buffers. Poll every child context to flush their
      // completed frees to the root, then retry. Flushing polls pending events,
      // which is a synchronizing operation; we always do it here (so behavior
      // is unchanged), but when logging is enabled we record that it happened
      // so we can gauge whether it is a performance problem.
      if (desc == nullptr) {
        LOG_MSG("alloc miss: flushing sibling free-lists (size " +
                TWINE(wrappedSize) + ")");
        memoryManager_.flushSiblingFreeLists(ctx_);
        if (wrappedSize <= memoryManager_.chunkSize_) {
          std::lock_guard<std::mutex> lock(memoryManager_.mu_);
          desc = allocateDesc(wrappedSize);
        }
      }
      if (desc != nullptr) {
        if (redzoneEnable_ || poisonEnable_) {
          return finalizeDebugAlloc(desc->ptr_, size);
        }
        return desc->ptr_;
      }
    }

    if (memoryManager_.useOnlyMemoryManagerMode_)
      return memoryManager_.allocationMissError(this, wrappedSize);

    // else fall through to allocating from the device driver.
    // This is a "backdoor"; we may exceed memory manager's budget as a result.
  }

  // Memory manager is disabled, or it is enabled but we allow backdoor
  // allocations from the device driver.
  if (memoryManager_.useOnlyMemoryManagerMode_ ||
      (ctx_ && ctx_->isGraphCapturing())) {
    // TODO: Eliminate this contradiction at source. We reach this point when
    // MemMgr is disabled. So what does it mean to "use only memory manager
    // mode"? Currently, this surgery is tricky because MemMgr is constructed
    // inside Device CTOR instead of being preconstructed in a factory function
    // (where we can raise Error) and passed in as a parameter.
    // MLRT/unittests/Driver:GPUMemoryManagerOnlyTest.RejectsWhenGpuMemoryManagerIsDisabled
    // verifies this combination so it too will need to be updated if this
    // changes.

    dumpFragmentationDiagnostic(
        wrappedSize, memoryManager_.useOnlyMemoryManagerMode_
                         ? "use-only-memory-manager-mode with mm disabled"
                         : "graph capturing in progress, no driver fallback");
    return Error(
        "[Use only memory manager mode]: Memory manager cannot satisfy "
        "allocation (memory manager is disabled): " +
        memoryManager_.getDeviceDescription() + " on " + TWINE_PTR(this) +
        " (size: " + humanMemorySize(wrappedSize) + ")");
  }

  // Allocate straight from device driver if cache was not able to fulfill
  // the request. If the cache is disabled, then we pass on the allocation
  // mode.
  void *ptr = nullptr;
  if (ctx_ == nullptr) {
    // In the case of the bottom most freelist being consulted for allocation
    // we might not have a DeviceContext associated, so we fail the allocation
    // with an error.
    LOG_MSG("allocate_buffer(" + TWINE(size) +
            ") - failed due to missing context");
    return Error("memAlloc failed due to missing context");
  }
  ErrorOrSuccess err = ctx_->memAlloc(&ptr, wrappedSize, mode);

  if (err.isError()) {
    LOG_MSG("allocate_buffer(" + TWINE(size) + ") - failed with " +
            err.getError());
    // The memory manager could not satisfy the request and the device driver
    // also failed: this is a true OOM. Dump fragmentation state so we can tell
    // a fragmented-pool OOM apart from a hard out-of-budget OOM.
    dumpFragmentationDiagnostic(wrappedSize, "device driver memAlloc failed: " +
                                                 llvm::Twine(err.getError()));
    return err.takeError();
  }

  LOG_MSG("allocate_buffer(" + TWINE_PTR(ptr) + " ; " + TWINE(size) + ")", 0,
          0);
  assert(ptr != nullptr);
  if (redzoneEnable_ || poisonEnable_) {
    // body starts immediately after the underflow redzone (if redzones are on)
    return finalizeDebugAlloc(ptr, size);
  }
  return ptr;
}

MemoryManager::Desc *MemoryManagerFreeList::allocateDesc(size_t size) {
  TRACY_ZONE_SCOPED_NCT("MemoryManagerFreeList::allocateDesc",
                        TRACY_COLOR_PURPLE, "size: " + std::to_string(size));
  assert((size & memoryManager_.alignmentMask_) == size);
  LOG_MSG("In allocateDesc with size " + TWINE(size) +
          " and primary free buffer id: " + TWINE(primaryFreeBuffers_));
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

      // Single-arena VMM: physically back the chosen region before handing it
      // out (lazy mapping) if any granule it spans is still unmapped. On
      // failure (driver error -- including an illegal map attempted during
      // graph capture) return the block to the free list and report a miss so
      // the caller can fall through to its recovery/error path.
      if (memoryManager_.vmmEnabled_ &&
          memoryManager_.descHasUnbackedRegion(desc)) {
        ErrorOrSuccess rb = memoryManager_.ensureRegionBacked(desc);
        if (rb.isError()) {
          LOG_MSG(llvm::Twine("allocateDesc ensureRegionBacked failed: ") +
                  rb.getError());
          freeDesc(desc);
          return nullptr;
        }
      }

      memoryManager_.allocatedMap_.add(desc);
      desc->allocatedBy_ = this;

      LOG_MSG("allocate from free buffer at id: " + TWINE(freeListIdToUse) +
                  " (" + TWINE_PTR(desc->ptr_) + " ; " + TWINE(size) + ")",
              iterations, -size);

      return desc;
    }
  }

  // Delegate to parent freelist if available.
  if (parent_ != nullptr) {
    MemoryManager::Desc *desc = parent_->allocateDesc(size);
    if (desc != nullptr)
      desc->allocatedBy_ = this;
    return desc;
  }

  // We only ever allocate chunks in the bottom most free list which is
  // associated with the MemoryManager directly.
  assert(parent_ == nullptr);
  if (!memoryManager_.allocateChunk()) {
    LOG_MSG("allocate(" + TWINE(size) + ") - nullptr (" + TWINE(iterations) +
            ")");
    // No success allocating a new Chunk, so we bail from allocating in the
    // cache. Instead we will try to allocate directly from the device and if
    // that fails, then we propagate the failure reported by the device.
    return nullptr;
  }
  // Allocated a new chunk, try again.
  return allocateDesc(size);
}

ErrorOrSuccess
MemoryManagerFreeList::validateRedzones(void *wrappedPtr,
                                        const RedzoneInfo &info) {
  uint8_t *underflowRedzonePtr = static_cast<uint8_t *>(wrappedPtr);
  uint8_t *overflowRedzonePtr = underflowRedzonePtr + redzoneSize_ + info.size_;

  uint8_t *underflowRedzone = underflowRedzonePtr;
  uint8_t *overflowRedzone = overflowRedzonePtr;

  std::vector<uint8_t> underflowRedzoneStorage;
  std::vector<uint8_t> overflowRedzoneStorage;
  if (!memoryManager_.isHost()) {
    // Copy the redzone data from device back to host memory
    underflowRedzoneStorage.resize(redzoneSize_);
    overflowRedzoneStorage.resize(redzoneSize_);

    if (auto result = ctx_->enqueueCopyFromDevice(
            underflowRedzoneStorage.data(), underflowRedzonePtr, redzoneSize_);
        result.isError())
      return Error("failed reading underflow redzone: " +
                   Twine(result.getError()));

    if (auto result = ctx_->enqueueCopyFromDevice(
            overflowRedzoneStorage.data(), overflowRedzonePtr, redzoneSize_);
        result.isError())
      return Error("failed reading overflow redzone: " +
                   Twine(result.getError()));

    if (auto result = ctx_->synchronize(); result.isError())
      return Error("failed reading redzone: " + Twine(result.getError()));

    underflowRedzone = underflowRedzoneStorage.data();
    overflowRedzone = overflowRedzoneStorage.data();
  }

  const char *sectionLine = "=================================================="
                            "==============================\n";

  bool bufferUnderflow =
      !std::all_of(underflowRedzone, underflowRedzone + redzoneSize_,
                   [&](uint8_t c) { return c == redzonePattern_; });
  if (bufferUnderflow) {
    (llvm::errs().changeColor(llvm::raw_ostream::YELLOW)
     << sectionLine
     << "MemoryManager detected a device buffer underflow before address: "
     << wrappedPtr << "\n")
        .resetColor();
    printRedzone(underflowRedzone, underflowRedzonePtr);
  }

  bool bufferOverflow =
      !std::all_of(overflowRedzone, overflowRedzone + redzoneSize_,
                   [&](uint8_t c) { return c == redzonePattern_; });
  if (bufferOverflow) {
    (llvm::errs().changeColor(llvm::raw_ostream::YELLOW)
     << sectionLine
     << "MemoryManager detected a device buffer overflow after "
        "address+size: "
     << wrappedPtr << "+" << info.size_ << " (" << overflowRedzonePtr << ")\n")
        .resetColor();
    printRedzone(overflowRedzone, overflowRedzonePtr);
  }

  if (bufferUnderflow || bufferOverflow) {
    llvm::errs().changeColor(llvm::raw_ostream::YELLOW)
        << "\nStack trace at the point of the device buffer allocation:\n\n";
    llvm::errs().resetColor() << info.stackTraceOnAlloc_;
    (llvm::errs().changeColor(llvm::raw_ostream::YELLOW) << sectionLine)
        .resetColor();
    std::abort();
  }

  return success();
}

void MemoryManagerFreeList::printRedzone(const uint8_t *dataPtr,
                                         const uint8_t *displayPtr) {
  llvm::errs() << "Redzone size: " << redzoneSize_ << " bytes, "
               << "fill pattern: "
               << llvm::format_hex_no_prefix(redzonePattern_, 2) << "\n\n";
  constexpr uint32_t bytesPerLine = 16;
  size_t numLines = redzoneSize_ / bytesPerLine;
  bool invalidRegion = false;
  for (uint32_t l = 0; l < numLines; l++) {
    const uint8_t *dataLinePtr = dataPtr + l * bytesPerLine;
    const uint8_t *displayLinePtr = displayPtr + l * bytesPerLine;

    if (!std::all_of(dataLinePtr, dataLinePtr + bytesPerLine,
                     [&](uint8_t c) { return c == redzonePattern_; })) {
      (llvm::errs().changeColor(llvm::raw_ostream::RED) << "=>").resetColor();
    } else
      llvm::errs() << "  ";

    llvm::errs() << displayLinePtr << ":";

    for (uint32_t b = 0; b < bytesPerLine; b++) {
      uint8_t byte = dataLinePtr[b];
      if (invalidRegion) {
        llvm::errs().changeColor(llvm::raw_ostream::RED);
        if (byte == redzonePattern_) {
          (llvm::errs() << "]").resetColor();
          invalidRegion = false;
        } else
          llvm::errs() << " ";
      } else {
        if (byte != redzonePattern_) {
          llvm::errs().changeColor(llvm::raw_ostream::RED) << "[";
          invalidRegion = true;
        } else
          llvm::errs() << " ";
      }
      llvm::errs() << llvm::format_hex_no_prefix(byte, 2);
    }
    if (invalidRegion && l == numLines - 1)
      llvm::errs() << "]";
    llvm::errs() << "\n";
  }
}

ErrorOrSuccess MemoryManagerFreeList::free(void *ptr) {
  TRACY_ZONE_SCOPED_NC("MemoryManagerFreeList::free", TRACY_COLOR_PURPLE);
  if (ptr == nullptr)
    return success();

  uint8_t *unwrappedPtr = static_cast<uint8_t *>(ptr);
  if (redzoneEnable_) {
    // Unwrap redzone padding applied in allocate() by finalizeDebugAlloc().
    // (Poison-only allocations have no padding, so there is nothing to unwrap.)
    unwrappedPtr -= redzoneSize_;
  }

  if (memoryManager_.chunks_ != nullptr) {
    LOG_MSG("Starting free of " + TWINE_PTR(ptr) +
            " with free id: " + TWINE(primaryFreeBuffers_));

    std::lock_guard<std::mutex> lock(memoryManager_.mu_);
    selfCheck();

    MemoryManager::Desc *desc =
        memoryManager_.allocatedMap_.remove(unwrappedPtr);

    if (desc != nullptr) {
      // The allocating free list gets lost through the different layers of
      // MLRT and MGP, meaning that the pointer being free'ed did not come from
      // the current free list. Dispatch to the allocating freelist to free
      // the descriptor.
      MemoryManagerFreeList *owner = desc->allocatedBy_;
      assert(owner && "desc freed without a valid allocatedBy_ - was it ever "
                      "allocated via allocateDesc?");
      return owner->freeAllocatedDesc(desc, ptr);
    }
  }

  if (redzoneEnable_) {
    RedzoneInfo info = redzoneDeviceAllocSizeMap_.at(ptr);
    redzoneDeviceAllocSizeMap_.erase(ptr);
    if (auto result = validateRedzones(unwrappedPtr, info); result.isError()) {
      return result;
    }
  }

  LOG_MSG("free_buffer(" + TWINE_PTR(ptr) + ")");
  return ctx_->memFree(ptr, memoryManager_.isHost());
}

void MemoryManagerFreeList::freeDesc(MemoryManager::Desc *desc) {
  void *ptr = desc->ptr_;
  size_t bufSize = desc->size_;
  desc->chunk_->mergeFreeBlockDesc(this, desc);
  LOG_MSG("free (" + TWINE_PTR(ptr) + " ; " + TWINE(bufSize) + ")", 0, bufSize);
}

ErrorOrSuccess
MemoryManagerFreeList::freeAllocatedDesc(MemoryManager::Desc *desc, void *ptr) {
  assert(desc->allocatedBy_ == this &&
         "descriptor must be allocated by the current free list");

  if (redzoneEnable_) {
    RedzoneInfo info = redzoneDeviceAllocSizeMap_.at(ptr);
    redzoneDeviceAllocSizeMap_.erase(ptr);
    if (auto result = validateRedzones(desc->ptr_, info); result.isError())
      return result;
  }

  freeDesc(desc);

  if ((ctx_ != nullptr) && !handlerPending_ && !disableFlush_) {
    // Free the list preceding the next primary so that we can keep buffers
    // cached for an extended period.
    // primaryFreeBuffers + 0: current primary
    // primaryFreeBuffers + 1: next primary to be used, should contain null now
    // primaryFreeBuffers + 2: oldest list of free buffers, to be released to
    //                         parent (for the default of 2 lists, this is
    //                         equivalent to the current primary)
    int toFreeId = (primaryFreeBuffers_ + 2) & (kNumFreeBufferLists - 1);
    primaryFreeBuffers_ = (primaryFreeBuffers_ + 1) & (kNumFreeBufferLists - 1);
    assert(freeBuffers_[primaryFreeBuffers_] == nullptr);
    handlerPending_ = true;
    ErrorOrSuccess err = ctx_->enqueueEventHandler(
        // We need to ensure that ctx_ survives while this handler is pending,
        // so we capture the DeviceContextRef.
        [contextRef = DeviceContextRef::copy(ctx_), freelist = this,
         id = toFreeId]() {
          std::lock_guard<std::mutex> lock(freelist->memoryManager_.mu_);
          if (freelist->memoryManager_.doLog_)
            freelist->logMsg("Running flush handler id: " + TWINE(id));
          assert(freelist->handlerPending_);
          freelist->flushFreeBuffersToParent(id);
          freelist->handlerPending_ = false;
        });
    if (err.isError())
      return err.takeError();
    LOG_MSG("Enqueued flush handler id: " + TWINE(toFreeId));
  }

  LOG_MSG("Completed free of " + TWINE_PTR(ptr) +
          " after queuing flush handler");

  return success();
}

MemoryManager *MemoryManager::onDevice(Device &device, Device::MemInfo memInfo,
                                       size_t alignment, int64_t memoryLimit) {
  bool useOnlyMemoryManagerMode =
      (device.api() != Device::cpuAPI) &&
      DeviceContext::configAsBool("memory_manager_only", false);

  auto disabledMemoryManager = [&device, alignment,
                                useOnlyMemoryManagerMode]() -> MemoryManager * {
    return new MemoryManager(
        /*device=*/device,
        /*maxSize=*/0,
        /*chunkSize=*/0,
        /*alignment=*/alignment,
        /*isHost=*/false,
        /*useOnlyMemoryManagerMode=*/useOnlyMemoryManagerMode);

    // TODO (GEX-3475): useOnlyMemoryManagerMode should not be required.
  };

  // Do not use memory manager if memory manager has been disabled by the
  // calling code (empty memInfo).
  // This is for example the case for CPU Device: We are relying on malloc
  // optimizations instead.
  if (memInfo.free == 0 || memoryLimit == 0)
    return disabledMemoryManager();

  // Determine cache size
  bool isDefaultCacheSize = false;
  size_t maxCacheSize;
  if (memoryLimit > 0) // Try to use parameter value first
    maxCacheSize = static_cast<size_t>(memoryLimit);
  else // memoryLimit is < 0: look in the config
    maxCacheSize = static_cast<size_t>(DeviceContext::configAsInt(
        "memory_manager_size", 0, &isDefaultCacheSize));

  // Disable the memory manager if the requested cache size is 0
  if (maxCacheSize == 0 && !isDefaultCacheSize) // User has explicitly set to 0
    return disabledMemoryManager();

  // If there's no provided size or if the provided size is larger than
  // the free memory, use a percentage of the available memory
  if (isDefaultCacheSize || maxCacheSize > memInfo.free) {
    // By default use 90% of the available memory
    int64_t percentMemoryLimit =
        DeviceContext::configAsInt("memory_manager_size_percent", 90);
    if (percentMemoryLimit == 0) // User has explicitly set to 0 => disable
      return disabledMemoryManager();

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
  // TODO: Check that memInfo.free is not violated by this alignment.
  maxCacheSize = llvm::alignToPowerOf2(maxCacheSize, 256 * KB);

  // Get chunk size as percentage of max cache size. Limit to between 1 and 100.
  // GEX-3330
  size_t defaultChunkSizePercent = 100;
  size_t chunkSizePercent = std::max(
      static_cast<size_t>(1),
      std::min(static_cast<size_t>(100),
               static_cast<size_t>(DeviceContext::configAsInt(
                   "memory_manager_chunk_percent", defaultChunkSizePercent))));

  size_t chunkSize = std::min(
      maxCacheSize,
      llvm::alignToPowerOf2((maxCacheSize * chunkSizePercent) / 100, 256 * KB));

  // VMM (CUDA Virtual Memory Management) defragmenting allocator. The default
  // is per-backend (device.vmmEnabledByDefault()); set
  // `memory_manager_vmm=0`/`1` to override, and it auto-disables where
  // vmmSupported() is false. When enabled we serve from a single oversized VA
  // arena backed by remappable physical granules, so a large contiguous
  // allocation can be assembled from physically-scattered free memory
  // (ensureRegionBacked / reclaimMappedGranules). Granule (page) size is
  // rounded up from `memory_manager_vmm_page_mb` (default 256 MiB) to the
  // device VMM granularity, and all VMM sizes are rounded to it.
  // Check vmmSupported() first so we skip the config lookup on backends that
  // cannot use VMM at all.
  bool vmmEnabled = device.vmmSupported() &&
                    DeviceContext::configAsBool("memory_manager_vmm",
                                                device.vmmEnabledByDefault());
  // VMM on AMD (HIP) is known to cause persistent VRAM leaks that can require
  // a reboot, so force it off regardless of config/env, and warn if it was
  // explicitly requested so the misconfiguration is visible.
  if (vmmEnabled && device.api() == Device::hipAPI) {
    llvm::errs() << "DeviceContext.MemoryManager: VMM was requested via "
                    "memory_manager_vmm, but VMM is forced off on AMD GPUs "
                    "due to known VRAM leaks. Ignoring request.\n";
    vmmEnabled = false;
  }
  size_t vmmPageSize = 0;
  if (vmmEnabled) {
    // vmmSupported() has already verified the granularity query succeeds, so
    // this does not error.
    ErrorOr<size_t> granularity = device.vmmGranularity();
    assert(!granularity.isError() &&
           "vmmSupported() implies vmmGranularity() succeeds");
    size_t pageMb = static_cast<size_t>(std::max<int64_t>(
        1, DeviceContext::configAsInt("memory_manager_vmm_page_mb", 256)));
    vmmPageSize = llvm::alignTo(pageMb * 1024 * KB, *granularity);
    // Round the budget to whole granules.
    maxCacheSize = llvm::alignTo(maxCacheSize, vmmPageSize);
    // Single-arena VMM: the whole budget is one lazily-backed VA arena, so
    // `memory_manager_chunk_percent` is ignored (no incremental chunking).
    chunkSize = maxCacheSize;
  }

  auto result = new MemoryManager(
      /*device=*/device,
      /*maxSize=*/maxCacheSize,
      /*chunkSize=*/chunkSize,
      /*alignment=*/alignment,
      /*isHost=*/false,
      /*useOnlyMemoryManagerMode=*/useOnlyMemoryManagerMode,
      /*vmmEnabled=*/vmmEnabled,
      /*vmmPageSize=*/vmmPageSize);

  if (result->doLog_) {
    result->logErr("initial memInfo(free: " + humanMemorySize(memInfo.free) +
                   ", total: " + humanMemorySize(memInfo.total) + ")");
  }
  return result;
}
MemoryManager *MemoryManager::onHost(Device &device, size_t alignment) {

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

  // Bump up the default chunk percent so that when the host memory manager is
  // primed by the Python layer (see `_prime_pinned_memory_cache`), then the
  // first allocation does not cause the entire cache to be allocated at once.
  // 8% of the default 16 GiB pool = 1.28 GiB, which is large enough to fit the
  // 1 GiB priming allocation without tripping the `useMaxSize` heuristic in
  // MemoryManagerFreeList::allocate. On 8x B200 this drops `max serve` startup
  // priming from ~55s to ~4s.
  size_t defaultChunkPercent = 8;

  size_t chunkSize = std::min(
      maxCacheSize, static_cast<size_t>(llvm::alignToPowerOf2(
                        (maxCacheSize * DeviceContext::configAsInt(
                                            "host_memory_manager_chunk_percent",
                                            defaultChunkPercent)) /
                            100,
                        256 * KB)));

  return new MemoryManager(
      /*device=*/device,
      /*maxSize=*/maxCacheSize,
      /*chunkSize=*/chunkSize,
      /*alignment=*/alignment,
      /*isHost=*/true,
      // We disable useOnlyMemoryManagerMode for host memory manager since it is
      // not useful by bazel to better binpack tests on a runner.
      /*useOnlyMemoryManagerMode=*/false);
}

} // namespace M::Driver
