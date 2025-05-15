//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the portable buffer cache used by device contexts..
//
//===----------------------------------------------------------------------===//

#include "BufferCache.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

class BufferCache::Chunk {
public:
  Chunk(BufferCache &cache, void *base, size_t size);
  ~Chunk() = default;

  Desc *splitFreeDesc(Desc *freeDesc, size_t size);

  // merged block descriptor to the freelist `owner`.
  void mergeFreeBlockDesc(BufferCacheFreeList *owner, Desc *desc);

private:
  friend class BufferCache;
  friend class BufferCacheFreeList;

  void selfCheck() const;

  BufferCache &cache_;
  void *base_ = nullptr;
  size_t size_ = 0;
  Chunk *next_ = nullptr;
  Desc *descs_ = nullptr; // Ordered list of all memory regions in this chunk.
};

// Describes a memory region managed by the buffer cache. These descriptors are
// maintained out of line of the allocated blocks so they can be accessed by the
// host, while the memory resides on a device.
class BufferCache::Desc {
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

  BufferCacheFreeList *owner() const { return owner_; }
  bool isFree() const { return owner_ != nullptr; }
  bool isInUse() const { return !isFree(); }

  void setInUse() { owner_ = nullptr; }
  void setFree(BufferCacheFreeList *owner) {
    assert(owner != nullptr);
    owner_ = owner;
  }

private:
  friend class BufferCache;
  friend class BufferCache::Chunk;
  friend class BufferCache::DescMap;
  friend class BufferCacheFreeList;

  // Fields owned by BufferCache::Chunk.
  // Note: Fields read by other parts of the code are marked as atomic.
  char *ptr_;
  std::atomic<size_t> size_;

  // Sub allocation of which chunk.
  Chunk *chunk_;

  Desc *next_;
  Desc *prev_;

  // Fields owned by the BufferCacheFreeList:
  // These fields are only accessed by the BufferCacheFreeList and do not need
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
  std::atomic<BufferCacheFreeList *> owner_;

  // Fields owned by BufferCache::DescMap.
  Desc *nextMap_;
};

BufferCache::DescMap::DescMap(int mapSizeLog2)
    : bucketCount_(1 << mapSizeLog2), bucketMask_(bucketCount_ - 1),
      buckets_(new Desc *[bucketCount_]) {
  for (size_t i = 0; i < bucketCount_; i++) {
    buckets_[i] = nullptr;
  }
}

BufferCache::DescMap::~DescMap() { delete[] buckets_; }

uint32_t BufferCache::DescMap::hash(void *ptr) {
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

void BufferCache::DescMap::add(Desc *desc) {
  assert(desc->nextMap_ == nullptr);
  assert(!desc->isFree());

  uint32_t h = hash(desc->ptr_);
  Desc **bucket = &buckets_[h & bucketMask_];
  desc->nextMap_ = *bucket;
  *bucket = desc;
}

BufferCache::Desc *BufferCache::DescMap::remove(void *ptr) {
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

BufferCache::Chunk::Chunk(BufferCache &cache, void *base, size_t size)
    : cache_(cache), base_(base), size_(size) {
  descs_ = cache.popDesc();
  descs_->reset();
  descs_->ptr_ = static_cast<char *>(base_);
  descs_->size_ = size_;
  descs_->chunk_ = this;

  cache_.freeList_->addToFreeList(descs_);
}

BufferCache::Desc *BufferCache::Chunk::splitFreeDesc(Desc *freeDesc,
                                                     size_t size) {
  // Split the descriptor. Keep the current freeDesc in the free list.
  assert(freeDesc->chunk_ == this);
  assert(freeDesc->isFree());
  // Get a new descriptor for the result. And split off the first part of the
  // freeDesc to the result.
  Desc *result = cache_.popDesc();
  result->chunk_ = this;
  result->ptr_ = freeDesc->ptr_;
  result->size_ = size;
  freeDesc->ptr_ = freeDesc->ptr_ + size;
  freeDesc->size_ = freeDesc->size_ - size;
  // Insert into buffer list.
  result->next_ = freeDesc;
  result->prev_ = freeDesc->prev_;
  if (descs_ == freeDesc) {
    assert(freeDesc->prev_ == nullptr);
    descs_ = result;
  } else {
    freeDesc->prev_->next_ = result;
  }
  freeDesc->prev_ = result;
  return result;
}

void BufferCache::Chunk::mergeFreeBlockDesc(BufferCacheFreeList *owner,
                                            Desc *desc) {
  assert(desc != nullptr);
  assert(desc->isInUse());

  // The chunk where the block is located and into which we are attempting to
  // merge.
  BufferCache::Chunk *chunk = desc->chunk_;
  size_t bufSize = desc->size_;
  BufferCache &bufferCache = chunk->cache_;

  // Check if we can merge with the next buffer.
  int codePath = 0;
  BufferCache::Desc *next = desc->next_;
  BufferCache::Desc *prev = desc->prev_;
  if ((next != nullptr) && (next->owner() == owner)) {
    // Check if we can also merge with the previous buffer.
    if ((prev != nullptr) && (prev->owner() == owner)) {
      // Merge prev, desc and next, keeping prev on the free list.
      prev->size_ += bufSize + next->size_;
      // Remove next from free list.
      owner->removeFromFreeList(next);
      // Remove desc and next from buffers.
      prev->next_ = next->next_;
      if (next->next_ != nullptr) {
        next->next_->prev_ = prev;
      }
      bufferCache.freeCount_--;
      bufferCache.pushDesc(desc);
      bufferCache.pushDesc(next);
    } else {
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
      codePath = -4;
      bufferCache.pushDesc(desc);
    }
    bufferCache.freeSize_ += bufSize;
  } else if ((prev != nullptr) && (prev->owner() == owner)) {
    // Merge prev and desc, keeping prev on the free list.
    prev->size_ += bufSize;
    // Remove desc from buffers.
    prev->next_ = desc->next_;
    if (desc->next_ != nullptr) {
      desc->next_->prev_ = prev;
    }
    codePath = -5;
    bufferCache.pushDesc(desc);
    bufferCache.freeSize_ += bufSize;
  } else {
    // Add desc to the free list.
    owner->addToFreeList(desc);
    codePath = -6;
  }
  if (bufferCache.doLog_) {
    bufferCache.errOut() << "##(" << bufferCache.deviceDescription_ << ":"
                         << this << ")## mergeFreeBlockDesc(" << desc->ptr_
                         << "; " << bufSize << ") -  (" << codePath << ") , "
                         << bufferCache.cacheSize_ << ", "
                         << bufferCache.chunkCount_ << ", "
                         << bufferCache.freeSize_ << ", "
                         << bufferCache.freeCount_ << ", " << 0 << ", "
                         << bufSize << "\n";
  }
}

void BufferCache::Chunk::selfCheck() const {
  // First buffer points to base of pre-allocated memory.
  assert(descs_->ptr_ == base_);

  Desc *curDesc = descs_;
  Desc *aheadDesc = curDesc;
  assert(curDesc != nullptr);
  assert(curDesc->prev_ == nullptr);

  // Check no missing buffers, no cycle
  while (curDesc != nullptr) {
    if (!curDesc->isFree()) {
      // nextFree_ is used in map collision resolution, so it is not
      // necessarily nullptr.
      // assert(curDesc->nextFree_ == nullptr);
      assert(curDesc->prevFree_ == nullptr);
    }
    Desc *nextDesc = curDesc->next_;
    if (nextDesc != nullptr) {
      assert((curDesc->ptr_ + curDesc->size_) == nextDesc->ptr_);
      assert(nextDesc->prev_ == curDesc);
    } else {
      assert((curDesc->ptr_ + curDesc->size_) ==
             (static_cast<const char *>(base_) + size_));
    }
    // Step ahead.
    if ((aheadDesc != nullptr) && (aheadDesc->next_ != nullptr)) {
      aheadDesc = aheadDesc->next_->next_;
      if (aheadDesc == curDesc) {
        cache_.errOut() << "##(" << cache_.deviceDescription_ << ":" << this
                        << ")## selfCheck: cycle - buffers\n";
        abort();
      }
    }
    curDesc = nextDesc;
  }
}

BufferCacheFreeList::BufferCacheFreeList(BufferCache &bufferCache,
                                         BufferCacheFreeList *parent)
    : bufferCache_(bufferCache), parent_(parent) {}

BufferCacheFreeList::~BufferCacheFreeList() {}

bool BufferCacheFreeList::empty() { return freeBuffers_ == nullptr; }

void BufferCacheFreeList::addToFreeList(BufferCache::Desc *desc) {
  assert(!desc->isFree());
  assert(desc->nextFree_ == nullptr);
  assert(desc->prevFree_ == nullptr);
  assert(desc->chunk_ != nullptr);

  desc->setFree(this);
  desc->nextFree_ = freeBuffers_;
  if (freeBuffers_ != nullptr) {
    freeBuffers_->prevFree_ = desc;
  }
  freeBuffers_ = desc;

  bufferCache_.freeSize_ += desc->size_;
  bufferCache_.freeCount_++;
}

void BufferCacheFreeList::removeFromFreeList(BufferCache::Desc *desc) {
  assert(desc->isFree());
  assert(desc->chunk_ != nullptr);

  if (desc == freeBuffers_) {
    assert(desc->prevFree_ == nullptr);
    freeBuffers_ = desc->nextFree_;
    if (freeBuffers_ != nullptr) {
      freeBuffers_->prevFree_ = nullptr;
    }
  } else {
    assert(desc->prevFree_ != nullptr);
    desc->prevFree_->nextFree_ = desc->nextFree_;
    if (desc->nextFree_ != nullptr) {
      desc->nextFree_->prevFree_ = desc->prevFree_;
    }
  }
  desc->nextFree_ = nullptr;
  desc->prevFree_ = nullptr;
}

void BufferCacheFreeList::selfCheck() const {
  // Check free markers, no cycle
  BufferCache::Desc *curDesc = freeBuffers_;
  BufferCache::Desc *aheadDesc = curDesc;
  if (curDesc != nullptr) {
    assert(curDesc->prevFree_ == nullptr);
  }
  while (curDesc != nullptr) {
    BufferCache::Desc *next = curDesc->nextFree_;
    assert(curDesc->isFree());
    if (next != nullptr) {
      assert(next->prevFree_ == curDesc);
    }
    if ((aheadDesc != nullptr) && (aheadDesc->nextFree_ != nullptr)) {
      aheadDesc = aheadDesc->nextFree_->nextFree_;
      if (aheadDesc == curDesc) {
        bufferCache_.errOut()
            << "##(" << bufferCache_.deviceDescription_ << ":" << this
            << ")## selfCheck: cycle - free buffers\n";
        abort();
      }
    }
    curDesc = next;
  }
}

llvm::raw_fd_ostream &BufferCache::errOut() const {
  if (deviceDescription_.empty()) {
    deviceDescription_ =
        (ctx_.device().api() + "[" + llvm::Twine(ctx_.device().id()) + "]")
            .str();
    // Print the intial information about this BufferCache.
    llvm::errs() << "##(" << deviceDescription_ << ":" << this
                 << ")##: buffer_cache(chunk_size: " << chunkSize_
                 << "; max_cache_size: " << maxCacheSize_ << ")\n";
  }
  return llvm::errs();
}

BufferCache::BufferCache(const DeviceContext &ctx, size_t chunkSize,
                         size_t maxCacheSize, size_t alignment)
    : ctx_(ctx), chunkSize_(chunkSize), maxCacheSize_(maxCacheSize),
      alignmentOffset_(alignment - 1), alignmentMask_(~(alignment - 1)),
      allocatedMap_(chunkSize == 0 ? 0 : 10) {
  // chunks_ will be lazily initialized on the first allocation request.
  freeList_ = std::make_unique<BufferCacheFreeList>(*this, nullptr);
  doLog_ = DeviceContext::configAsBool("buffer_cache_log", false);
  doSelfCheck_ = DeviceContext::configAsBool("buffer_cache_self_check", false);
}

BufferCache::~BufferCache() {
  if (doLog_) {
    errOut() << "##(" << deviceDescription_ << ":" << this
             << ")##: destructor\n";
  }
  assert(chunks_ == nullptr);
  assert(freeList_->empty());
  assert(unusedDescs_ == nullptr);
}

bool BufferCache::allocateChunk() {
  const size_t allocSize =
      std::min(chunkSize_, std::max(0UL, maxCacheSize_ - cacheSize_));
  const size_t minChunkSize = std::min(128UL * 1024, (chunkSize_ * 25) / 100);
  if (allocSize < minChunkSize) {
    return false;
  }
  void *ptr = nullptr;
  ErrorOrSuccess err = ctx_.device().chunkAlloc(&ptr, allocSize);
  if (err.isError()) {
    if (doLog_) {
      errOut() << "##(" << deviceDescription_ << ":" << this
               << ")## allocate_chunk(" << allocSize << ") - failed with "
               << err.getError() << "\n";
    }
    // Keep chunks_ unmodified.
    return false;
  }
  Chunk *chunk = new Chunk(*this, ptr, allocSize);
  chunk->next_ = chunks_;
  chunks_ = chunk;
  cacheSize_ += allocSize;
  chunkCount_++;
  if (doLog_) {
    if (deviceDescription_.empty()) {
      deviceDescription_ =
          (ctx_.device().api() + "[" + llvm::Twine(ctx_.device().id()) + "]")
              .str();
    }
    errOut() << "##(" << deviceDescription_ << ":" << this
             << ")## allocate_chunk(" << ptr << "; " << allocSize << ") - "
             << cacheSize_ << ", " << chunkCount_ << ", " << freeSize_ << ", "
             << freeCount_ << ", " << 0 << ", " << 0 << "\n";
  }
  return true;
}

void BufferCache::selfCheck() const {
  if (!doSelfCheck_ || chunks_ == nullptr) {
    return;
  }

  Chunk *curChunk = chunks_;
  while (curChunk != nullptr) {
    curChunk->selfCheck();
    curChunk = curChunk->next_;
  }

  freeList_->selfCheck();
  if (doLog_) {
    errOut() << "##(" << deviceDescription_ << ":" << this
             << ")## selfCheck: OK\n";
  }
}

BufferCache::Desc *BufferCache::popDesc() {
  if (unusedDescs_ == nullptr) {
    return new Desc();
  }
  Desc *result = unusedDescs_;
  unusedDescs_ = result->next_;
  // Make sure all values are reset before using.
  result->reset();
  return result;
}

void BufferCache::pushDesc(Desc *desc) {
  // Do not preemptively clear the desc. The reset work could be unnecessary and
  // reset before writing will load the memory locations in the cache.
  desc->next_ = unusedDescs_;
  unusedDescs_ = desc;
}

void BufferCache::clear() {
  if (chunks_ == nullptr) {
    return;
  }

  selfCheck();

  // Ensure each free desc is spanning the whole chunk area.
  Desc *desc = freeList_->freeBuffers_;
  while (desc != nullptr) {
    assert(desc->ptr_ == desc->chunk_->base_);
    assert(desc->size_ == desc->chunk_->size_);

    desc = desc->nextFree_;
  }

  Chunk *chunk = chunks_;
  while (chunk != nullptr) {
    Chunk *nextChunk = chunk->next_;
    [[maybe_unused]] void *base = chunk->base_;
    [[maybe_unused]] size_t size = chunk->size_;

    // Ensure each chunk only contains one free desc.
    assert(chunk->descs_->ptr_ == base);
    assert(chunk->descs_->size_ == size);
    assert(chunk->descs_->prev_ == nullptr);
    assert(chunk->descs_->next_ == nullptr);
    assert(chunk->descs_->isFree());

    freeList_->removeFromFreeList(chunk->descs_);
    delete chunk->descs_;
    // Release the chunk memory.
    if (doLog_) {
      errOut() << "##(" << deviceDescription_ << ":" << this
               << ")## free_chunk(" << chunk->base_ << ", " << chunk->size_
               << ")\n";
    }
    ErrorOrSuccess err = ctx_.device().chunkFree(chunk->base_);
    if (err.isError()) {
      errOut() << err.getError() << "\n";
      abort();
    }
    assert(!err.isError());
    delete chunk;

    chunk = nextChunk;
  }
  chunks_ = nullptr;
  assert(freeList_->freeBuffers_ == nullptr);

  Desc *cur = unusedDescs_;
  while (cur != nullptr) {
    Desc *next = cur->next_;
    delete cur;
    cur = next;
  }
  unusedDescs_ = nullptr;
}

ErrorOr<void *> BufferCache::allocate(size_t size, Device::AllocMode mode) {
  assert(size != 0);
  void *buf = nullptr;
  // Only attempt to allocate from the buffer cache if this is an on-device
  // allocation.
  if (mode == Device::AllocMode::HOST) {
    ErrorOrSuccess err = ctx_.memAlloc(&buf, size, Device::AllocMode::HOST);
    if (err.isError()) {
      return err.takeError();
    }
    return buf;
  }

  assert((mode == Device::AllocMode::ASYNC) ||
         (mode == Device::AllocMode::SYNC));
  if (chunkSize_ != 0) {
    size = (size + alignmentOffset_) & alignmentMask_;

    if (size <= chunkSize_) {
      std::lock_guard<std::mutex> lock(mu_);
      selfCheck();

      buf = freeList_->allocate(size);
    }
  }

  if (buf == nullptr) {
    // Allocate straight from device driver if cache was not able to fulfill
    // the request. If the cache is disabled, the we pass on the allocation
    // mode.
    ErrorOrSuccess err = ctx_.memAlloc(
        &buf, size, (chunkSize_ != 0) ? Device::AllocMode::SYNC : mode);
    if (err.isError()) {
      if (doLog_) {
        errOut() << "##(" << deviceDescription_ << ":" << this
                 << ")## allocate_buffer(" << size << ") - failed with "
                 << err.getError() << "\n";
      }
      return err.takeError();
    }
    if (doLog_) {
      errOut() << "##(" << deviceDescription_ << ":" << this
               << ")## allocate_buffer(" << (buf) << "; " << size << ") - , "
               << cacheSize_ << ", " << chunkCount_ << ", " << freeSize_ << ", "
               << freeCount_ << ", " << 0 << ", " << 0 << "\n";
    }
  }
  assert(buf != nullptr);
  return buf;
}

void *BufferCacheFreeList::allocate(size_t size) {
  assert((size & bufferCache_.alignmentMask_) == size);
  int iterations = 0;
  {
    // Find a fit in the free list.
    BufferCache::Desc *desc = freeBuffers_;
    while ((desc != nullptr) && (desc->size_ < size)) {
      desc = desc->nextFree_;
      iterations++;
    }
    if (desc != nullptr) {

      assert(desc->isFree());
      if (desc->size_ == size) {
        // Unhook from free list.
        if (desc->nextFree_ != nullptr) {
          desc->nextFree_->prevFree_ = desc->prevFree_;
        }
        if (desc->prevFree_ != nullptr) {
          desc->prevFree_->nextFree_ = desc->nextFree_;
        } else {
          freeBuffers_ = desc->nextFree_;
        }
        bufferCache_.freeCount_--;
      } else {
        desc = desc->chunk_->splitFreeDesc(desc, size);
      }

      desc->nextFree_ = nullptr;
      desc->prevFree_ = nullptr;
      desc->setInUse();
      bufferCache_.allocatedMap_.add(desc);
      bufferCache_.freeSize_ -= size;
      if (bufferCache_.doLog_) {
        bufferCache_.errOut()
            << "##(" << bufferCache_.deviceDescription_ << ":" << this
            << ")## allocate(" << reinterpret_cast<void *>(desc->ptr_) << "; "
            << size << ") - "
            << " (" << iterations << ") , " << bufferCache_.cacheSize_ << ", "
            << bufferCache_.chunkCount_ << ", " << bufferCache_.freeSize_
            << ", " << bufferCache_.freeCount_ << ", " << iterations << ", -"
            << size << "\n";
      }
      return desc->ptr_;
    }
  }

  // Delegate to parent freelist if available.
  if (parent_ != nullptr) {
    return parent_->allocate(size);
  }

  if (!bufferCache_.allocateChunk()) {
    if (bufferCache_.doLog_) {
      bufferCache_.errOut()
          << "##(" << bufferCache_.deviceDescription_ << ":" << this
          << ")## allocate(" << size << ") - nullptr (" << iterations << ")\n";
    }
    // No success allocating a new Chunk, so we bail from the cache.
    return nullptr;
  }
  // Allocated a new chunk, try again.
  return allocate(size);
}

ErrorOrSuccess BufferCache::free(void *buf, bool isHost) {
  if (buf == nullptr)
    return success();

  if ((chunks_ != nullptr) && !isHost) {
    std::lock_guard<std::mutex> lock(mu_);
    selfCheck();

    Desc *desc = allocatedMap_.remove(buf);
    if (desc != nullptr) {
      freeList_->free(desc);
      return success();
    }
  }

  if (doLog_ && !isHost) {
    errOut() << "##(" << deviceDescription_ << ":" << this << ")## free_buffer("
             << buf << ") -  (" << 0 << ") , " << cacheSize_ << ", "
             << chunkCount_ << ", " << freeSize_ << ", " << freeCount_ << ", "
             << 0 << ", " << 0 << "\n";
  }
  return ctx_.memFree(buf, isHost);
}

void BufferCacheFreeList::free(BufferCache::Desc *desc) {
  desc->chunk_->mergeFreeBlockDesc(this, desc);
}

} // namespace M::AsyncRT
