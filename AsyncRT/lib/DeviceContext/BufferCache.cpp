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
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

class BufferCache::Chunk {
public:
  Chunk(BufferCache &cache, void *base, size_t size);
  ~Chunk() = default;

  Desc *splitFreeDesc(Desc *freeDesc, size_t size);

private:
  friend class BufferCache;

  BufferCache &cache_;
  void *base_ = nullptr;
  size_t size_ = 0;
  Chunk *next_ = nullptr;
  Desc *descs_ = nullptr; // Ordered list of all memory regions in this chunk.
};

struct BufferCache::Desc {
  Desc() { reset(); }

  void reset() {
    ptr_ = nullptr;
    size_ = 0;
    chunk_ = nullptr;
    next_ = nullptr;
    prev_ = nullptr;
    next_free_ = nullptr;
    prev_free_ = nullptr;
    free_ = false;
  }

  char *ptr_;
  size_t size_;

  Chunk *chunk_;

  Desc *next_;
  Desc *prev_;

  Desc *next_free_;
  Desc *prev_free_;

  bool free_;
};

BufferCache::DescMap::DescMap(int map_size_log2)
    : bucket_count_(1 << map_size_log2), bucket_mask_(bucket_count_ - 1),
      buckets_(new Desc *[bucket_count_]) {
  for (size_t i = 0; i < bucket_count_; i++) {
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
  assert(!desc->free_);
  assert(desc->next_free_ == nullptr);
  assert(desc->prev_free_ == nullptr);

  uint32_t h = hash(desc->ptr_);
  Desc **bucket = &buckets_[h & bucket_mask_];
  desc->next_free_ = *bucket;
  *bucket = desc;
}

BufferCache::Desc *BufferCache::DescMap::remove(void *ptr) {
  uint32_t h = hash(ptr);
  Desc **bucket = &buckets_[h & bucket_mask_];
  Desc *result = *bucket;
  Desc *prev = nullptr;
  while ((result != nullptr) && (result->ptr_ != ptr)) {
    prev = result;
    result = result->next_free_;
  }
  if (result != nullptr) {
    if (prev == nullptr) {
      *bucket = result->next_free_;
    } else {
      prev->next_free_ = result->next_free_;
    }
    result->next_free_ = nullptr;
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

  cache_.addToFreeList(descs_);
}

BufferCache::Desc *BufferCache::Chunk::splitFreeDesc(Desc *freeDesc,
                                                     size_t size) {
  // Split the descriptor. Keep the current freeDesc in the free list.
  assert(freeDesc->chunk_ == this);
  assert(freeDesc->free_);
  // Get a new descriptor for the result. And split of the first part of the
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

llvm::raw_fd_ostream &BufferCache::errOut() {
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
  doLog_ = DeviceContext::configAsBool("buffer_cache_log", false);
  doSelfCheck_ = DeviceContext::configAsBool("buffer_cache_self_check", false);
}

BufferCache::~BufferCache() {
  if (doLog_) {
    errOut() << "##(" << deviceDescription_ << ":" << this
             << ")##: destructor\n";
  }
  assert(chunks_ == nullptr);
  assert(freeBuffers_ == nullptr);
  assert(unusedDescs_ == nullptr);
}

void BufferCache::addToFreeList(Desc *desc) {
  assert(!desc->free_);
  assert(desc->next_free_ == nullptr);
  assert(desc->prev_free_ == nullptr);
  assert(desc->chunk_ != nullptr);

  desc->free_ = true;
  desc->next_free_ = freeBuffers_;
  if (freeBuffers_ != nullptr) {
    freeBuffers_->prev_free_ = desc;
  }
  freeBuffers_ = desc;

  freeSize_ += desc->size_;
  freeCount_++;
}

void BufferCache::removeFromFreeList(Desc *desc) {
  assert(desc->free_);
  assert(desc->chunk_ != nullptr);

  if (desc == freeBuffers_) {
    assert(desc->prev_free_ == nullptr);
    freeBuffers_ = desc->next_free_;
    if (freeBuffers_ != nullptr) {
      freeBuffers_->prev_free_ = nullptr;
    }
  } else {
    assert(desc->prev_free_ != nullptr);
    desc->prev_free_->next_free_ = desc->next_free_;
    if (desc->next_free_ != nullptr) {
      desc->next_free_->prev_free_ = desc->prev_free_;
    }
  }
  desc->next_free_ = nullptr;
  desc->prev_free_ = nullptr;
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

void BufferCache::selfCheck() {
  if (!doSelfCheck_ || chunks_ == nullptr) {
    return;
  }

  Chunk *curChunk = chunks_;
  while (curChunk != nullptr) {
    // First buffer points to base of pre-allocated memory.
    [[maybe_unused]] void *base = curChunk->base_;
    [[maybe_unused]] size_t size = curChunk->size_;
    assert(curChunk->descs_->ptr_ == base);

    Desc *curDesc = curChunk->descs_;
    Desc *aheadDesc = curDesc;
    assert(curDesc != nullptr);
    assert(curDesc->prev_ == nullptr);

    // Check no missing buffers, no cycle
    while (curDesc != nullptr) {
      if (!curDesc->free_) {
        // next_free_ is used in map collision resolution, so it is not
        // necessarily nullptr.
        // assert(curDesc->next_free_ == nullptr);
        assert(curDesc->prev_free_ == nullptr);
      }
      Desc *nextDesc = curDesc->next_;
      if (nextDesc != nullptr) {
        assert((curDesc->ptr_ + curDesc->size_) == nextDesc->ptr_);
        assert(nextDesc->prev_ == curDesc);
      } else {
        assert((curDesc->ptr_ + curDesc->size_) ==
               (static_cast<const char *>(base) + size));
      }
      // Step ahead.
      if ((aheadDesc != nullptr) && (aheadDesc->next_ != nullptr)) {
        aheadDesc = aheadDesc->next_->next_;
        if (aheadDesc == curDesc) {
          errOut() << "##(" << deviceDescription_ << ":" << this
                   << ")## selfCheck: cycle - buffers\n";
          abort();
        }
      }
      curDesc = nextDesc;
    }
    curChunk = curChunk->next_;
  }

  // Check free markers, no cycle
  Desc *curDesc = freeBuffers_;
  Desc *aheadDesc = curDesc;
  if (curDesc != nullptr) {
    assert(curDesc->prev_free_ == nullptr);
  }
  while (curDesc != nullptr) {
    Desc *next = curDesc->next_free_;
    assert(curDesc->free_);
    if (next != nullptr) {
      assert(next->prev_free_ == curDesc);
    }
    if ((aheadDesc != nullptr) && (aheadDesc->next_free_ != nullptr)) {
      aheadDesc = aheadDesc->next_free_->next_free_;
      if (aheadDesc == curDesc) {
        errOut() << "##(" << deviceDescription_ << ":" << this
                 << ")## selfCheck: cycle - free buffers\n";
        abort();
      }
    }
    curDesc = next;
  }
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
  result->next_ = nullptr;
  return result;
}

void BufferCache::pushDesc(Desc *desc) {
  desc->reset();
  desc->next_ = unusedDescs_;
  unusedDescs_ = desc;
}

void BufferCache::clear() {
  if (chunks_ == nullptr) {
    return;
  }

  selfCheck();

  // Ensure each free desc is spanning the whole chunk area.
  Desc *desc = freeBuffers_;
  while (desc != nullptr) {
    [[maybe_unused]] Chunk *chunk = desc->chunk_;
    assert(desc->ptr_ == chunk->base_);
    assert(desc->size_ == chunk->size_);

    desc = desc->next_free_;
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
    assert(chunk->descs_->free_);

    removeFromFreeList(chunk->descs_);
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
  assert(freeBuffers_ == nullptr);

  Desc *cur = unusedDescs_;
  while (cur != nullptr) {
    Desc *next = cur->next_;
    delete cur;
    cur = next;
  }
  unusedDescs_ = nullptr;
}

ErrorOr<void *> BufferCache::allocate(size_t size,
                                      DeviceContext::AllocMode mode) {
  assert(size != 0);
  void *buf = nullptr;
  // Only attempt to allocate from the buffer cache if this is an on-device
  // allocation.
  if (mode == DeviceContext::AllocMode::HOST) {
    ErrorOrSuccess err =
        ctx_.memAlloc(&buf, size, DeviceContext::AllocMode::HOST);
    if (err.isError()) {
      return err.takeError();
    }
    return buf;
  }

  assert((mode == DeviceContext::AllocMode::ASYNC) ||
         (mode == DeviceContext::AllocMode::SYNC));
  if (chunkSize_ != 0) {
    size = (size + alignmentOffset_) & alignmentMask_;

    if (size <= chunkSize_) {
      std::lock_guard<std::mutex> lock(mu_);
      selfCheck();

      buf = allocate_locked(size);
    }
  }

  if (buf == nullptr) {
    // Allocate straight from device driver if cache was not able to fulfill
    // the request. If the cache is disabled, the we pass on the allocation
    // mode.
    ErrorOrSuccess err = ctx_.memAlloc(
        &buf, size, (chunkSize_ != 0) ? DeviceContext::AllocMode::SYNC : mode);
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

void *BufferCache::allocate_locked(size_t size) {
  assert((size & alignmentMask_) == size);
  // Find a fit in the free list.
  Desc *desc = freeBuffers_;
  int iterations = 0;
  while ((desc != nullptr) && (desc->size_ < size)) {
    desc = desc->next_free_;
    iterations++;
  }
  if (desc == nullptr) {
    if (!allocateChunk()) {
      if (doLog_) {
        errOut() << "##(" << deviceDescription_ << ":" << this
                 << ")## allocate(" << size << ") - nullptr (" << iterations
                 << ")\n";
      }
      // No success allocating a new Chunk, so we bail from the cache.
      return nullptr;
    }
    // Allocated a new chunk, try again.
    return allocate_locked(size);
  }

  assert(desc->free_);
  if (desc->size_ == size) {
    // Unhook from free list.
    if (desc->next_free_ != nullptr) {
      desc->next_free_->prev_free_ = desc->prev_free_;
    }
    if (desc->prev_free_ != nullptr) {
      desc->prev_free_->next_free_ = desc->next_free_;
    } else {
      freeBuffers_ = desc->next_free_;
    }
    freeCount_--;
  } else {
    desc = desc->chunk_->splitFreeDesc(desc, size);
  }

  desc->next_free_ = nullptr;
  desc->prev_free_ = nullptr;
  desc->free_ = false;
  allocatedMap_.add(desc);
  freeSize_ -= size;
  if (doLog_) {
    errOut() << "##(" << deviceDescription_ << ":" << this << ")## allocate("
             << reinterpret_cast<void *>(desc->ptr_) << "; " << size << ") - "
             << " (" << iterations << ") , " << cacheSize_ << ", "
             << chunkCount_ << ", " << freeSize_ << ", " << freeCount_ << ", "
             << iterations << ", -" << size << "\n";
  }
  return desc->ptr_;
}

ErrorOrSuccess BufferCache::free(void *buf, bool isHost) {
  if (buf == nullptr)
    return success();

  bool freed = false;
  if ((chunks_ != nullptr) && !isHost) {
    std::lock_guard<std::mutex> lock(mu_);
    selfCheck();
    freed = free_locked(buf);
  }

  if (!freed) {
    if (doLog_) {
      errOut() << "##(" << deviceDescription_ << ":" << this
               << ")## free_buffer(" << buf << ") -  (" << 0 << ") , "
               << cacheSize_ << ", " << chunkCount_ << ", " << freeSize_ << ", "
               << freeCount_ << ", " << 0 << ", " << 0 << "\n";
    }
    return ctx_.memFree(buf, isHost);
  }
  return success();
}

bool BufferCache::free_locked(void *buf) {
  assert(chunks_ != nullptr);
  assert(buf != nullptr);

  Desc *desc = allocatedMap_.remove(buf);
  if (desc == nullptr) {
    if (doLog_) {
      errOut() << "##(" << deviceDescription_ << ":" << this << ")## free("
               << buf << ") - not found\n";
    }
    return false;
  }
  assert(!desc->free_);
  size_t bufSize = desc->size_;

  // The chunk where all the descriptors are situated.
  Chunk *chunk = desc->chunk_;

  // Check if we can merge with the next buffer.
  int codePath = 0;
  Desc *next = desc->next_;
  Desc *prev = desc->prev_;
  if ((next != nullptr) && next->free_) {
    // Check if we can also merge with the previous buffer.
    if ((prev != nullptr) && prev->free_) {
      // Merge all three keeping prev on the free list.
      prev->size_ += bufSize + next->size_;
      // Remove next from free list.
      if (prev->next_free_ == next) {
        // next is following previous in free list.
        prev->next_free_ = next->next_free_;
        if (next->next_free_ != nullptr) {
          next->next_free_->prev_free_ = prev;
        }
        codePath = -1;
      } else if (freeBuffers_ == next) {
        // next is at head.
        freeBuffers_ = next->next_free_;
        next->next_free_->prev_free_ = nullptr;
        codePath = -2;
      } else {
        // next is not at head.
        next->prev_free_->next_free_ = next->next_free_;
        if (next->next_free_ != nullptr) {
          next->next_free_->prev_free_ = next->prev_free_;
        }
        codePath = -3;
      }
      // Remove desc and next from buffers.
      prev->next_ = next->next_;
      if (next->next_ != nullptr) {
        next->next_->prev_ = prev;
      }
      freeCount_--;
      pushDesc(desc);
      pushDesc(next);
    } else {
      // Merge, keeping next on the free list.
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
      pushDesc(desc);
    }
    freeSize_ += bufSize;
  } else if ((prev != nullptr) && prev->free_) {
    // Merge, keeping prev on the free list.
    prev->size_ += bufSize;
    // Remove desc from buffers.
    prev->next_ = desc->next_;
    if (desc->next_ != nullptr) {
      desc->next_->prev_ = prev;
    }
    codePath = -5;
    pushDesc(desc);
    freeSize_ += bufSize;
  } else {
    // Add desc to the free list.
    addToFreeList(desc);
    codePath = -6;
  }
  if (doLog_) {
    errOut() << "##(" << deviceDescription_ << ":" << this << ")## free(" << buf
             << "; " << bufSize << ") -  (" << codePath << ") , " << cacheSize_
             << ", " << chunkCount_ << ", " << freeSize_ << ", " << freeCount_
             << ", " << 0 << ", " << bufSize << "\n";
  }
  return true;
}

} // namespace M::AsyncRT
