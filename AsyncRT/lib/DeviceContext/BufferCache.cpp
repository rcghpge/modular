//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements the portable buffer cache used by device contexts..
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

static constexpr bool do_log = false;
static constexpr bool do_self_check = true;

struct BufferCache::Desc {
  Desc() { reset(); }

  void reset() {
    ptr_ = nullptr;
    size_ = 0;
    next_ = nullptr;
    prev_ = nullptr;
    next_free_ = nullptr;
    prev_free_ = nullptr;
    free_ = false;
  }

  char *ptr_;
  size_t size_;

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
  uintptr_t hash_val = reinterpret_cast<uintptr_t>(ptr);
  hash_val = (~hash_val) + (hash_val << 18); // key = (key << 18) - key - 1;
  hash_val = hash_val ^ (hash_val >> 31);
  hash_val = hash_val * 21; // key = (key + (key << 2)) + (key << 4);
  hash_val = hash_val ^ (hash_val >> 11);
  hash_val = hash_val + (hash_val << 6);
  hash_val = hash_val ^ (hash_val >> 22);
  return static_cast<uint32_t>(hash_val);
}

void BufferCache::DescMap::add(Desc *buf) {
  assert(!buf->free_);
  assert(buf->next_free_ == nullptr);
  assert(buf->prev_free_ == nullptr);

  uint32_t h = hash(buf->ptr_);
  Desc **bucket = &buckets_[h & bucket_mask_];
  buf->next_free_ = *bucket;
  *bucket = buf;
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
  }
  return result;
}

BufferCache::BufferCache(void *base, size_t size, size_t alignment)
    : base_(base), size_(size), alignment_offset_(alignment - 1),
      alignment_mask_(~(alignment - 1)), allocated_map(10) {
  if (size == 0)
    return;

  if (do_log) {
    llvm::errs() << "##(" << this << ")##: base = " << base
                 << ", size = " << size << "\n";
  }

  buffers_ = new Desc();
  buffers_->ptr_ = static_cast<char *>(base);
  buffers_->size_ = size;
  buffers_->free_ = true;

  free_buffers_ = buffers_;
  free_size_ = size;
  free_count_ = 1;
}

BufferCache::~BufferCache() {
  if (do_log && (size_ != 0)) {
    llvm::errs() << "##(" << this << ")##: destructor\n";
  }
  assert(buffers_ == nullptr);
  assert(free_buffers_ == nullptr);
  assert(unused_descs_ == nullptr);
}

void BufferCache::self_check() {
  if (!do_self_check || size_ == 0) {
    return;
  }

  // First buffer points to base of pre-allocated memory.
  assert(buffers_->ptr_ == base_);
  // Check no missing buffers, no cycle
  Desc *cur = buffers_;
  Desc *ahead = cur;
  if (cur != nullptr) {
    assert(cur->prev_ == nullptr);
  }
  while (cur != nullptr) {
    Desc *next = cur->next_;
    if (next != nullptr) {
      assert((cur->ptr_ + cur->size_) == next->ptr_);
      assert(next->prev_ == cur);
    } else {
      assert((cur->ptr_ + cur->size_) ==
             (static_cast<const char *>(base_) + size_));
    }
    // Step ahead.
    if ((ahead != nullptr) && (ahead->next_ != nullptr)) {
      ahead = ahead->next_->next_;
      if (ahead == cur) {
        llvm::errs() << "##(" << this << ")## self_check: cycle - buffers\n";
      }
    }
    cur = next;
  }

  // Check free markers, no cycle
  cur = free_buffers_;
  ahead = cur;
  if (cur != nullptr) {
    assert(cur->prev_free_ == nullptr);
  }
  while (cur != nullptr) {
    Desc *next = cur->next_free_;
    assert(cur->free_);
    if (next != nullptr) {
      assert(next->prev_free_ == cur);
    }
    if ((ahead != nullptr) && (ahead->next_free_ != nullptr)) {
      ahead = ahead->next_free_->next_free_;
      if (ahead == cur) {
        llvm::errs() << "##(" << this
                     << ")## self_check: cycle - free buffers\n";
      }
    }
    cur = next;
  }
  if (do_log) {
    llvm::errs() << "##(" << this << ")## self_check: OK\n";
  }
}

BufferCache::Desc *BufferCache::popDesc() {
  if (unused_descs_ == nullptr) {
    return new Desc();
  }
  Desc *result = unused_descs_;
  unused_descs_ = result->next_;
  result->next_ = nullptr;
  return result;
}

void BufferCache::pushDesc(Desc *desc) {
  desc->reset();
  desc->next_ = unused_descs_;
  unused_descs_ = desc;
}

void BufferCache::clear() {
  if (size_ == 0) {
    return;
  }

  self_check();
  // Ensure one free buffer spanning the whole memory area.
  assert(free_buffers_->ptr_ == base_);
  assert(free_buffers_->size_ == size_);
  assert(free_buffers_->next_free_ == nullptr);
  assert(buffers_->ptr_ == base_);
  assert(buffers_->size_ == size_);
  assert(buffers_->next_ == nullptr);
  assert(buffers_ == free_buffers_);

  delete buffers_;

  Desc *cur = unused_descs_;
  while (cur != nullptr) {
    Desc *next = cur->next_;
    delete cur;
    cur = next;
  }

  buffers_ = nullptr;
  free_buffers_ = nullptr;
  unused_descs_ = nullptr;
}

void *BufferCache::allocate(size_t size) {
  if (size_ == 0) {
    return nullptr;
  }

  assert(size != 0);
  size = (size + alignment_offset_) & alignment_mask_;

  std::lock_guard<std::mutex> lock(mu_);
  self_check();

  // Find a fit in the free list.
  Desc *result = free_buffers_;
  int iterations = 0;
  while ((result != nullptr) && (result->size_ < size)) {
    result = result->next_free_;
    iterations++;
  }
  if (result == nullptr) {
    if (do_log) {
      llvm::errs() << "##(" << this << ")## allocate(" << size
                   << ") - nullptr (" << iterations << ")\n";
    }
    return nullptr;
  }

  assert(result->free_);
  if (result->size_ == size) {
    // Unhook from free list.
    if (result->next_free_ != nullptr) {
      result->next_free_->prev_free_ = result->prev_free_;
    }
    if (result->prev_free_ != nullptr) {
      result->prev_free_->next_free_ = result->next_free_;
    } else {
      free_buffers_ = result->next_free_;
    }
    free_count_--;
  } else {
    // Split the descriptor. Keep the current descriptor in the free list.
    Desc *buf = result;
    result = popDesc();
    result->ptr_ = buf->ptr_;
    result->size_ = size;
    buf->ptr_ = buf->ptr_ + size;
    buf->size_ = buf->size_ - size;
    // Insert into buffer list.
    result->next_ = buf;
    result->prev_ = buf->prev_;
    if (buffers_ == buf) {
      assert(buf->prev_ == nullptr);
      buffers_ = result;
    } else {
      buf->prev_->next_ = result;
    }
    buf->prev_ = result;
  }

  result->next_free_ = nullptr;
  result->prev_free_ = nullptr;
  result->free_ = false;
  allocated_map.add(result);
  free_size_ -= size;
  if (do_log) {
    llvm::errs() << "##(" << this << ")## allocate(" << size << ") - "
                 << reinterpret_cast<void *>(result->ptr_) << " (" << iterations
                 << ") -- " << free_size_ << ", " << free_count_ << "\n";
  }
  return result->ptr_;
}

bool BufferCache::free(void *ptr) {
  if (size_ == 0) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mu_);
  self_check();

  Desc *buf = allocated_map.remove(ptr);
  if (buf == nullptr) {
    if (do_log) {
      llvm::errs() << "##(" << this << ")## free(" << ptr << ") - not found\n";
    }
    return false;
  }
  assert(!buf->free_);
  free_size_ += buf->size_;

  // Check if we can merge with the next buffer.
  int iterations = 0;
  Desc *next = buf->next_;
  Desc *prev = buf->prev_;
  if ((next != nullptr) && next->free_) {
    // Check if we can also merge with the previous buffer.
    if ((prev != nullptr) && prev->free_) {
      // Merge all three keeping prev on the free list.
      prev->size_ += buf->size_ + next->size_;
      // Remove next from free list.
      if (prev->next_free_ == next) {
        // next is following previous in free list.
        prev->next_free_ = next->next_free_;
        if (next->next_free_ != nullptr) {
          next->next_free_->prev_free_ = prev;
        }
        iterations = -1;
      } else if (free_buffers_ == next) {
        // next is at head.
        free_buffers_ = next->next_free_;
        next->next_free_->prev_free_ = nullptr;
        iterations = -2;
      } else {
        // next is not at head.
        next->prev_free_->next_free_ = next->next_free_;
        if (next->next_free_ != nullptr) {
          next->next_free_->prev_free_ = next->prev_free_;
        }
        iterations = -3;
      }
      // Remove buf and next from buffers.
      prev->next_ = next->next_;
      if (next->next_ != nullptr) {
        next->next_->prev_ = prev;
      }
      free_count_--;
      pushDesc(buf);
      pushDesc(next);
    } else {
      // Merge, keeping next on the free list.
      next->ptr_ = buf->ptr_;
      next->size_ += buf->size_;
      // Remove buf from buffers.
      if (buffers_ == buf) {
        buffers_ = next;
        next->prev_ = nullptr;
      } else {
        buf->prev_->next_ = next;
        next->prev_ = buf->prev_;
      }
      iterations = -4;
      pushDesc(buf);
    }
  } else if ((prev != nullptr) && prev->free_) {
    // Merge, keeping prev on the free list.
    prev->size_ += buf->size_;
    // Remove buf from buffers.
    prev->next_ = buf->next_;
    if (buf->next_ != nullptr) {
      buf->next_->prev_ = prev;
    }
    iterations = -5;
    pushDesc(buf);
  } else {
    // Add buf to the free list.
    if (free_buffers_ != nullptr) {
      free_buffers_->prev_free_ = buf;
    }
    buf->next_free_ = free_buffers_;
    assert(buf->prev_free_ == nullptr);
    free_buffers_ = buf;
    buf->free_ = true;
    free_count_++;
    iterations = -6;
  }
  if (do_log) {
    llvm::errs() << "##(" << this << ")## free(" << ptr << ") -  ("
                 << iterations << ") -- " << free_size_ << ", " << free_count_
                 << "\n";
  }
  return true;
}

} // namespace M::AsyncRT
