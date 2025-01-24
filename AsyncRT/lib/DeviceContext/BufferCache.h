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

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

class BufferCache {
public:
  struct Desc;

  class DescMap {
  public:
    DescMap(int map_size_log2);
    ~DescMap();

    void add(Desc *buf);
    Desc *remove(void *ptr);

  private:
    static uint32_t hash(void *ptr);

    size_t bucket_count_;
    size_t bucket_mask_;
    Desc **buckets_;
  };

  BufferCache(void *base, size_t size, size_t alignment);
  ~BufferCache();

  void clear();

  void *allocate(size_t size);
  bool free(void *ptr);

private:
  void self_check();

  // Avoid malloc under lock.
  Desc *popDesc();
  void pushDesc(Desc *desc);

  [[maybe_unused]] const void *const base_;
  const size_t size_;

  const size_t alignment_offset_;
  const size_t alignment_mask_;

  std::mutex mu_;
  Desc *buffers_ = nullptr;      // Ordered list of all memory regions.
  Desc *free_buffers_ = nullptr; // List of unallocated memory regions.

  Desc *unused_descs_ = nullptr; // Free list of descriptors to avoid malloc.
  DescMap allocated_map;

  // For logging
  size_t free_size_ = 0;
  size_t free_count_ = 0;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_LIB_DEVICECONTEXT_BUFFERCACHE_H
