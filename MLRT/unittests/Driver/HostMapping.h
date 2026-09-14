//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_UNITTESTS_DRIVER_HOST_MAPPING_H
#define MLRT_UNITTESTS_DRIVER_HOST_MAPPING_H

#include <cstddef>
#include <cstdint>
#include <sys/mman.h>
#include <unistd.h>

namespace {

inline size_t hostPageSize() { return static_cast<size_t>(getpagesize()); }

// A page-aligned mapping the test owns, standing in for MAX's host KV tier:
// one mmap that wrapped handles alias in pieces. Null `data()` means it failed.
class HostMapping {
public:
  explicit HostMapping(size_t byteSize) : byteSize_(byteSize) {
    void *mapped = mmap(nullptr, byteSize_, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ptr_ = (mapped == MAP_FAILED) ? nullptr : mapped;
  }

  ~HostMapping() {
    if (ptr_ != nullptr)
      munmap(ptr_, byteSize_);
  }

  HostMapping(const HostMapping &) = delete;
  HostMapping &operator=(const HostMapping &) = delete;
  HostMapping(HostMapping &&) = delete;
  HostMapping &operator=(HostMapping &&) = delete;

  void *data() const { return ptr_; }
  uint8_t *bytes() const { return static_cast<uint8_t *>(ptr_); }
  size_t byteSize() const { return byteSize_; }

private:
  void *ptr_ = nullptr;
  size_t byteSize_;
};

} // namespace

#endif // MLRT_UNITTESTS_DRIVER_HOST_MAPPING_H
