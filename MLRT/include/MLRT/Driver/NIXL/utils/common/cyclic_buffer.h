//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_H
#define MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace M::MLRT {

template <typename T>
class sharedRingBuffer {
public:
  sharedRingBuffer(const std::string &name, bool create, int version,
                   size_t size = 0);
  sharedRingBuffer(sharedRingBuffer &&) = delete;
  ~sharedRingBuffer();

  sharedRingBuffer(const sharedRingBuffer &) = delete;
  sharedRingBuffer &operator=(sharedRingBuffer &&) = delete;
  sharedRingBuffer &operator=(const sharedRingBuffer &) = delete;

  bool push(const T &item);
  bool pop(T &item);
  size_t size() const;
  bool empty() const;
  bool full() const;
  uint32_t version() const;
  size_t capacity() const;

private:
  struct bufferHeader {
    std::atomic<size_t> writePos{0};
    std::atomic<size_t> readPos{0};
    std::atomic<int> version{0};
    const size_t capacity;
    size_t mask;

    explicit bufferHeader(size_t size);
  };

  size_t getTotalSize() const;
  void createCyclicBuffer(const std::string &name, int version);
  void openCyclicBuffer(const std::string &name, int version);

  bufferHeader *header_ = nullptr;
  T *data_ = nullptr;
  size_t bufferSize_ = 0;
};

} // namespace M::MLRT

#include "cyclic_buffer.tpp"

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_H
