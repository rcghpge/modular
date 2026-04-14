//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_TPP
#define MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_TPP

#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"

#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "llvm/Support/ErrorHandling.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace M::MLRT::NIXL {

namespace {

[[maybe_unused]] [[noreturn]] void cyclicBufferFatal(const char *message) {
  llvm::report_fatal_error(message);
}

} // namespace

template <typename T>
sharedRingBuffer<T>::sharedRingBuffer(const std::string &name, bool create,
                                      int version, size_t size)
    : bufferSize_(size) {
  if (create) {
    createCyclicBuffer(name, version);
  } else {
    openCyclicBuffer(name, version);
  }
}

template <typename T> sharedRingBuffer<T>::~sharedRingBuffer() {
  if (header_) {
    msync(header_, getTotalSize(), MS_SYNC);
    munmap(header_, getTotalSize());
  }
}

template <typename T> bool sharedRingBuffer<T>::push(const T &item) {
  const size_t writePos = header_->writePos.load(std::memory_order_relaxed);
  const size_t nextWrite = (writePos + 1) & header_->mask;
  if (nextWrite == header_->readPos.load(std::memory_order_acquire)) {
    return false;
  }

  data_[writePos] = item;
  header_->writePos.store(nextWrite, std::memory_order_release);
  return true;
}

template <typename T> bool sharedRingBuffer<T>::pop(T &item) {
  const size_t readPos = header_->readPos.load(std::memory_order_relaxed);
  if (readPos == header_->writePos.load(std::memory_order_acquire)) {
    return false;
  }

  item = data_[readPos];
  const size_t nextRead = (readPos + 1) & header_->mask;
  header_->readPos.store(nextRead, std::memory_order_release);
  return true;
}

template <typename T> size_t sharedRingBuffer<T>::size() const {
  const size_t writePos = header_->writePos.load(std::memory_order_acquire);
  const size_t readPos = header_->readPos.load(std::memory_order_acquire);
  return (writePos - readPos) & header_->mask;
}

template <typename T> bool sharedRingBuffer<T>::empty() const {
  return header_->readPos.load(std::memory_order_acquire) ==
         header_->writePos.load(std::memory_order_acquire);
}

template <typename T> bool sharedRingBuffer<T>::full() const {
  const size_t writePos = header_->writePos.load(std::memory_order_acquire);
  const size_t nextWrite = (writePos + 1) & header_->mask;
  return nextWrite == header_->readPos.load(std::memory_order_acquire);
}

template <typename T> uint32_t sharedRingBuffer<T>::version() const {
  return header_->version.load(std::memory_order_acquire);
}

template <typename T> size_t sharedRingBuffer<T>::capacity() const {
  return header_->capacity;
}

template <typename T>
sharedRingBuffer<T>::bufferHeader::bufferHeader(size_t size)
    : capacity(size), mask(size - 1) {
  if ((size & (size - 1)) != 0) {
    cyclicBufferFatal("Telemetry buffer size must be a power of 2");
  }
  static_assert(std::is_trivially_copyable_v<T>,
                "T must be trivially copyable for shared memory");
}

template <typename T> size_t sharedRingBuffer<T>::getTotalSize() const {
  return sizeof(bufferHeader) + sizeof(T) * bufferSize_;
}

template <typename T>
void sharedRingBuffer<T>::createCyclicBuffer(const std::string &name,
                                             int version) {
  if (bufferSize_ == 0) {
    cyclicBufferFatal("Cannot create buffer with size 0");
  }

  const int fd = open(name.c_str(), O_CREAT | O_RDWR | O_TRUNC,
                      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (fd == -1) {
    NIXL_PERROR << "Failed to open telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to open telemetry buffer file");
  }

  if (ftruncate(fd, static_cast<off_t>(getTotalSize())) == -1) {
    close(fd);
    unlink(name.c_str());
    NIXL_PERROR << "Failed to size telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to size telemetry buffer file");
  }

  void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    unlink(name.c_str());
    NIXL_PERROR << "Failed to map telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to map telemetry buffer file");
  }

  header_ = static_cast<bufferHeader *>(ptr);
  data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(bufferHeader));

  new (header_) bufferHeader(bufferSize_);
  header_->version.store(version, std::memory_order_release);
}

template <typename T>
void sharedRingBuffer<T>::openCyclicBuffer(const std::string &name,
                                           int version) {
  const int fd = open(name.c_str(), O_RDWR);
  if (fd == -1) {
    NIXL_PERROR << "Failed to open telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to open telemetry buffer file");
  }

  struct stat st;
  if (fstat(fd, &st) == -1) {
    close(fd);
    NIXL_PERROR << "Failed to stat telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to stat telemetry buffer file");
  }

  if (static_cast<size_t>(st.st_size) < sizeof(bufferHeader)) {
    close(fd);
    cyclicBufferFatal("Telemetry buffer file too small");
  }

  void *headerPtr =
      mmap(nullptr, sizeof(bufferHeader), PROT_READ | PROT_WRITE, MAP_SHARED,
           fd, 0);
  if (headerPtr == MAP_FAILED) {
    close(fd);
    NIXL_PERROR << "Failed to map telemetry buffer header: " << name;
    cyclicBufferFatal("Failed to map telemetry buffer header");
  }

  auto *tempHeader = static_cast<bufferHeader *>(headerPtr);
  const int currentVersion = tempHeader->version.load(std::memory_order_acquire);
  if (currentVersion != version) {
    munmap(tempHeader, sizeof(bufferHeader));
    close(fd);
    cyclicBufferFatal("Telemetry buffer version mismatch");
  }

  bufferSize_ = tempHeader->capacity;
  munmap(tempHeader, sizeof(bufferHeader));

  void *ptr = mmap(nullptr, getTotalSize(), PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    NIXL_PERROR << "Failed to map telemetry buffer file: " << name;
    cyclicBufferFatal("Failed to map telemetry buffer file");
  }

  header_ = static_cast<bufferHeader *>(ptr);
  data_ = reinterpret_cast<T *>(static_cast<char *>(ptr) + sizeof(bufferHeader));
}

} // namespace M::MLRT::NIXL

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_CYCLIC_BUFFER_TPP
