//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_SYNC_H
#define ASYNCRT_NIXL_SYNC_H

#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/utils/common/util.h"
#include <mutex>
#include <shared_mutex>

namespace M::AsyncRT::NIXL {

class nixlLock {
public:
  nixlLock(const nixl_thread_sync_t sync_mode) : syncMode(sync_mode) {}

  void lock() {
    switch (syncMode) {
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE:
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT:
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_RW:
      m.lock();
      break;
    }
  }

  void lock_shared() {
    switch (syncMode) {
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE:
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT:
      m.lock();
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_RW:
      m.lock_shared();
      break;
    }
  }

  void unlock() {
    switch (syncMode) {
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE:
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT:
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_RW:
      m.unlock();
      break;
    }
  }

  void unlock_shared() {
    switch (syncMode) {
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE:
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT:
      m.unlock();
      break;
    case nixl_thread_sync_t::NIXL_THREAD_SYNC_RW:
      m.unlock_shared();
      break;
    }
  }

private:
  nixl_thread_sync_t syncMode;
  std::shared_mutex m;
};

#define NIXL_LOCK_GUARD(lock)                                                  \
  const std::lock_guard<nixlLock> NIXL_UNIQUE_NAME(lock_guard)(lock)

#define NIXL_SHARED_LOCK_GUARD(lock)                                           \
  const std::shared_lock<nixlLock> NIXL_UNIQUE_NAME(lock_guard)(lock)

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_SYNC_H
