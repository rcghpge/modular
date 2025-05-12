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

#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/utils/common/util.h"
#include <mutex>

namespace M::AsyncRT::NIXL {

class nixlLock {
public:
  nixlLock(const nixl_thread_sync_t sync_mode) : syncMode(sync_mode) {}

  void lock() {
    if (syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT) {
      m.lock();
    }
  }

  void unlock() {
    if (syncMode == nixl_thread_sync_t::NIXL_THREAD_SYNC_STRICT) {
      m.unlock();
    }
  }

private:
  nixl_thread_sync_t syncMode;
  std::mutex m;
};

#define NIXL_LOCK_GUARD(lock)                                                  \
  const std::lock_guard<nixlLock> NIXL_UNIQUE_NAME(lock_guard)(lock)

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_SYNC_H
