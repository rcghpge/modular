//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_TIME_H
#define MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_TIME_H

#include <chrono>

namespace M::AsyncRT::NIXL::nixlTime {

using namespace std::chrono;

using ns_t = uint64_t;
using us_t = uint64_t;
using ms_t = uint64_t;
using sec_t = uint64_t;

static inline ns_t getNs() {
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
      .count();
}

static inline us_t getUs() {
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch())
      .count();
}

static inline ms_t getMs() {
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
      .count();
}

static inline sec_t getSec() {
  return duration_cast<seconds>(steady_clock::now().time_since_epoch()).count();
}

} // namespace M::AsyncRT::NIXL::nixlTime

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_NIXL_TIME_H
