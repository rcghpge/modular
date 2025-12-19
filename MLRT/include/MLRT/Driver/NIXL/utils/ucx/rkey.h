//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_UCX_RKEY_H
#define MLRT_DRIVER_NIXL_UTILS_UCX_RKEY_H

#include <memory>

#include "Support/ErrorOr.h"

extern "C" {
#include <ucp/api/ucp.h>
}

namespace M::AsyncRT::NIXL {

class nixlUcxEp;

namespace UCX {

class Rkey {
public:
  // Factory method since we can't use exceptions in constructors
  static ErrorOr<std::unique_ptr<Rkey>> create(const nixlUcxEp &ep,
                                               const void *rkeyBuffer);

  ~Rkey();

  // Delete copy/move constructors
  Rkey(const Rkey &) = delete;
  Rkey &operator=(const Rkey &) = delete;
  Rkey(Rkey &&) = delete;
  Rkey &operator=(Rkey &&) = delete;

  [[nodiscard]] ucp_rkey_h get() const noexcept { return rkey_; }

private:
  // Private constructor - use create() factory method
  explicit Rkey(ucp_rkey_h rkey) : rkey_(rkey) {}

  ucp_rkey_h rkey_;
};

} // namespace UCX
} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_UTILS_UCX_RKEY_H
