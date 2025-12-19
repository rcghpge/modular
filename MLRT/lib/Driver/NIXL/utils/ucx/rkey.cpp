//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/utils/ucx/rkey.h"

#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "MLRT/Driver/NIXL/utils/ucx/ucx_utils.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;
using namespace M::AsyncRT::NIXL;

M::ErrorOr<std::unique_ptr<UCX::Rkey>>
UCX::Rkey::create(const nixlUcxEp &ep, const void *rkeyBuffer) {
  ucp_rkey_h rkey = nullptr;
  const auto status = ucp_ep_rkey_unpack(ep.getEp(), rkeyBuffer, &rkey);
  if (status != UCS_OK) {
    NIXL_ERROR << "Failed to unpack UCX rkey: " << ucs_status_string(status);
    return Error(llvm::formatv("Failed to unpack UCX rkey: {}",
                               ucs_status_string(status)));
  }
  return std::unique_ptr<Rkey>(new Rkey(rkey));
}

UCX::Rkey::~Rkey() {
  if (rkey_)
    ucp_rkey_destroy(rkey_);
}
