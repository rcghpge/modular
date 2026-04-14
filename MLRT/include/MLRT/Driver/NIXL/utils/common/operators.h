//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_OPERATORS_H
#define MLRT_DRIVER_NIXL_UTILS_COMMON_OPERATORS_H

#include "MLRT/Driver/NIXL/nixl_types.h"
#include "llvm/Support/raw_ostream.h"

namespace M::MLRT::NIXL {

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_mem_t value) {
  return os << NixlEnumStrings::memTypeStr(value);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_xfer_op_t value) {
  return os << NixlEnumStrings::xferOpStr(value);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_status_t value) {
  return os << NixlEnumStrings::statusStr(value);
}

} // namespace M::MLRT::NIXL

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_OPERATORS_H
