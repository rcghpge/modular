//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_COMMON_OPERATORS_H
#define ASYNCRT_NIXL_UTILS_COMMON_OPERATORS_H

#include "AsyncRT/NIXL/nixl_types.h"
#include "llvm/Support/raw_ostream.h"

namespace M::AsyncRT::NIXL {

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_mem_t value) {
  return os << nixlEnumStrings::memTypeStr(value);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_xfer_op_t value) {
  return os << nixlEnumStrings::xferOpStr(value);
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const nixl_status_t value) {
  return os << nixlEnumStrings::statusStr(value);
}

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_COMMON_OPERATORS_H
