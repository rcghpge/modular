//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_COMMON_STR_TOOLS_H
#define ASYNCRT_NIXL_UTILS_COMMON_STR_TOOLS_H

#include <string>
#include <string_view>
#include <vector>

namespace M::AsyncRT::NIXL {

// Splits a string such that each view is a slice of the input.
std::vector<std::string_view> str_split_substr(std::string_view str,
                                               std::string_view delim);

class strEqual {
public:
  bool operator()(const std::string_view t1, const std::string_view t2) const {
    size_t s1 = t1.size();
    size_t s2 = t2.size();

    if (s1 != s2)
      return false;
    if (((s1 & 7) != 0) || (s1 > 64))
      return (t1 == t2);

    size_t i = 0;
    const char *d1 = t1.data();
    const char *d2 = t2.data();

    for (i = 0; i < s1; i += 8) {
      if (*((const uint64_t *)(d1 + i)) != *((const uint64_t *)(d2 + i)))
        return false;
    }
    return true;
  }
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_COMMON_STR_TOOLS_H
