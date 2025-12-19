//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_COMMON_UTIL_H
#define ASYNCRT_NIXL_UTILS_COMMON_UTIL_H

#define NIXL_CONCAT(a, b) NIXL_CONCAT_0(a, b)
#define NIXL_CONCAT_0(a, b) a##b
#define NIXL_UNIQUE_NAME(name) NIXL_CONCAT(name, __COUNTER__)

#endif // ASYNCRT_NIXL_UTILS_COMMON_UTIL_H
