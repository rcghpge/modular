//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_COMMON_LIST_ELEM_H
#define MLRT_DRIVER_NIXL_UTILS_COMMON_LIST_ELEM_H

#include <utility>

namespace M::AsyncRT::NIXL {

template <typename T>
class nixlLinkElem {
private:
  T *_next{nullptr};

public:
  nixlLinkElem() = default;

  /* Link this element into the chain after "elem" */
  void link(T *elem) {
    elem->_next = _next;
    _next = elem;
  }

  /* Exclude this element from the chain, return the new head */
  T *unlink() {
    /* Forget my place */
    return std::exchange(_next, nullptr);
  }

  T *next() { return _next; }
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_UTILS_COMMON_LIST_ELEM_H
