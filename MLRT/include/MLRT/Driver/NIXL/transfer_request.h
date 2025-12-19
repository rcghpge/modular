//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TRANSFER_REQUEST_H
#define MLRT_DRIVER_NIXL_TRANSFER_REQUEST_H

#include "MLRT/Driver/NIXL/backend/backend_engine.h"
#include <string>
#include <unordered_map>

namespace M::AsyncRT::NIXL {

// Contains pointers to corresponding backend engine and its handler, and
// populated and verified DescLists, and other state and metadata needed for a
// NIXL transfer
class nixlXferReqH {
private:
  nixlBackendEngine *engine = nullptr;
  nixlBackendReqH *backendHandle = nullptr;

  nixl_meta_dlist_t *initiatorDescs = nullptr;
  nixl_meta_dlist_t *targetDescs = nullptr;

  std::string remoteAgent;
  nixl_blob_t notifMsg;
  bool hasNotif = false;

  nixl_xfer_op_t backendOp;
  nixl_status_t status;

public:
  nixlXferReqH() = default;

  ~nixlXferReqH() {
    // delete checks for nullptr itself
    delete initiatorDescs;
    delete targetDescs;
    if (backendHandle != nullptr)
      engine->releaseReqH(backendHandle);
  }

  nixlXferReqH(const nixlXferReqH &) = delete;
  nixlXferReqH &operator=(const nixlXferReqH &) = delete;
  nixlXferReqH(nixlXferReqH &&) = delete;
  nixlXferReqH &operator=(nixlXferReqH &&) = delete;

  friend class nixlAgent;
};

class nixlDlistH {
private:
  std::unordered_map<nixlBackendEngine *, nixl_meta_dlist_t *> descs;

  std::string remoteAgent;
  bool isLocal;

public:
  inline nixlDlistH() {}

  inline ~nixlDlistH() {
    for (auto &elm : descs)
      delete elm.second;
  }

  friend class nixlAgent;
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_TRANSFER_REQUEST_H
