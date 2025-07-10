//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_UCX_UCX_UTILS_H
#define ASYNCRT_NIXL_UTILS_UCX_UCX_UTILS_H

extern "C" {
#include "ucp/api/ucp.h"
}

#include "AsyncRT/NIXL/nixl_types.h"
#include <cstdint>
#include <string_view>
#include <vector>

namespace M::AsyncRT::NIXL {

enum class UcxCallbackOp : std::uint8_t { CONN_CHECK, NOTIF_STR, DISCONNECT };

struct nixl_ucx_am_hdr {
  UcxCallbackOp op;
};

enum nixl_ucx_mt_t {
  NIXL_UCX_MT_SINGLE,
  NIXL_UCX_MT_CTX,
  NIXL_UCX_MT_WORKER,
  NIXL_UCX_MT_MAX
};

class nixlUcxEp {
private:
  ucp_ep_h eph;

public:
  friend class nixlUcxWorker;
};

class nixlUcxMem {
private:
  void *base;
  size_t size;
  ucp_mem_h memh;

public:
  friend class nixlUcxWorker;
};

class nixlUcxRkey {
private:
  ucp_rkey_h rkeyh;

public:
  friend class nixlUcxWorker;
};

using nixlUcxReq = void *;

class nixlUcxContext {
private:
  /* Local UCX stuff */
  ucp_context_h ctx;
  nixl_ucx_mt_t mt_type;

public:
  using req_cb_t = void(void *request);
  nixlUcxContext(std::vector<std::string_view> devices, size_t req_size,
                 req_cb_t init_cb, req_cb_t fini_cb, nixl_ucx_mt_t mt_type);
  ~nixlUcxContext();

  static bool mtLevelIsSupported(nixl_ucx_mt_t mt_type);

  friend class nixlUcxWorker;
};

class nixlUcxWorker {
private:
  /* Local UCX stuff */
  nixlUcxContext *ctx;
  ucp_worker_h worker;

public:
  nixlUcxWorker(nixlUcxContext *ctx);
  ~nixlUcxWorker();

  /* Connection */
  int epAddr(uint64_t &addr, size_t &size);
  int connect(const void *addr, size_t size, nixlUcxEp &ep);
  int disconnect(nixlUcxEp &ep);
  int disconnect_nb(nixlUcxEp &ep);

  /* Memory management */
  int memReg(void *addr, size_t size, nixlUcxMem &mem);
  size_t packRkey(nixlUcxMem &mem, uint64_t &addr, size_t &size);
  void memDereg(nixlUcxMem &mem);

  /* Rkey */
  int rkeyImport(nixlUcxEp &ep, blob_ref data, nixlUcxRkey &rkey);
  void rkeyDestroy(nixlUcxRkey &rkey);

  /* Active message handling */
  int regAmCallback(UcxCallbackOp callbackOp, ucp_am_recv_callback_t cb,
                    void *arg);
  nixl_status_t sendAm(nixlUcxEp &ep, unsigned msg_id, void *hdr,
                       size_t hdr_len, void *buffer, size_t len, uint32_t flags,
                       nixlUcxReq &req);
  int getRndvData(void *data_desc, void *buffer, size_t len,
                  const ucp_request_param_t *param, nixlUcxReq &req);

  /* Data access */
  int progress();
  nixl_status_t flushEp(nixlUcxEp &ep, nixlUcxReq &req);
  nixl_status_t read(nixlUcxEp &ep, uint64_t raddr, nixlUcxRkey &rk,
                     void *laddr, nixlUcxMem &mem, size_t size,
                     nixlUcxReq &req);
  nixl_status_t write(nixlUcxEp &ep, void *laddr, nixlUcxMem &mem,
                      uint64_t raddr, nixlUcxRkey &rk, size_t size,
                      nixlUcxReq &req);
  nixl_status_t test(nixlUcxReq req);

  void reqRelease(nixlUcxReq req);
  void reqCancel(nixlUcxReq req);
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_UCX_UCX_UTILS_H
