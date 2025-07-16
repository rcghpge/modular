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
#include "Support/ErrorOr.h"
#include <cstdint>
#include <memory>
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

using nixlUcxReq = void *;

class nixlUcxRkey;
class nixlUcxMem;

class nixlUcxEp {
  enum class nixl_ucx_ep_state_t : std::uint8_t {
    NIXL_UCX_EP_STATE_NULL,
    NIXL_UCX_EP_STATE_CONNECTED,
    NIXL_UCX_EP_STATE_FAILED,
    NIXL_UCX_EP_STATE_DISCONNECTED
  };

private:
  ucp_ep_h eph{nullptr};
  nixl_ucx_ep_state_t state{nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL};

  void setState(nixl_ucx_ep_state_t new_state);
  nixl_status_t closeImpl(ucp_worker_h worker, ucp_ep_close_flags_t flags);
  nixl_status_t closeNonblocking() {
    return closeImpl(nullptr, ucp_ep_close_flags_t(0));
  }

  /* Connection */
  nixl_status_t disconnectNonblocking();

public:
  void err_cb(ucp_ep_h ucp_ep, ucs_status_t status);

  nixl_status_t checkTxState() const {
    switch (state) {
    case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED:
      return nixl_status_t::NIXL_SUCCESS;
    case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED:
      return nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT;
    case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL:
    case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_DISCONNECTED:
      return nixl_status_t::NIXL_ERR_BACKEND;
    }
  }

  nixlUcxEp(ucp_worker_h worker, void *addr);
  ~nixlUcxEp();
  nixlUcxEp(const nixlUcxEp &) = delete;
  nixlUcxEp &operator=(const nixlUcxEp &) = delete;
  nixlUcxEp(const nixlUcxEp &&) = delete;
  nixlUcxEp &operator=(const nixlUcxEp &&) = delete;

  /* Rkey */
  int rkeyImport(void *addr, size_t size, nixlUcxRkey &rkey);
  void rkeyDestroy(nixlUcxRkey &rkey);

  /* Active message handling */
  nixl_status_t sendAm(unsigned msg_id, void *hdr, size_t hdr_len, void *buffer,
                       size_t len, uint32_t flags, nixlUcxReq &req);

  /* Data access */
  nixl_status_t read(uint64_t raddr, nixlUcxRkey &rk, void *laddr,
                     nixlUcxMem &mem, size_t size, nixlUcxReq &req);
  nixl_status_t write(void *laddr, nixlUcxMem &mem, uint64_t raddr,
                      nixlUcxRkey &rk, size_t size, nixlUcxReq &req);
  nixl_status_t flushEp(nixlUcxReq &req);

  friend class nixlUcxWorker;
};

class nixlUcxMem {
private:
  void *base;
  size_t size;
  ucp_mem_h memh;

public:
  friend class nixlUcxWorker;
  friend class nixlUcxContext;
  friend class nixlUcxEp;
};

class nixlUcxRkey {
private:
  ucp_rkey_h rkeyh;

public:
  friend class nixlUcxWorker;
  friend class nixlUcxEp;
};

class nixlUcxContext {
private:
  /* Local UCX stuff */
  ucp_context_h ctx;
  nixl_ucx_mt_t mt_type;

public:
  using req_cb_t = void(void *request);
  nixlUcxContext(std::vector<std::string_view> devices, size_t req_size,
                 req_cb_t init_cb, req_cb_t fini_cb, nixl_ucx_mt_t mt_type,
                 bool prog_thread = false);
  ~nixlUcxContext();

  static bool mtLevelIsSupported(nixl_ucx_mt_t mt_type);

  /* Memory management */
  int memReg(void *addr, size_t size, nixlUcxMem &mem);
  std::unique_ptr<char[]> packRkey(nixlUcxMem &mem, size_t &size);
  void memDereg(nixlUcxMem &mem);

  friend class nixlUcxWorker;
};

class nixlUcxWorker {
private:
  /* Local UCX stuff */
  std::shared_ptr<nixlUcxContext> ctx;
  ucp_worker_h worker;

public:
  nixlUcxWorker(std::shared_ptr<nixlUcxContext> &_ctx);
  ~nixlUcxWorker();

  /* Connection */
  std::unique_ptr<char[]> epAddr(size_t &size);
  M::ErrorOr<std::unique_ptr<nixlUcxEp>> connect(void *addr, size_t size);

  /* Active message handling */
  int regAmCallback(UcxCallbackOp callbackOp, ucp_am_recv_callback_t cb,
                    void *arg);
  int getRndvData(void *data_desc, void *buffer, size_t len,
                  const ucp_request_param_t *param, nixlUcxReq &req);

  /* Data access */
  int progress();
  nixl_status_t test(nixlUcxReq req);

  void reqRelease(nixlUcxReq req);
  void reqCancel(nixlUcxReq req);

  ucp_worker_h getWorker() const { return worker; }
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_UCX_UCX_UTILS_H
