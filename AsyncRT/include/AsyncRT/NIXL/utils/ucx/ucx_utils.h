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

enum class nixl_ucx_mt_t { SINGLE, CTX, WORKER };

using nixlUcxReq = void *;

// Helper function to get backend parameters with default values
template <typename T>
[[nodiscard]] T nixl_b_params_get(const nixl_b_params_t *custom_params,
                                  const std::string &key, T default_value) {
  if (!custom_params) {
    return default_value;
  }

  auto it = custom_params->find(key);
  if (it == custom_params->end()) {
    return default_value;
  }

  if constexpr (std::is_same_v<T, size_t>) {
    const char *start = it->second.c_str();
    char *end;
    unsigned long long parsed = std::strtoull(start, &end, 10);
    // Check for conversion errors: no digits parsed or invalid characters
    if (end == start || *end != '\0') {
      return default_value;
    }
    return static_cast<size_t>(parsed);
  }
  return default_value;
}

namespace UCX {
class Rkey;
}
class nixlUcxMem;

class nixlUcxEp {
  enum class nixl_ucx_ep_state_t : std::uint8_t {
    NIXL_UCX_EP_STATE_NULL,
    NIXL_UCX_EP_STATE_CONNECTED,
    NIXL_UCX_EP_STATE_FAILED,
    NIXL_UCX_EP_STATE_DISCONNECTED,
  };

private:
  ucp_ep_h eph{nullptr};
  nixl_ucx_ep_state_t state{nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL};

  void setState(nixl_ucx_ep_state_t new_state);
  nixl_status_t closeImpl(ucp_ep_close_flags_t flags);

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

  nixlUcxEp(ucp_worker_h worker, void *addr,
            ucp_err_handling_mode_t err_handling_mode);
  ~nixlUcxEp();
  nixlUcxEp(const nixlUcxEp &) = delete;
  nixlUcxEp &operator=(const nixlUcxEp &) = delete;
  nixlUcxEp(const nixlUcxEp &&) = delete;
  nixlUcxEp &operator=(const nixlUcxEp &&) = delete;

  ucp_ep_h getEp() const { return eph; }

  /* Active message handling */
  nixl_status_t sendAm(unsigned msg_id, void *hdr, size_t hdr_len, void *buffer,
                       size_t len, uint32_t flags, nixlUcxReq &req);

  /* Data access */
  [[nodiscard]] nixl_status_t read(uint64_t raddr, const UCX::Rkey &rkey,
                                   void *laddr, nixlUcxMem &mem, size_t size,
                                   nixlUcxReq &req);
  [[nodiscard]] nixl_status_t write(void *laddr, nixlUcxMem &mem,
                                    uint64_t raddr, const UCX::Rkey &rkey,
                                    size_t size, nixlUcxReq &req);
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

class nixlUcxContext {
private:
  /* Local UCX stuff */
  ucp_context_h ctx;
  nixl_ucx_mt_t mt_type;

public:
  using req_cb_t = void(void *request);
  nixlUcxContext(std::vector<std::string_view> devices, size_t req_size,
                 req_cb_t init_cb, req_cb_t fini_cb, bool prog_thread,
                 unsigned long num_workers, nixl_thread_sync_t sync_mode);
  ~nixlUcxContext();

  /* Memory management */
  int memReg(void *addr, size_t size, nixlUcxMem &mem,
             nixl_mem_t nixl_mem_type = nixl_mem_t::DRAM_SEG);
  std::string packRkey(nixlUcxMem &mem);
  void memDereg(nixlUcxMem &mem);

  /* GPU signal support */
  M::ErrorOr<int> prepGpuSignal(const nixlUcxMem &mem, void *signal) const;
  M::ErrorOr<size_t> getGpuSignalSize() const;

  friend class nixlUcxWorker;
};

bool nixlUcxMtLevelIsSupported(nixl_ucx_mt_t mt_type);

class nixlUcxWorker {
public:
  /// Fallible constructor for UCX worker.
  static M::ErrorOr<std::unique_ptr<nixlUcxWorker>> create(
      const nixlUcxContext &ctx,
      ucp_err_handling_mode_t err_handling_mode = UCP_ERR_HANDLING_MODE_NONE);

  nixlUcxWorker(const nixlUcxContext &ctx, ucp_worker_h _worker,
                ucp_err_handling_mode_t err_handling_mode);
  ~nixlUcxWorker() = default;

  nixlUcxWorker(const nixlUcxWorker &) = delete;
  nixlUcxWorker &operator=(const nixlUcxWorker &) = delete;
  nixlUcxWorker(nixlUcxWorker &&) = delete;
  nixlUcxWorker &operator=(nixlUcxWorker &&) = delete;

  /* Connection */
  std::string epAddr();
  M::ErrorOr<std::unique_ptr<nixlUcxEp>> connect(void *addr, size_t size);

  /* Active message handling */
  int regAmCallback(UcxCallbackOp callbackOp, ucp_am_recv_callback_t cb,
                    void *arg);

  /* Data access */
  int progress();
  nixl_status_t test(nixlUcxReq req);

  void reqRelease(nixlUcxReq req);
  void reqCancel(nixlUcxReq req);

  [[nodiscard]] nixl_status_t arm() const noexcept;
  [[nodiscard]] M::ErrorOr<int> getEfd() const;

  ucp_worker_h getWorker() const { return worker.get(); }

private:
  static M::ErrorOr<ucp_worker_h> createUcpWorker(const nixlUcxContext &ctx);

  std::unique_ptr<ucp_worker, void (*)(ucp_worker *)> worker;
  ucp_err_handling_mode_t err_handling_mode_;
};

constexpr std::string_view nixl_ucx_err_handling_param_name =
    "ucx_error_handling_mode";

[[nodiscard]] nixl_b_params_t get_ucx_backend_common_options();

nixl_status_t ucx_status_to_nixl(ucs_status_t status);

[[nodiscard]] std::string_view
ucx_err_mode_to_string(ucp_err_handling_mode_t t);

[[nodiscard]] ucp_err_handling_mode_t
ucx_err_mode_from_string(std::string_view s);

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_UCX_UCX_UTILS_H
