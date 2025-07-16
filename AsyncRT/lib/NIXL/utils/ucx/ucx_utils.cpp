//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/utils/ucx/ucx_utils.h"
#include "AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "Support/STLExtras.h"
#include <cassert>
#include <cstring>
#include <string>
#include <vector>

using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

static nixl_status_t ucx_status_to_nixl(ucs_status_t status) {
  if (status == UCS_OK) [[likely]] {
    return nixl_status_t::NIXL_SUCCESS;
  }

  switch (status) {
  case UCS_INPROGRESS:
    return nixl_status_t::NIXL_IN_PROG;
  case UCS_ERR_NOT_CONNECTED:
  case UCS_ERR_CONNECTION_RESET:
  case UCS_ERR_ENDPOINT_TIMEOUT:
    return nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT;
  case UCS_ERR_INVALID_PARAM:
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  default:
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
}

void nixlUcxEp::err_cb(void *arg, ucp_ep_h ucp_ep, ucs_status_t status) {
  nixlUcxEp *ep = reinterpret_cast<nixlUcxEp *>(arg);
  ucs_status_ptr_t request;

  NIXL_DEBUG << "ep " << ep->getHandle() << ": state "
             << to_underlying(ep->state)
             << ", UCX error handling callback was invoked with status "
             << status << " (" << ucs_status_string(status) << ")";

  assert(ep->getHandle() == ucp_ep);

  switch (ep->state) {
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL:
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED:
  // The error was already handled, nothing to do
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_DISCONNECTED:
    // The EP has been disconnected, nothing to do
    return;
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED:
    ep->setState(nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED);
    request = ucp_ep_close_nb(ucp_ep, UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(request)) {
      ucp_request_free(request);
    }
    return;
  }
}

void nixlUcxEp::setState(nixl_ucx_ep_state_t new_state) {
  assert(new_state != state);
  NIXL_DEBUG << "ep " << eph << ": state " << to_underlying(state) << " -> "
             << to_underlying(new_state);
  state = new_state;
}

nixl_status_t nixlUcxEp::connect(ucp_worker_h worker, void *addr) {
  ucp_ep_params_t ep_params;
  nixl_status_t status;

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                         UCP_EP_PARAM_FIELD_ERR_HANDLER |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
  ep_params.err_handler.cb = err_cb;
  ep_params.err_handler.arg = reinterpret_cast<void *>(this);
  ep_params.address = reinterpret_cast<ucp_address_t *>(addr);

  status = ucx_status_to_nixl(ucp_ep_create(worker, &ep_params, &eph));
  if (status == nixl_status_t::NIXL_SUCCESS) {
    setState(nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED);
  }

  return status;
}

nixl_status_t nixlUcxEp::closeImpl(ucp_worker_h worker,
                                   ucp_ep_close_flags_t flags) {
  ucs_status_ptr_t request = nullptr;
  ucp_request_param_t req_param = {};
  req_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
  req_param.flags = flags;

  switch (state) {
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL:
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_DISCONNECTED:
    // The EP has not been connected, or already disconnected.
    // Nothing to do.
    assert(eph == nullptr);
    return nixl_status_t::NIXL_SUCCESS;
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED:
    // The EP was closed in error callback, just return error.
    eph = nullptr;
    return nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT;
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED:
    request = ucp_ep_close_nbx(eph, &req_param);
    if (request == nullptr) {
      eph = nullptr;
      return nixl_status_t::NIXL_SUCCESS;
    }

    if (UCS_PTR_IS_ERR(request)) {
      eph = nullptr;
      return ucx_status_to_nixl(UCS_PTR_STATUS(request));
    }

    if (worker == nullptr) {
      ucp_request_free(request);
      eph = nullptr;
      return nixl_status_t::NIXL_SUCCESS;
    }
    break;
  }

  assert(UCS_PTR_IS_PTR(request));
  assert(worker != nullptr);

  // Blocking close.
  ucs_status_t status;
  do {
    ucp_worker_progress(worker);
    status = ucp_request_check_status(request);
  } while (status == UCS_INPROGRESS);

  ucp_request_free(request);
  eph = nullptr;
  setState(nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_DISCONNECTED);

  return ucx_status_to_nixl(status);
}

bool nixlUcxContext::mtLevelIsSupported(nixl_ucx_mt_t mt_type) {
  ucp_lib_attr_t attr;
  attr.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
  ucp_lib_query(&attr);

  switch (mt_type) {
  case NIXL_UCX_MT_SINGLE:
    return (attr.max_thread_level >= UCS_THREAD_MODE_SERIALIZED);
  case NIXL_UCX_MT_CTX:
  case NIXL_UCX_MT_WORKER:
    return (attr.max_thread_level >= UCS_THREAD_MODE_MULTI);
  default:
    assert(mt_type < NIXL_UCX_MT_MAX);
    abort();
  }
  return false;
}

nixlUcxContext::nixlUcxContext(std::vector<std::string_view> devs,
                               size_t req_size,
                               nixlUcxContext::req_cb_t init_cb,
                               nixlUcxContext::req_cb_t fini_cb,
                               nixl_ucx_mt_t __mt_type) {
  ucp_params_t ucp_params;
  ucp_config_t *ucp_config;
  ucs_status_t status = UCS_OK;

  mt_type = __mt_type;

  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_MT_WORKERS_SHARED |
                          UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
  ucp_params.features =
      UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
  switch (mt_type) {
  case NIXL_UCX_MT_SINGLE:
  case NIXL_UCX_MT_WORKER:
    ucp_params.mt_workers_shared = 0;
    break;
  case NIXL_UCX_MT_CTX:
    ucp_params.mt_workers_shared = 1;
    break;
  default:
    assert(mt_type < NIXL_UCX_MT_MAX);
    abort();
  }
  ucp_params.estimated_num_eps = 3;

  if (req_size) {
    ucp_params.request_size = req_size;
    ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_SIZE;
  }

  if (init_cb) {
    ucp_params.request_init = init_cb;
    ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_INIT;
  }

  if (fini_cb) {
    ucp_params.request_cleanup = fini_cb;
    ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_CLEANUP;
  }

  ucp_config_read(nullptr, nullptr, &ucp_config);

  /* If requested, restrict the set of network devices */
  if (devs.size()) {
    /* TODO: check if this is the best way */
    std::string dev_str = "";
    unsigned int i;
    for (i = 0; i < devs.size() - 1; i++)
      dev_str = (Twine(dev_str) + devs[i] + ":1,").str();
    dev_str = (Twine(dev_str) + devs[i] + ":1").str();
    ucp_config_modify(ucp_config, "NET_DEVICES", dev_str.c_str());
  }

  status = ucp_init(&ucp_params, ucp_config, &ctx);
  if (status != UCS_OK) {
    /* TODO: proper cleanup */
    // TODO: MSW_NET_ERROR(priv->net, "failed to ucp_init(%s)\n",
    // ucs_status_string(status));
    return;
  }
  ucp_config_release(ucp_config);
}

nixlUcxContext::~nixlUcxContext() { ucp_cleanup(ctx); }

nixlUcxWorker::nixlUcxWorker(nixlUcxContext *_ctx) {
  ucp_worker_params_t worker_params;
  ucs_status_t status = UCS_OK;

  ctx = _ctx;

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;

  switch (ctx->mt_type) {
  case NIXL_UCX_MT_CTX:
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    break;
  case NIXL_UCX_MT_SINGLE:
    worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
    break;
  case NIXL_UCX_MT_WORKER:
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    break;
  default:
    assert(ctx->mt_type < NIXL_UCX_MT_MAX);
    abort();
  }

  status = ucp_worker_create(ctx->ctx, &worker_params, &worker);
  if (status != UCS_OK) {
    // TODO: MSW_NET_ERROR(priv->net, "failed to create ucp_worker (%s)\n",
    // ucs_status_string(status));
    return;
  }
}

nixlUcxWorker::~nixlUcxWorker() { ucp_worker_destroy(worker); }

std::unique_ptr<char[]> nixlUcxWorker::epAddr(size_t &size) {
  ucp_worker_attr_t wattr;
  ucs_status_t status;

  wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
  status = ucp_worker_query(worker, &wattr);
  if (UCS_OK != status) {
    // TODO: printf
    return nullptr;
  }

  auto res = std::make_unique<char[]>(wattr.address_length);
  memcpy(res.get(), wattr.address, wattr.address_length);
  ucp_worker_release_address(worker, wattr.address);

  size = wattr.address_length;
  return res;
}

/* ===========================================
 * EP management
 * =========================================== */

int nixlUcxWorker::connect(const void *addr, size_t size, nixlUcxEp &ep) {
  nixl_status_t status = ep.connect(worker, const_cast<void *>(addr));
  return (status == nixl_status_t::NIXL_SUCCESS) ? 0 : -1;
}

int nixlUcxWorker::disconnect(nixlUcxEp &ep) {
  nixl_status_t status = ep.close(worker);
  return (status == nixl_status_t::NIXL_SUCCESS) ? 0 : -1;
}

int nixlUcxWorker::disconnectNonblocking(nixlUcxEp &ep) {
  nixl_status_t status = ep.closeNonblocking();
  return (status == nixl_status_t::NIXL_SUCCESS) ? 0 : -1;
}

/* ===========================================
 * Memory management
 * =========================================== */

int nixlUcxWorker::memReg(void *addr, size_t size, nixlUcxMem &mem) {
  ucs_status_t status;

  // mem.uw = this;
  mem.base = addr;
  mem.size = size;

  ucp_mem_map_params_t mem_params;
  memset(&mem_params, 0, sizeof(mem_params));
  mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                          UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
  mem_params.address = mem.base;
  mem_params.length = mem.size;

  status = ucp_mem_map(ctx->ctx, &mem_params, &mem.memh);
  if (status != UCS_OK) {
    /* TODOL: MSW_NET_ERROR(priv->net, "failed to ucp_mem_map (%s)\n",
     * ucs_status_string(status)); */
    return -1;
  }

  return 0;
}

std::unique_ptr<char[]> nixlUcxWorker::packRkey(nixlUcxMem &mem, size_t &size) {
  ucs_status_t status;
  void *rkey_buf;

  status = ucp_rkey_pack(ctx->ctx, mem.memh, &rkey_buf, &size);
  if (status != UCS_OK) {
    /* TODO: MSW_NET_ERROR(priv->net, "failed to ucp_rkey_pack (%s)\n",
     * ucs_status_string(status)); */
    return nullptr;
  }

  /* Allocate the buffer */
  auto res = std::make_unique<char[]>(size);
  memcpy(res.get(), rkey_buf, size);
  ucp_rkey_buffer_release(rkey_buf);

  return res;
}

void nixlUcxWorker::memDereg(nixlUcxMem &mem) {
  ucp_mem_unmap(ctx->ctx, mem.memh);
}

/* ===========================================
 * RKey management
 * =========================================== */

int nixlUcxWorker::rkeyImport(nixlUcxEp &ep, blob_ref blob, nixlUcxRkey &rkey) {
  ucs_status_t status;

  status = ucp_ep_rkey_unpack(ep.getHandle(), blob.data, &rkey.rkeyh);
  if (status != UCS_OK) {
    /* TODO: MSW_NET_ERROR(priv->net, "unable to unpack key!\n"); */
    return -1;
  }

  return 0;
}

void nixlUcxWorker::rkeyDestroy(nixlUcxRkey &rkey) {
  ucp_rkey_destroy(rkey.rkeyh);
}

/* ===========================================
 * Active message handling
 * =========================================== */

int nixlUcxWorker::regAmCallback(UcxCallbackOp callbackOp,
                                 ucp_am_recv_callback_t cb, void *arg) {
  ucs_status_t status;
  ucp_am_handler_param_t params;
  memset(&params, 0, sizeof(params));

  params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                      UCP_AM_HANDLER_PARAM_FIELD_CB |
                      UCP_AM_HANDLER_PARAM_FIELD_ARG;

  params.id = to_underlying(callbackOp);
  params.cb = cb;
  params.arg = arg;

  status = ucp_worker_set_am_recv_handler(worker, &params);

  if (status != UCS_OK) {
    // TODO: error handling
    return -1;
  }
  return 0;
}

nixl_status_t nixlUcxWorker::sendAm(nixlUcxEp &ep, unsigned msg_id, void *hdr,
                                    size_t hdr_len, void *buffer, size_t len,
                                    uint32_t flags, nixlUcxReq &req) {
  ucs_status_ptr_t request;
  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));

  param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
  param.flags = flags;

  request = ucp_am_send_nbx(ep.getHandle(), msg_id, hdr, hdr_len, buffer, len,
                            &param);

  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

int nixlUcxWorker::getRndvData(void *data_desc, void *buffer, size_t len,
                               const ucp_request_param_t *param,
                               nixlUcxReq &req) {
  ucs_status_ptr_t status;

  status = ucp_am_recv_data_nbx(worker, data_desc, buffer, len, param);
  if (UCS_PTR_IS_ERR(status)) {
    // TODO: error handling
    return -1;
  }
  req = (void *)status;

  return 0;
}

/* ===========================================
 * Data transfer
 * =========================================== */

int nixlUcxWorker::progress() { return ucp_worker_progress(worker); }

nixl_status_t nixlUcxWorker::read(nixlUcxEp &ep, uint64_t raddr,
                                  nixlUcxRkey &rk, void *laddr, nixlUcxMem &mem,
                                  size_t size, nixlUcxReq &req) {
  nixl_status_t status = ep.checkTxState();
  if (status != nixl_status_t::NIXL_SUCCESS) {
    return status;
  }

  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
  param.memh = mem.memh;

  ucs_status_ptr_t request =
      ucp_get_nbx(ep.getHandle(), laddr, size, raddr, rk.rkeyh, &param);
  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxWorker::write(nixlUcxEp &ep, void *laddr, nixlUcxMem &mem,
                                   uint64_t raddr, nixlUcxRkey &rk, size_t size,
                                   nixlUcxReq &req) {
  nixl_status_t status = ep.checkTxState();
  if (status != nixl_status_t::NIXL_SUCCESS) {
    return status;
  }

  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
  param.memh = mem.memh;

  ucs_status_ptr_t request =
      ucp_put_nbx(ep.getHandle(), laddr, size, raddr, rk.rkeyh, &param);
  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxWorker::test(nixlUcxReq req) {
  ucs_status_t status;

  if (req == nullptr) {
    return nixl_status_t::NIXL_SUCCESS;
  }

  ucp_worker_progress(worker);
  status = ucp_request_check_status(req);
  if (status == UCS_INPROGRESS) {
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(status);
}

nixl_status_t nixlUcxWorker::flushEp(nixlUcxEp &ep, nixlUcxReq &req) {
  ucp_request_param_t param;
  ucs_status_ptr_t request;

  param.op_attr_mask = 0;
  request = ucp_ep_flush_nbx(ep.getHandle(), &param);

  if (request == nullptr) {
    return nixl_status_t::NIXL_SUCCESS;
  } else if (UCS_PTR_IS_ERR(request)) {
    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
  }

  req = (void *)request;
  return nixl_status_t::NIXL_IN_PROG;
}

void nixlUcxWorker::reqRelease(nixlUcxReq req) {
  ucp_request_free((void *)req);
}

void nixlUcxWorker::reqCancel(nixlUcxReq req) {
  ucp_request_cancel(worker, (void *)req);
}
