//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/AsyncRT/NIXL/utils/ucx/ucx_utils.h"
#include "MLRT/AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "MLRT/AsyncRT/NIXL/utils/common/nixl_log.h"
#include "MLRT/AsyncRT/NIXL/utils/serdes/serdes.h"
#include "MLRT/AsyncRT/NIXL/utils/ucx/config.h"
#include "MLRT/AsyncRT/NIXL/utils/ucx/rkey.h"
#include "Support/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace M;
using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

nixl_status_t NIXL::ucx_status_to_nixl(ucs_status_t status) {
  if (status == UCS_OK) {
    return nixl_status_t::NIXL_SUCCESS;
  }

  switch (status) {
  case UCS_INPROGRESS:
  case UCS_ERR_BUSY:
    return nixl_status_t::NIXL_IN_PROG;
  case UCS_ERR_NOT_CONNECTED:
  case UCS_ERR_CONNECTION_RESET:
  case UCS_ERR_ENDPOINT_TIMEOUT:
    return nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT;
  case UCS_ERR_INVALID_PARAM:
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  default:
    NIXL_WARN << "Unexpected UCX error: " << ucs_status_string(status);
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
}

[[nodiscard]] nixl_b_params_t NIXL::get_ucx_backend_common_options() {
  nixl_b_params_t params = {{"ucx_devices", ""}, {"num_workers", "1"}};

  params.emplace(
      std::string(nixl_ucx_err_handling_param_name),
      std::string(ucx_err_mode_to_string(UCP_ERR_HANDLING_MODE_PEER)));
  return params;
}

[[nodiscard]] std::string_view
NIXL::ucx_err_mode_to_string(ucp_err_handling_mode_t t) {
  switch (t) {
  case UCP_ERR_HANDLING_MODE_NONE:
    return "none";
  case UCP_ERR_HANDLING_MODE_PEER:
    return "peer";
  }
  llvm_unreachable("Invalid UCX error handling mode");
}

[[nodiscard]] ucp_err_handling_mode_t
NIXL::ucx_err_mode_from_string(std::string_view s) {
  constexpr std::array<ucp_err_handling_mode_t, 2> nixl_ucx_err_handling_modes =
      {
          UCP_ERR_HANDLING_MODE_NONE,
          UCP_ERR_HANDLING_MODE_PEER,
      };

  for (const auto mode : nixl_ucx_err_handling_modes) {
    if (ucx_err_mode_to_string(mode) == s) {
      return mode;
    }
  }

  std::stringstream err_msg;
  err_msg << "Invalid error handling mode: " << s << ". Valid values are: <";
  for (size_t i = 0; i < nixl_ucx_err_handling_modes.size(); ++i) {
    err_msg << ucx_err_mode_to_string(nixl_ucx_err_handling_modes[i]);
    if (i < nixl_ucx_err_handling_modes.size() - 1) {
      err_msg << "|";
    }
  }

  err_msg << ">";
  NIXL_ERROR << err_msg.str();
  return UCP_ERR_HANDLING_MODE_NONE; // Default in case of error
}

static void err_cb_wrapper(void *arg, ucp_ep_h ucp_ep, ucs_status_t status) {
  nixlUcxEp *ep = reinterpret_cast<nixlUcxEp *>(arg);
  ep->err_cb(ucp_ep, status);
}

void nixlUcxEp::err_cb(ucp_ep_h ucp_ep, ucs_status_t status) {
  ucs_status_ptr_t request;

  NIXL_DEBUG << "ep " << eph << ": state "
             << static_cast<int>(to_underlying(state))
             << ", UCX error handling callback was invoked with status "
             << status << " (" << ucs_status_string(status) << ")";

  assert(eph == ucp_ep);

  switch (state) {
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_NULL:
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED:
  // The error was already handled, nothing to do
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_DISCONNECTED:
    // The EP has been disconnected, nothing to do
    return;
  case nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED:
    setState(nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_FAILED);
    request = ucp_ep_close_nb(ucp_ep, UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(request)) {
      ucp_request_free(request);
    }
    return;
  }
  llvm_unreachable("Invalid endpoint state");
}

void nixlUcxEp::setState(nixl_ucx_ep_state_t new_state) {
  assert(new_state != state);
  state = new_state;
}

nixlUcxEp::nixlUcxEp(ucp_worker_h worker, void *addr,
                     ucp_err_handling_mode_t err_handling_mode) {
  ucp_ep_params_t ep_params;
  nixl_status_t status;

  memset(&ep_params, 0, sizeof(ep_params));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                         UCP_EP_PARAM_FIELD_ERR_HANDLER |
                         UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.err_mode = err_handling_mode;
  ep_params.err_handler.cb = err_cb_wrapper;
  ep_params.err_handler.arg = reinterpret_cast<void *>(this);
  ep_params.address = reinterpret_cast<ucp_address_t *>(addr);

  status = ucx_status_to_nixl(ucp_ep_create(worker, &ep_params, &eph));
  if (status == nixl_status_t::NIXL_SUCCESS) {
    setState(nixl_ucx_ep_state_t::NIXL_UCX_EP_STATE_CONNECTED);
  } else {
    // In AsyncRT, we don't throw exceptions. Log error instead.
    NIXL_ERROR << "Failed to create UCX endpoint with status: "
               << to_underlying(status);
  }
}

nixlUcxEp::~nixlUcxEp() {
  nixl_status_t status = disconnectNonblocking();
  if (status != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to disconnect ep with status "
               << to_underlying(status);
  }
}

nixl_status_t nixlUcxEp::closeImpl(ucp_ep_close_flags_t flags) {
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

    ucp_request_free(request);
    eph = nullptr;
    return nixl_status_t::NIXL_SUCCESS;
  }
  llvm_unreachable("Invalid endpoint state");
}

/* ===========================================
 * EP management
 * =========================================== */

nixl_status_t nixlUcxEp::disconnectNonblocking() {
  nixl_status_t status = closeImpl(ucp_ep_close_flags_t(0));

  // At step of disconnect we can ignore the remote disconnect error.
  return (status == nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT)
             ? nixl_status_t::NIXL_SUCCESS
             : status;
}

/* ===========================================
 * Active message handling
 * =========================================== */

nixl_status_t nixlUcxEp::sendAm(unsigned msg_id, void *hdr, size_t hdr_len,
                                void *buffer, size_t len, uint32_t flags,
                                nixlUcxReq &req) {
  ucs_status_ptr_t request;
  ucp_request_param_t param;

  memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
  param.flags = flags;

  request = ucp_am_send_nbx(eph, msg_id, hdr, hdr_len, buffer, len, &param);

  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

/* ===========================================
 * Data transfer
 * =========================================== */

nixl_status_t nixlUcxEp::read(uint64_t raddr, const UCX::Rkey &rkey,
                              void *laddr, nixlUcxMem &mem, size_t size,
                              nixlUcxReq &req) {
  nixl_status_t status = checkTxState();
  if (status != nixl_status_t::NIXL_SUCCESS) {
    return status;
  }

  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FLAG_MULTI_SEND;
  param.memh = mem.memh;

  ucs_status_ptr_t request =
      ucp_get_nbx(eph, laddr, size, raddr, rkey.get(), &param);
  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxEp::write(void *laddr, nixlUcxMem &mem, uint64_t raddr,
                               const UCX::Rkey &rkey, size_t size,
                               nixlUcxReq &req) {
  nixl_status_t status = checkTxState();
  if (status != nixl_status_t::NIXL_SUCCESS) {
    return status;
  }

  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH | UCP_OP_ATTR_FLAG_MULTI_SEND;
  param.memh = mem.memh;

  ucs_status_ptr_t request =
      ucp_put_nbx(eph, laddr, size, raddr, rkey.get(), &param);
  if (UCS_PTR_IS_PTR(request)) {
    req = (void *)request;
    return nixl_status_t::NIXL_IN_PROG;
  }

  return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxEp::flushEp(nixlUcxReq &req) {
  ucp_request_param_t param;
  memset(&param, 0, sizeof(param));
  param.op_attr_mask = 0;

  ucs_status_ptr_t request;
  request = ucp_ep_flush_nbx(eph, &param);

  if (request == nullptr) {
    return nixl_status_t::NIXL_SUCCESS;
  } else if (UCS_PTR_IS_ERR(request)) {
    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
  }

  req = (void *)request;
  return nixl_status_t::NIXL_IN_PROG;
}

bool NIXL::nixlUcxMtLevelIsSupported(nixl_ucx_mt_t mt_type) {
  ucp_lib_attr_t attr;
  attr.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
  ucp_lib_query(&attr);

  switch (mt_type) {
  case nixl_ucx_mt_t::SINGLE:
    return (attr.max_thread_level >= UCS_THREAD_MODE_SERIALIZED);
  case nixl_ucx_mt_t::CTX:
  case nixl_ucx_mt_t::WORKER:
    return (attr.max_thread_level >= UCS_THREAD_MODE_MULTI);
  }
  llvm_unreachable("Invalid mt type");
}

nixlUcxContext::nixlUcxContext(std::vector<std::string_view> devs,
                               size_t req_size,
                               nixlUcxContext::req_cb_t init_cb,
                               nixlUcxContext::req_cb_t fini_cb,
                               bool prog_thread, unsigned long num_workers,
                               nixl_thread_sync_t sync_mode) {
  ucp_params_t ucp_params;
  ucs_status_t status = UCS_OK;

  // With strict synchronization model nixlAgent serializes access to backends,
  // with more permissive models backends need to account for concurrent access
  // and ensure their internal state is properly protected. Progress thread
  // creates internal concurrency in UCX backend irrespective of nixlAgent
  // synchronization model.
  mt_type =
      (sync_mode == nixl_thread_sync_t::NIXL_THREAD_SYNC_RW || prog_thread)
          ? nixl_ucx_mt_t::WORKER
          : nixl_ucx_mt_t::SINGLE;

  ucp_params.field_mask =
      UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
  ucp_params.features =
      UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
  if (prog_thread)
    ucp_params.features |= UCP_FEATURE_WAKEUP;
  ucp_params.mt_workers_shared = num_workers > 1 ? 1 : 0;

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

  auto configOrError = UCX::Config::create();
  if (configOrError.isError()) {
    NIXL_ERROR << "Failed to create UCX config: " << configOrError.getError();
    return;
  }
  UCX::Config config = configOrError.takeValue();

  /* If requested, restrict the set of network devices */
  if (devs.size()) {
    /* TODO: check if this is the best way */
    std::string device = "";
    unsigned int i;
    for (i = 0; i < devs.size() - 1; ++i)
      device = (Twine(device) + devs[i] + ":1,").str();
    device = (Twine(device) + devs[i] + ":1").str();
    config.modifyAlways("NET_DEVICES", device);
  }

  unsigned majorVersion, minorVersion, releaseNumber;
  ucp_get_version(&majorVersion, &minorVersion, &releaseNumber);

  config.modify("ADDRESS_VERSION", "v2");
  config.modify("RNDV_THRESH", "inf");

  unsigned ucpVersion = UCP_VERSION(majorVersion, minorVersion);
  if (ucpVersion >= UCP_VERSION(1, 19)) {
    config.modify("MAX_COMPONENT_MDS", "32");
  }

  if (ucpVersion >= UCP_VERSION(1, 20)) {
    config.modify("MAX_RMA_RAILS", "4");
  } else {
    config.modify("MAX_RMA_RAILS", "2");
  }

  status = ucp_init(&ucp_params, config.getUcpConfig(), &ctx);
  if (status != UCS_OK) {
    /* TODO: proper cleanup */
    // TODO: MSW_NET_ERROR(priv->net, "failed to ucp_init(%s)\n",
    // ucs_status_string(status));
    return;
  }
}

nixlUcxContext::~nixlUcxContext() { ucp_cleanup(ctx); }

static ucs_thread_mode_t toUcsThreadModeChecked(nixl_ucx_mt_t t) {
  switch (t) {
  case nixl_ucx_mt_t::CTX:
    return UCS_THREAD_MODE_SINGLE;
  case nixl_ucx_mt_t::SINGLE:
    return UCS_THREAD_MODE_SERIALIZED;
  case nixl_ucx_mt_t::WORKER:
    return UCS_THREAD_MODE_MULTI;
  }
  llvm_unreachable("Invalid UCX worker type");
}

struct nixlUcpWorkerParams : ucp_worker_params_t {
  explicit nixlUcpWorkerParams(nixl_ucx_mt_t t) {
    field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    thread_mode = toUcsThreadModeChecked(t);
  }
};

static_assert(sizeof(nixlUcpWorkerParams) == sizeof(ucp_worker_params_t));

ErrorOr<ucp_worker_h>
nixlUcxWorker::createUcpWorker(const nixlUcxContext &ctx) {
  ucp_worker_h worker = nullptr;
  nixlUcpWorkerParams params(ctx.mt_type);
  ucs_status_t status = ucp_worker_create(ctx.ctx, &params, &worker);
  if (status != UCS_OK) {
    auto err_str = std::string("Failed to create UCX worker: ") +
                   ucs_status_string(status);
    NIXL_ERROR << err_str;
    return Error(err_str);
  }
  return worker;
}

ErrorOr<std::unique_ptr<nixlUcxWorker>>
nixlUcxWorker::create(const nixlUcxContext &ctx,
                      ucp_err_handling_mode_t err_handling_mode) {
  ErrorOr<ucp_worker_h> workerOrError = createUcpWorker(ctx);
  if (workerOrError.isError()) {
    return workerOrError.takeError();
  }
  return std::make_unique<nixlUcxWorker>(ctx, workerOrError.takeValue(),
                                         err_handling_mode);
}

nixlUcxWorker::nixlUcxWorker(const nixlUcxContext &ctx, ucp_worker_h _worker,
                             ucp_err_handling_mode_t err_handling_mode)
    : worker(_worker, &ucp_worker_destroy),
      err_handling_mode_(err_handling_mode) {
  // Constructor stores the pre-created worker handle with custom deleter
}

std::string nixlUcxWorker::epAddr() {
  ucp_worker_attr_t wattr;

  wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
  ucs_status_t status = ucp_worker_query(worker.get(), &wattr);
  if (UCS_OK != status) {
    NIXL_WARN << "Unable to query UCX endpoint address";
    return {};
  }
  std::string result =
      nixlSerDes::_bytesToString(wattr.address, wattr.address_length);
  ucp_worker_release_address(worker.get(), wattr.address);
  return result;
}

/* ===========================================
 * EP management
 * =========================================== */

ErrorOr<std::unique_ptr<nixlUcxEp>> nixlUcxWorker::connect(void *addr,
                                                           size_t size) {
  auto ep = std::make_unique<nixlUcxEp>(worker.get(), addr, err_handling_mode_);
  // Check if the endpoint was successfully created by checking its state
  if (ep->eph == nullptr) {
    return Error("Failed to create UCX endpoint");
  }
  return ep;
}

/* ===========================================
 * Memory management
 * =========================================== */

int nixlUcxContext::memReg(void *addr, size_t size, nixlUcxMem &mem,
                           nixl_mem_t nixl_mem_type) {
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

  status = ucp_mem_map(ctx, &mem_params, &mem.memh);
  if (status != UCS_OK) {
    /* TODOL: MSW_NET_ERROR(priv->net, "failed to ucp_mem_map (%s)\n",
     * ucs_status_string(status)); */
    return -1;
  }

  if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
    ucp_mem_attr_t attr;
    attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
    status = ucp_mem_query(mem.memh, &attr);
    if (status != UCS_OK) {
      NIXL_ERROR << "Failed to ucp_mem_query: " << ucs_status_string(status);
      ucp_mem_unmap(ctx, mem.memh);
      return -1;
    }

    if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
      NIXL_WARN << "memory is detected as host, check that UCX is configured"
                   " with CUDA support";
    }
  }

  return 0;
}

std::string nixlUcxContext::packRkey(nixlUcxMem &mem) {
  void *rkey_buf;
  size_t size;

  ucs_status_t status = ucp_rkey_pack(ctx, mem.memh, &rkey_buf, &size);
  if (status != UCS_OK) {
    /* TODO: MSW_NET_ERROR(priv->net, "failed to ucp_rkey_pack (%s)\n",
     * ucs_status_string(status)); */
    return {};
  }
  std::string result = nixlSerDes::_bytesToString(rkey_buf, size);
  ucp_rkey_buffer_release(rkey_buf);
  return result;
}

void nixlUcxContext::memDereg(nixlUcxMem &mem) { ucp_mem_unmap(ctx, mem.memh); }

M::ErrorOr<int>
nixlUcxContext::prepGpuSignal([[maybe_unused]] const nixlUcxMem &mem,
                              [[maybe_unused]] void *signal) const {
  return M::Error("UCX was not compiled with GPU device API support");
}

M::ErrorOr<size_t> nixlUcxContext::getGpuSignalSize() const {
  return M::Error("UCX was not compiled with GPU device API support");
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

  status = ucp_worker_set_am_recv_handler(worker.get(), &params);

  if (status != UCS_OK) {
    // TODO: error handling
    return -1;
  }
  return 0;
}

/* ===========================================
 * Data transfer
 * =========================================== */

int nixlUcxWorker::progress() { return ucp_worker_progress(worker.get()); }

nixl_status_t nixlUcxWorker::test(nixlUcxReq req) {
  if (req == nullptr) {
    return nixl_status_t::NIXL_SUCCESS;
  }

  ucp_worker_progress(worker.get());
  return ucx_status_to_nixl(ucp_request_check_status(req));
}

void nixlUcxWorker::reqRelease(nixlUcxReq req) {
  ucp_request_free((void *)req);
}

void nixlUcxWorker::reqCancel(nixlUcxReq req) {
  ucp_request_cancel(worker.get(), req);
}

nixl_status_t nixlUcxWorker::arm() const noexcept {
  return ucx_status_to_nixl(ucp_worker_arm(worker.get()));
}

ErrorOr<int> nixlUcxWorker::getEfd() const {
  int fd;
  const auto status = ucp_worker_get_efd(worker.get(), &fd);
  if (status != UCS_OK) {
    const std::string errorStr =
        (Twine("Couldn't obtain fd for a worker: ") + ucs_status_string(status))
            .str();
    NIXL_ERROR << errorStr;
    return Error(errorStr);
  }
  return fd;
}
