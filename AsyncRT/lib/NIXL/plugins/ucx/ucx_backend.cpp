//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/serdes/serdes.h"
#include "AsyncRT/NIXL/utils/ucx/ucx_utils.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"
#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "Support/STLExtras.h"

#include "AsyncRT/NIXL/utils/common/str_tools.h"
#include <optional>
#include <string.h>
#include <unistd.h>

using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

/****************************************
 * CUDA related code
 *****************************************/

static int cudaQueryAddr(void *address, bool &is_dev, CUdevice &dev,
                         CUcontext &ctx) {
  CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
  uint32_t is_managed = 0;
#define NUM_ATTRS 4
  CUpointer_attribute attr_type[NUM_ATTRS];
  void *attr_data[NUM_ATTRS];
  CUDADriver::Result result;

  attr_type[0] = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
  attr_data[0] = &mem_type;
  attr_type[1] = CU_POINTER_ATTRIBUTE_IS_MANAGED;
  attr_data[1] = &is_managed;
  attr_type[2] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
  attr_data[2] = &dev;
  attr_type[3] = CU_POINTER_ATTRIBUTE_CONTEXT;
  attr_data[3] = &ctx;

  const CUDADriver &driver = CUDADriver::get().get();
  result = driver.PointerGetAttributes(4, attr_type, attr_data,
                                       (CUdeviceptr)address);

  is_dev = (mem_type == CU_MEMORYTYPE_DEVICE);

  return (CUDADriver::Result::SUCCESS != result);
}

// Check if the device pointer is associated with the expected device.
M::LogicalResult nixlUcxEngine::checkDevicePointerContext(void *address,
                                                          int addrDeviceId) {

  /* TODO: proper error codes and log outputs through this method */
  if (addrDeviceId == -1)
    return failure();

  // Incorrect dev id from first registration?
  int myDevId = deviceCtx ? deviceCtx->device().id() : -1;
  if (myDevId != -1 && addrDeviceId != myDevId)
    return failure();

  bool isDevicePointer;
  CUdevice dev;
  CUcontext ctx;
  // Call failed?
  if (0 != cudaQueryAddr(address, isDevicePointer, dev, ctx))
    return failure();

  // We don't care about device for host pointers.
  if (!isDevicePointer)
    return success();

  if (dev != addrDeviceId) {
    // User provided address that does not match dev_id
    return failure();
  }

  // For a mysterious reason, sometimes the queried context for a device pointer
  // is null. However, the program appears to continue without issue.
  if (!ctx)
    std::cerr << "Warning: Unable to identify CUDA context which owns address "
              << std::hex << address << std::endl;

  if (deviceCtx && ctx) {
    // Context was already set previously, and does not match new context
    auto myContext = CUDADeviceContext::cast(*deviceCtx).cudaContext();
    if (myContext != ctx) {
      return failure();
    }
  }
  return success();
}

/****************************************
 * UCX request management
 *****************************************/

class nixlUcxIntReq : public nixlLinkElem<nixlUcxIntReq> {
private:
  int _completed;

public:
  std::string amBuffer;

  nixlUcxIntReq() : nixlLinkElem(), _completed{0} {}

  bool is_complete() const { return _completed; }
  void completed() { _completed = 1; }
};

static void _internalRequestInit(void *request) {
  /* Initialize request in-place (aka "placement new")*/
  new (request) nixlUcxIntReq;
}

static void _internalRequestFini(void *request) {
  /* Finalize request */
  nixlUcxIntReq *req = (nixlUcxIntReq *)request;
  req->~nixlUcxIntReq();
}

static void _internalRequestReset(nixlUcxIntReq *req) {
  _internalRequestFini((void *)req);
  _internalRequestInit((void *)req);
}

/****************************************
 * Backend request management
 *****************************************/

class nixlUcxBackendH : public nixlBackendReqH {
private:
  nixlUcxIntReq head;
  const nixlUcxEngine &eng;
  size_t worker_id;

  // Notification to be sent after completion of all requests.
  struct Notif {
    std::string agent;
    nixl_blob_t payload;
  };
  std::optional<Notif> notif;

public:
  std::optional<Notif> &notification() { return notif; }

  nixlUcxBackendH(const nixlUcxEngine &eng_, size_t worker_id_)
      : eng(eng_), worker_id(worker_id_) {}

  void append(nixlUcxIntReq *req) { head.link(req); }

  nixl_status_t release() {
    nixlUcxIntReq *req = head.next();

    if (!req)
      return nixl_status_t::NIXL_SUCCESS;

    const auto &uw = eng.getWorker(worker_id);
    // TODO: Error log: uncompleted requests found! Cancelling ...
    while (req) {
      nixlUcxIntReq *cur = req;
      bool done = cur->is_complete();
      req = cur->unlink();
      if (!done) {
        // TODO: Need process this properly.
        // it may not be enough to cancel UCX request
        uw->reqCancel((nixlUcxReq)cur);
      }
      _internalRequestReset(cur);
      uw->reqRelease((nixlUcxReq)cur);
    }
    return nixl_status_t::NIXL_SUCCESS;
  }

  nixl_status_t status() {
    nixlUcxIntReq *req = head.next();
    nixl_status_t out_ret = nixl_status_t::NIXL_SUCCESS;

    if (req == nullptr) {
      /* No pending transmissions */
      return nixl_status_t::NIXL_SUCCESS;
    }

    const auto &uw = eng.getWorker(worker_id);
    /* Go over all request updating their status */
    while (req) {
      nixl_status_t ret;
      if (!req->is_complete()) {
        ret = uw->test((nixlUcxReq)req);
        switch (ret) {
        case nixl_status_t::NIXL_SUCCESS:
          /* Mark as completed */
          req->completed();
          break;
        case nixl_status_t::NIXL_IN_PROG:
          out_ret = nixl_status_t::NIXL_IN_PROG;
          break;
        default:
          /* Any other ret value is ERR and will be returned */
          return ret;
        }
      }
      req = req->next();
    }

    /* Remove completed requests keeping the first one as
    request representative */
    req = head.unlink();
    while (req) {
      nixlUcxIntReq *next_req = req->unlink();
      if (req->is_complete()) {
        _internalRequestReset(req);
        uw->reqRelease((nixlUcxReq)req);
      } else {
        /* Enqueue back */
        append(req);
      }
      req = next_req;
    }

    return out_ret;
  }

  size_t getWorkerId() const { return worker_id; }
};

/****************************************
 * Progress thread management
 *****************************************/

void nixlUcxEngine::progressFunc() {
  // Set active device context for the progress thread.
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    std::cerr << "Error: Failed to activate context for progress thread"
              << std::endl;
    return;
  }

  {
    std::unique_lock<std::mutex> lock(pthrActiveLock);
    pthrActive = true;
  }
  pthrActiveCV.notify_one();

  // Set POLLIN event on all worker fds so that the main loop would process them
  // all on first iteration
  for (size_t wid = 0; wid < pollFds.size() - 1; wid++)
    pollFds[wid].revents = POLLIN;

  bool pthrStop = false;
  while (!pthrStop) {
    for (size_t wid = 0; wid < pollFds.size() - 1; wid++) {
      if (!(pollFds[wid].revents & POLLIN))
        continue;
      pollFds[wid].revents = 0;

      bool made_progress = false;
      ucs_status_t status = UCS_INPROGRESS;
      const auto &uw = uws[wid];
      do {
        while (uw->progress())
          made_progress = true;

        status = ucp_worker_arm(uw->getWorker());
      } while (status == UCS_ERR_BUSY);
      assert(status == UCS_OK);

      if (made_progress && !wid)
        notifProgress();
    }

    while (poll(pollFds.data(), pollFds.size(), -1) <= 0)
      NIXL_ERROR << "Call to poll() was interrupted, retrying. Error: "
                 << strerror(errno) << "\n";

    if (pollFds.back().revents & POLLIN) {
      pollFds.back().revents = 0;

      char signal;
      int ret = read(pollFds.back().fd, &signal, sizeof(signal));
      if (ret < 0)
        NIXL_ERROR << "read() on control pipe failed. Error: "
                   << strerror(errno) << "\n";

      pthrStop = true;
    }
  }
}

void nixlUcxEngine::progressThreadStart() {
  {
    std::unique_lock<std::mutex> lock(pthrActiveLock);
    pthrActive = false;
  }
  if (!pthrOn) {
    // not enabled
    return;
  }

  pthr = std::thread(&nixlUcxEngine::progressFunc, this);

  std::unique_lock<std::mutex> lock(pthrActiveLock);
  pthrActiveCV.wait(lock, [&] { return pthrActive.load(); });
}

void nixlUcxEngine::progressThreadStop() {
  if (!pthrOn) {
    // not enabled
    return;
  }

  const char signal = 'X';
  int ret = write(pthrControlPipe[1], &signal, sizeof(signal));
  if (ret < 0)
    NIXL_ERROR << "write to progress thread control pipe failed, error: "
               << strerror(errno) << "\n";
  pthr.join();
}

void nixlUcxEngine::progressThreadRestart() {
  progressThreadStop();
  progressThreadStart();
}

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
  unsigned long numWorkers;
  std::vector<std::string_view> devs;
  nixl_b_params_t *custom_params = init_params->customParams;

  if (init_params->enableProgTh) {
    pthrOn = true;
    if (!nixlUcxContext::mtLevelIsSupported(NIXL_UCX_MT_WORKER)) {
      NIXL_ERROR << "UCX library does not support multi-threading\n";
      initErr = true;
      return;
    }
    if (pipe(pthrControlPipe) < 0) {
      NIXL_ERROR << "Couldn't create progress thread control pipe, error: "
                 << strerror(errno) << "\n";
      initErr = true;
      return;
    }
  } else {
    pthrOn = false;
  }

  if (custom_params->count("device_list") != 0)
    devs = str_split_substr((*custom_params)["device_list"], ", ");

  const auto num_workers_iter = custom_params->find("num_workers");
  if (num_workers_iter == custom_params->end()) {
    numWorkers = 1;
  } else {
    const char *start = num_workers_iter->second.c_str();
    char *end;
    numWorkers = std::strtoul(start, &end, 10);
    if (end == start || *end != '\0' || numWorkers == 0) {
      numWorkers = 1;
    }
  }

  // If the caller passes a GPU device ID, set up the corresponding device
  // context.
  if (custom_params->count("gpu_device_id")) {
    int gpuDeviceId = std::stoi((*custom_params)["gpu_device_id"]);
    ErrorOr<DeviceContextRef> errorOrDeviceCtx =
        DeviceContext::create(Device::gpuAPI, gpuDeviceId);
    if (errorOrDeviceCtx.isError()) {
      initErr = true;
      return;
    }
    this->deviceCtx = errorOrDeviceCtx.takeValue();
  }
  // Set the context as active for the UCX initialization in this function.
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    std::cerr << "Error: Failed to activate context for UCX initialization"
              << std::endl;
    return;
  }

  uc = std::make_shared<nixlUcxContext>(
      devs, sizeof(nixlUcxIntReq), _internalRequestInit, _internalRequestFini,
      NIXL_UCX_MT_WORKER, pthrOn);
  for (unsigned int i = 0; i < numWorkers; i++)
    uws.emplace_back(std::make_unique<nixlUcxWorker>(uc));

  const auto &uw = uws.front();
  workerAddr = uw->epAddr(workerSize);

  if (workerAddr == nullptr) {
    NIXL_ERROR << "Failed to get UCX worker address\n";
    initErr = true;
    return;
  }

  if (pthrOn) {
    for (auto &uw : uws) {
      int fd;
      ucs_status_t ret = ucp_worker_get_efd(uw->getWorker(), &fd);
      if (ret != UCS_OK) {
        NIXL_ERROR << "Couldn't obtain fd for a worker, status: "
                   << ucs_status_string(ret) << "\n";
        initErr = true;
        return;
      }

      pollFds.push_back({fd, POLLIN, 0});
    }
    pollFds.push_back({pthrControlPipe[0], POLLIN, 0});
  }

  uw->regAmCallback(UcxCallbackOp::CONN_CHECK, connectionCheckAmCb, this);
  uw->regAmCallback(UcxCallbackOp::DISCONNECT, connectionTermAmCb, this);
  uw->regAmCallback(UcxCallbackOp::NOTIF_STR, notifAmCb, this);

  progressThreadStart();
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems() const {
  nixl_mem_list_t mems;
  mems.push_back(nixl_mem_t::DRAM_SEG);
  mems.push_back(nixl_mem_t::VRAM_SEG);
  return mems;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
  // per registered memory deregisters it, which removes the corresponding
  // metadata too parent destructor takes care of the desc list For remote
  // metadata, they should be removed here
  if (initErr) {
    // Nothing to do
    return;
  }

  progressThreadStop();
  if (pthrOn) {
    close(pthrControlPipe[0]);
    close(pthrControlPipe[1]);
  }
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  return remoteConnMap.count(remote_agent) ? nixl_status_t::NIXL_SUCCESS
                                           : nixl_status_t::NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::endConn(const std::string &remote_agent) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  auto search = remoteConnMap.find(remote_agent);

  if (search == remoteConnMap.end())
    return nixl_status_t::NIXL_ERR_NOT_FOUND;

  // thread safety?
  remoteConnMap.erase(search);

  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  str = nixlSerDes::_bytesToString(workerAddr.get(), workerSize);
  return nixl_status_t::NIXL_SUCCESS;
}

ucs_status_t nixlUcxEngine::connectionCheckAmCb(
    void *arg, const void *header, size_t header_length, void *data,
    size_t length, const ucp_am_recv_param_t *param) {
  const struct nixl_ucx_am_hdr *hdr = (const struct nixl_ucx_am_hdr *)header;

  std::string remote_agent((char *)data, length);
  nixlUcxEngine *engine = (nixlUcxEngine *)arg;

  if (hdr->op != UcxCallbackOp::CONN_CHECK) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }

  // send_am should be forcing EAGER protocol
  if ((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }

  if (engine->checkConn(remote_agent) != nixl_status_t::NIXL_SUCCESS) {
    // TODO: received connect AM from agent we don't recognize
    return UCS_ERR_INVALID_PARAM;
  }

  return UCS_OK;
}

ucs_status_t nixlUcxEngine::connectionTermAmCb(
    void *arg, const void *header, size_t header_length, void *data,
    size_t length, const ucp_am_recv_param_t *param) {
  const struct nixl_ucx_am_hdr *hdr = (const struct nixl_ucx_am_hdr *)header;

  std::string remote_agent((char *)data, length);

  if (hdr->op != UcxCallbackOp::DISCONNECT) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }

  // send_am should be forcing EAGER protocol
  if ((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }
  /*
      // TODO: research UCX connection logic and fix.
      nixlUcxEngine* engine = (nixlUcxEngine*) arg;
      if(NIXL_SUCCESS != engine->endConn(remote_agent)) {
          //TODO: received connect AM from agent we don't recognize
          return UCS_ERR_INVALID_PARAM;
      }
  */
  return UCS_OK;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  struct nixl_ucx_am_hdr hdr;
  uint32_t flags = 0;

  if (remote_agent == localAgent)
    return loadRemoteConnInfo(
        remote_agent, nixlSerDes::_bytesToString(workerAddr.get(), workerSize));

  auto search = remoteConnMap.find(remote_agent);

  if (search == remoteConnMap.end()) {
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  hdr.op = UcxCallbackOp::CONN_CHECK;
  // agent names should never be long enough to need RNDV
  flags |= UCP_AM_SEND_FLAG_EAGER;

  bool error = false;
  nixl_status_t ret = nixl_status_t::NIXL_SUCCESS;
  std::vector<nixlUcxReq> reqs;
  for (size_t i = 0; i < uws.size(); i++) {
    reqs.emplace_back();
    ret = search->second->getEp(i).sendAm(
        to_underlying(UcxCallbackOp::CONN_CHECK), &hdr,
        sizeof(struct nixl_ucx_am_hdr), (void *)localAgent.data(),
        localAgent.size(), flags, reqs.back());
    if (isErrorStatus(ret)) {
      error = true;
      break;
    }
  }

  // wait for AM to send
  ret = nixl_status_t::NIXL_IN_PROG;
  for (size_t i = 0; i < reqs.size(); i++)
    while (ret == nixl_status_t::NIXL_IN_PROG)
      ret = getWorker(i)->test(reqs[i]);

  return error ? nixl_status_t::NIXL_ERR_BACKEND : nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  static struct nixl_ucx_am_hdr hdr;
  uint32_t flags = 0;

  if (remote_agent != localAgent) {
    auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }

    nixl_status_t ret = nixl_status_t::NIXL_SUCCESS;
    for (size_t i = 0; i < uws.size(); i++) {
      if (search->second->getEp(i).checkTxState() ==
          nixl_status_t::NIXL_SUCCESS) {
        hdr.op = UcxCallbackOp::DISCONNECT;
        // agent names should never be long enough to need RNDV
        flags |= UCP_AM_SEND_FLAG_EAGER;

        nixlUcxReq req;
        ret = search->second->getEp(i).sendAm(
            to_underlying(UcxCallbackOp::DISCONNECT), &hdr,
            sizeof(struct nixl_ucx_am_hdr), (void *)localAgent.data(),
            localAgent.size(), flags, req);
        // don't care
        if (ret == nixl_status_t::NIXL_IN_PROG)
          getWorker(i)->reqRelease(req);
      }
    }
  }

  endConn(remote_agent);

  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                  const std::string &remote_conn_info) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  size_t size = remote_conn_info.size();
  std::vector<char> addr(size);
  nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);

  auto conn = std::make_shared<nixlUcxConnection>();
  bool error = false;
  for (auto &uw : uws) {
    auto result = uw->connect(addr.data(), size);
    if (result.isError()) {
      error = true;
      break;
    }
    conn->eps.push_back(result.takeValue());
  }

  if (error)
    return nixl_status_t::NIXL_ERR_BACKEND;

  conn->remoteAgent = remote_agent;

  auto [iter, inserted] = remoteConnMap.emplace(remote_agent, conn);
  if (!inserted)
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;

  return nixl_status_t::NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/
nixl_status_t nixlUcxEngine::registerMem(const nixlBlobDesc &mem,
                                         const nixl_mem_t &nixl_mem,
                                         nixlBackendMD *&out) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  int ret;
  auto priv = std::make_unique<nixlUcxPrivateMetadata>();
  size_t rkey_size;

  if (nixl_mem == nixl_mem_t::VRAM_SEG) {
    // This call checks if the device memory we are registering is associated
    // with our device context.
    // With tighter integration between NIXL and AsyncRT DeviceContext, we could
    // simplify or eliminate this check, since it is trivial to get the context
    // associated with a DeviceBuffer. But since we only have a raw
    // pointer here, we must rely on a CUDA call to recover the associated
    // context.
    if (failed(checkDevicePointerContext((void *)mem.addr, mem.devId))) {
      return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
      // TODO Add to logging
    }
  }

  // TODO: Add nixl_mem check?
  ret = uc->memReg((void *)mem.addr, mem.len, priv->mem);
  if (ret) {
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
  auto rkey = uc->packRkey(priv->mem, rkey_size);
  if (rkey == nullptr) {
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
  priv->rkeyStr = nixlSerDes::_bytesToString(rkey.get(), rkey_size);

  out = priv.release();
  return nixl_status_t::NIXL_SUCCESS; // Or errors
}

nixl_status_t nixlUcxEngine::deregisterMem(nixlBackendMD *meta) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata *)meta;
  uc->memDereg(priv->mem);
  delete priv;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData(const nixlBackendMD *meta,
                                           std::string &str) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  const nixlUcxPrivateMetadata *priv = (const nixlUcxPrivateMetadata *)meta;
  str = priv->get();
  return nixl_status_t::NIXL_SUCCESS;
}

// To be cleaned up
nixl_status_t nixlUcxEngine::internalMDHelper(const blob_ref blob,
                                              const std::string &agent,
                                              nixlBackendMD *&output) {
  auto md = std::make_unique<nixlUcxPublicMetadata>();
  size_t size = blob.size;

  auto search = remoteConnMap.find(agent);
  if (search == remoteConnMap.end())
    return nixl_status_t::NIXL_ERR_NOT_FOUND;

  md->conn = search->second;

  std::vector<char> addr(size);
  // Create a temporary string from blob_ref for nixlSerDes
  std::string blobStr(static_cast<const char *>(blob.data), blob.size);
  nixlSerDes::_stringToBytes(addr.data(), blobStr, size);

  bool error = false;
  for (size_t wid = 0; wid < uws.size(); wid++) {
    nixlUcxRkey rkey;
    error = md->conn->getEp(wid).rkeyImport(addr.data(), size, rkey);
    if (error)
      break;
    md->rkeys.push_back(rkey);
  }
  if (error) {
    for (size_t wid = 0; wid < md->rkeys.size(); wid++)
      md->conn->getEp(wid).rkeyDestroy(md->rkeys[wid]);
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  output = (nixlBackendMD *)md.release();

  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadLocalMD(nixlBackendMD *input,
                                         nixlBackendMD *&output) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxPrivateMetadata *input_md = (nixlUcxPrivateMetadata *)input;
  return internalMDHelper(blob_ref{input_md->rkeyStr}, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD(const nixlBlobDesc &input,
                                          const nixl_mem_t &nixl_mem,
                                          const std::string &remote_agent,
                                          nixlBackendMD *&output) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  return internalMDHelper(blob_ref{input.metaInfo}, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD(nixlBackendMD *input) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata *)input; // typecast?

  for (size_t wid = 0; wid < md->rkeys.size(); wid++)
    md->conn->getEp(wid).rkeyDestroy(md->rkeys[wid]);
  delete md;

  return nixl_status_t::NIXL_SUCCESS;
}

/****************************************
 * Data movement
 *****************************************/

static nixl_status_t _retHelper(nixl_status_t ret, nixlUcxBackendH *hndl,
                                nixlUcxReq &req) {
  /* if transfer wasn't immediately completed */
  switch (ret) {
  case nixl_status_t::NIXL_IN_PROG:
    hndl->append((nixlUcxIntReq *)req);
    break;
  case nixl_status_t::NIXL_SUCCESS:
    // Nothing to do
    break;
  default:
    // Error. Release all previously initiated ops and exit:
    hndl->release();
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::prepXfer(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH *&handle,
                                      const nixl_opt_b_args_t *opt_args) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  /* TODO: try to get from a pool first */
  nixlUcxBackendH *intHandle = new nixlUcxBackendH(*this, getWorkerId());

  handle = (nixlBackendReqH *)intHandle;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                                      const nixl_meta_dlist_t &local,
                                      const nixl_meta_dlist_t &remote,
                                      const std::string &remote_agent,
                                      nixlBackendReqH *&handle,
                                      const nixl_opt_b_args_t *opt_args) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  size_t lcnt = local.descCount();
  size_t rcnt = remote.descCount();
  size_t i;
  nixl_status_t ret;
  nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
  nixlUcxPrivateMetadata *lmd;
  nixlUcxPublicMetadata *rmd;
  nixlUcxReq req;
  size_t workerId = intHandle->getWorkerId();

  if (lcnt != rcnt) {
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  for (i = 0; i < lcnt; i++) {
    void *laddr = (void *)local[i].addr;
    size_t lsize = local[i].len;
    void *raddr = (void *)remote[i].addr;
    size_t rsize = remote[i].len;

    lmd = (nixlUcxPrivateMetadata *)local[i].metadataP;
    rmd = (nixlUcxPublicMetadata *)remote[i].metadataP;

    if (lsize != rsize) {
      return nixl_status_t::NIXL_ERR_INVALID_PARAM;
    }

    // TODO: remote_agent and msg should be cached in nixlUCxReq or another way

    switch (operation) {
    case nixl_xfer_op_t::NIXL_READ:
      ret = rmd->conn->getEp(workerId).read(
          (uint64_t)raddr, rmd->getRkey(workerId), laddr, lmd->mem, lsize, req);
      break;
    case nixl_xfer_op_t::NIXL_WRITE:
      ret = rmd->conn->getEp(workerId).write(
          laddr, lmd->mem, (uint64_t)raddr, rmd->getRkey(workerId), lsize, req);
      break;
    }

    if (_retHelper(ret, intHandle, req) != nixl_status_t::NIXL_SUCCESS) {
      return ret;
    }
  }

  /*
   * Flush keeps intHandle non-empty until the operation is actually
   * completed, which can happen after local requests completion.
   */
  rmd = (nixlUcxPublicMetadata *)remote[0].metadataP;
  ret = rmd->conn->getEp(workerId).flushEp(req);
  if (_retHelper(ret, intHandle, req) != nixl_status_t::NIXL_SUCCESS)
    return ret;

  ret = intHandle->status();
  if (opt_args && opt_args->hasNotif) {
    if (ret == nixl_status_t::NIXL_SUCCESS) {
      ret = notifSendPriv(remote_agent, opt_args->notifMsg, req, workerId);
      if (_retHelper(ret, intHandle, req) != nixl_status_t::NIXL_SUCCESS)
        return ret;

      ret = intHandle->status();
    } else if (ret == nixl_status_t::NIXL_IN_PROG) {
      intHandle->notification().emplace(remote_agent, opt_args->notifMsg);
    }
  }

  return ret;
}

nixl_status_t nixlUcxEngine::checkXfer(nixlBackendReqH *handle) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
  size_t workerId = intHandle->getWorkerId();

  nixl_status_t status = intHandle->status();
  auto &notif = intHandle->notification();
  if (status == nixl_status_t::NIXL_SUCCESS && notif.has_value()) {
    nixlUcxReq req;
    status = notifSendPriv(notif->agent, notif->payload, req, workerId);
    notif.reset();
    if (_retHelper(status, intHandle, req) != nixl_status_t::NIXL_SUCCESS) {
      return status;
    }

    status = intHandle->status();
  }

  return status;
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH *handle) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
  nixl_status_t status = intHandle->release();

  /* TODO: return to a pool instead. */
  delete intHandle;

  return status;
}

int nixlUcxEngine::progress() {
  // TODO: add listen for connection handling if necessary
  int ret = 0;
  for (auto &uw : uws)
    ret += uw->progress();
  return ret;
}

/****************************************
 * Notifications
 *****************************************/

// agent will provide cached msg
nixl_status_t nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                                           const std::string_view msg,
                                           nixlUcxReq &req,
                                           size_t worker_id) const {
  nixlSerDes ser_des;
  // TODO - temp fix, need to have an mpool
  static struct nixl_ucx_am_hdr hdr;
  uint32_t flags = 0;
  nixl_status_t ret;

  auto search = remoteConnMap.find(remote_agent);

  if (search == remoteConnMap.end()) {
    // TODO: err: remote connection not found
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  hdr.op = UcxCallbackOp::NOTIF_STR;
  flags |= UCP_AM_SEND_FLAG_EAGER;

  ser_des.addStr("name", localAgent);
  ser_des.addStr("msg", msg);
  // TODO: replace with mpool for performance
  std::string buffer = ser_des.exportStr();

  ret = search->second->getEp(worker_id).sendAm(
      to_underlying(UcxCallbackOp::NOTIF_STR), &hdr,
      sizeof(struct nixl_ucx_am_hdr), (void *)buffer.data(), buffer.size(),
      flags, req);

  if (ret == nixl_status_t::NIXL_IN_PROG) {
    nixlUcxIntReq *nReq = (nixlUcxIntReq *)req;
    nReq->amBuffer = std::move(buffer);
  }
  return ret;
}

ucs_status_t nixlUcxEngine::notifAmCb(void *arg, const void *header,
                                      size_t header_length, void *data,
                                      size_t length,
                                      const ucp_am_recv_param_t *param) {
  const struct nixl_ucx_am_hdr *hdr = (const struct nixl_ucx_am_hdr *)header;
  nixlSerDes ser_des;

  nixlUcxEngine *engine = (nixlUcxEngine *)arg;
  std::string remote_name, msg;

  if (hdr->op != UcxCallbackOp::NOTIF_STR) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }

  // send_am should be forcing EAGER protocol
  if ((param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) != 0) {
    // is this the best way to ERR?
    return UCS_ERR_INVALID_PARAM;
  }

  ser_des.importBlob(blob_ref{data, length});
  remote_name = ser_des.getStr("name");
  msg = ser_des.getStr("msg");

  if (engine->isProgressThread()) {
    /* Append to the private list to allow batching */
    engine->notifPthrPriv.push_back(std::make_pair(remote_name, msg));
  } else {
    engine->notifMainList.push_back(std::make_pair(remote_name, msg));
  }

  return UCS_OK;
}

void nixlUcxEngine::notifCombineHelper(notif_list_t &src, notif_list_t &tgt) {
  if (!src.size()) {
    // Nothing to do. Exit
    return;
  }

  move(src.begin(), src.end(), back_inserter(tgt));
  src.erase(src.begin(), src.end());
}

void nixlUcxEngine::notifProgressCombineHelper(notif_list_t &src,
                                               notif_list_t &tgt) {
  notifMtx.lock();

  if (src.size()) {
    move(src.begin(), src.end(), back_inserter(tgt));
    src.erase(src.begin(), src.end());
  }

  notifMtx.unlock();
}

void nixlUcxEngine::notifProgress() {
  notifProgressCombineHelper(notifPthrPriv, notifPthr);
}

nixl_status_t nixlUcxEngine::getNotifs(notif_list_t &notif_list) {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  if (notif_list.size() != 0)
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;

  if (!pthrOn)
    while (progress())
      ;

  notifCombineHelper(notifMainList, notif_list);
  notifProgressCombineHelper(notifPthr, notif_list);

  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::genNotif(const std::string &remote_agent,
                                      const std::string_view msg) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixl_status_t ret;
  nixlUcxReq req;
  size_t wid = getWorkerId();

  ret = notifSendPriv(remote_agent, msg, req, wid);

  switch (ret) {
  case nixl_status_t::NIXL_IN_PROG:
    /* do not track the request */
    getWorker(wid)->reqRelease(req);
    break;
  case nixl_status_t::NIXL_SUCCESS:
    break;
  default:
    /* error case */
    return ret;
  }
  return nixl_status_t::NIXL_SUCCESS;
}
