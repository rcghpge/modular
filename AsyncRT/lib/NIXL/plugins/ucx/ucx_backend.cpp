//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// ai-dynamo provides a good description of each of these APIs here:
// https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md#the-south-bound-api
//
// The nixlUcxEngine is NOT thread-safe. You must ensure that you use the
// appropriate locking when using the engine from multiple threads. The
// nixlUcxEngine has its own internal locking due to its optional use of a
// progress thread.
//
// nixlUcxEngine is an implementation of the nixlBackendEngine interface.
// The nixlBackendEngine is used by nixlAgent which has its own locking.

#include "AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/plugins/ucx/ucx_backend.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/serdes/serdes.h"
#include "AsyncRT/NIXL/utils/ucx/ucx_utils.h"
#include "Support/ErrorOr.h"
#include "Support/LogicalResult.h"
#include "Support/STLExtras.h"

#include "AsyncRT/NIXL/utils/common/str_tools.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <string.h>
#include <unistd.h>

using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

static void moveNotifList(notif_list_t &src, notif_list_t &tgt) {
  if (src.size() > 0) {
    std::move(src.begin(), src.end(), std::back_inserter(tgt));
    src.clear();
  }
}

/****************************************
 * Device-specific code
 *****************************************/

#if defined(PLUGIN_UCX_CUDA)

#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"

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
    NIXL_ERROR << "Warning: Unable to identify CUDA context which owns address "
               << address;

  if (deviceCtx && ctx) {
    // Context was already set previously, and does not match new context
    auto myContext = CUDADeviceContext::cast(*deviceCtx).cudaContext();
    if (myContext != ctx) {
      return failure();
    }
  }
  return success();
}

#elif defined(PLUGIN_UCX_ROCM)

#include "AsyncRT/lib/DeviceContext/HIP/HIPDriver.h"

M::LogicalResult nixlUcxEngine::checkDevicePointerContext(void *address,
                                                          int addrDeviceId) {
  /* TODO: proper error codes and log outputs through this method */
  if (addrDeviceId == -1)
    return failure();

  // Incorrect dev id from first registration?
  int myDevId = deviceCtx ? deviceCtx->device().id() : -1;
  if (myDevId != -1 && addrDeviceId != myDevId)
    return failure();

  auto hipDriverOr = HIPDriver::get();
  if (hipDriverOr.isError()) {
    return failure();
  }
  HIPDriver const &hipDriver = hipDriverOr.get();
  hipPointerAttribute_t addressAttr;
  if (HIPDriver::Result::SUCCESS !=
      hipDriver.PointerGetAttributes(&addressAttr, address)) {
    return failure();
  }
  bool const isDevicePointer =
      (addressAttr.type == hipMemoryType::hipMemoryTypeDevice) ||
      (addressAttr.type == hipMemoryType::hipMemoryTypeArray);
  // We don't care about device for host pointers.
  if (!isDevicePointer) {
    return success();
  }

  if (addressAttr.device != addrDeviceId) {
    // User provided address that does not match dev_id
    return failure();
  }

  return success();
}

#else

#error "Must pick one of CUDA or ROCm for UCX GPU transport. Neither picked."

#endif

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

    /* Maximum progress */
    while (uw->progress())
      ;

    /* Go over all request updating their status */
    while (req) {
      nixl_status_t ret;
      if (!req->is_complete()) {
        ret = ucx_status_to_nixl(ucp_request_check_status((nixlUcxReq)req));
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
    NIXL_ERROR << "Error: Failed to activate context for progress thread";
    return;
  }

  progressThreadActive = true;
  progressThreadActiveCV.notify_one();

  // Set timeout event so that the main loop would progress all workers on first
  // iteration
  bool timeout = true;
  bool progressThreadStop = false;
  while (!progressThreadStop) {
    for (size_t wid = 0; wid < pollFds.size() - 1; wid++) {
      if (!(pollFds[wid].revents & POLLIN) && !timeout)
        continue;
      pollFds[wid].revents = 0;

      bool made_progress = false;
      nixl_status_t status;
      const auto &uw = uws[wid];
      do {
        while (uw->progress())
          made_progress = true;

        status = uw->arm();
      } while (status == nixl_status_t::NIXL_IN_PROG);
      assert(status == nixl_status_t::NIXL_SUCCESS);

      if (made_progress && !wid)
        notifProgress();
    }

    // Check active transfers for completion and send out notifs to receiver
    // promptly when they are complete.
    {
      std::lock_guard<std::mutex> lock(inflightSendXfersMtx);
      std::set<nixlBackendReqH *> handlesToDelete;
      for (auto *handle : inflightSendXfers) {
        if (checkXferWithoutLock(handle) == nixl_status_t::NIXL_SUCCESS) {
          handlesToDelete.insert(handle);
        }
      }
      // Remove completed transfers from the set.
      for (auto *handle : handlesToDelete) {
        inflightSendXfers.erase(handle);
      }
    }

    timeout = false;

    int ret;
    while ((ret = poll(pollFds.data(), pollFds.size(), pthrDelay.count())) < 0)
      NIXL_PTRACE << "Call to poll() was interrupted, retrying";

    if (!ret) {
      timeout = true;
    } else if (pollFds.back().revents & POLLIN) {
      pollFds.back().revents = 0;

      char signal;
      int ret = read(pollFds.back().fd, &signal, sizeof(signal));
      if (ret < 0)
        NIXL_PERROR << "read() on control pipe failed";

      progressThreadStop = true;
    }
  }
}

void nixlUcxEngine::progressThreadStart() {
  {
    std::unique_lock<std::mutex> lock(progressThreadActiveLock);
    progressThreadActive = false;
  }
  if (!pthrOn) {
    // not enabled
    return;
  }

  pthr = std::thread(&nixlUcxEngine::progressFunc, this);

  std::unique_lock<std::mutex> lock(progressThreadActiveLock);
  progressThreadActiveCV.wait(lock,
                              [&] { return progressThreadActive.load(); });
}

void nixlUcxEngine::progressThreadStop() {
  if (!pthrOn) {
    // not enabled
    return;
  }

  const char signal = 'X';
  int ret = write(pthrControlPipe[1], &signal, sizeof(signal));
  if (ret < 0)
    NIXL_PERROR << "write to progress thread control pipe failed";
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
    : nixlBackendEngine(init_params), pthrControlPipe{0, 0} {
  size_t numWorkers;
  std::vector<std::string_view> devs;
  nixl_b_params_t *customParams = init_params->customParams;

  if (init_params->enableProgTh) {
    if (!nixlUcxMtLevelIsSupported(nixl_ucx_mt_t::WORKER)) {
      NIXL_ERROR << "UCX library does not support multi-threading";
      initErr = true;
      return;
    }

    if (pipe(pthrControlPipe) < 0) {
      NIXL_PERROR << "Couldn't create progress thread control pipe";
      initErr = true;
      return;
    }

    // This will ensure that the resulting delay is at least 1ms and fits into
    // int in order for it to be compatible with poll()
    pthrDelay =
        std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(
            init_params->pthrDelay < std::numeric_limits<int>::max()
                ? init_params->pthrDelay
                : std::numeric_limits<int>::max()));
    pthrOn = true;
  } else {
    pthrOn = false;
  }

  if (customParams->count("device_list") != 0)
    devs = str_split_substr((*customParams)["device_list"], ", ");

  const auto numWorkersIter = customParams->find("num_workers");
  if (numWorkersIter == customParams->end()) {
    numWorkers = 1;
  } else {
    const char *start = numWorkersIter->second.c_str();
    char *end;
    unsigned long parsed = std::strtoul(start, &end, 10);
    if (end == start || *end != '\0' || parsed == 0) {
      numWorkers = 1;
    } else {
      numWorkers = parsed;
    }
  }

  // If the caller passes a GPU device ID, set up the corresponding device
  // context.
  if (customParams->count("gpu_device_id")) {
    int gpuDeviceId = std::stoi((*customParams)["gpu_device_id"]);
    ErrorOr<DeviceContextRef> errorOrDeviceCtx =
        DeviceContext::create(Device::gpuAPI, gpuDeviceId);
    if (errorOrDeviceCtx.isError()) {
      initErr = true;
      return;
    }
    this->deviceCtx = errorOrDeviceCtx.takeValue();
  }
  // Parse error handling mode
  ucp_err_handling_mode_t errHandlingMode;
  const auto errHandlingModeIt =
      customParams->find(std::string(nixl_ucx_err_handling_param_name));
  if (errHandlingModeIt == customParams->end()) {
    errHandlingMode = UCP_ERR_HANDLING_MODE_PEER;
  } else {
    errHandlingMode = ucx_err_mode_from_string(errHandlingModeIt->second);
  }

  // Set the context as active for the UCX initialization in this function.
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    NIXL_ERROR << "Error: Failed to activate context for UCX initialization";
    return;
  }

  uc = std::make_unique<nixlUcxContext>(
      devs, sizeof(nixlUcxIntReq), _internalRequestInit, _internalRequestFini,
      pthrOn, numWorkers, init_params->syncMode);

  for (size_t i = 0; i < numWorkers; i++) {
    auto workerOrError = nixlUcxWorker::create(*uc, errHandlingMode);
    if (workerOrError.isError()) {
      NIXL_ERROR << "Failed to create UCX worker " << i << ": "
                 << workerOrError.getError();
      initErr = true;
      return;
    }
    uws.emplace_back(workerOrError.takeValue());
  }

  workerAddr = uws.front()->epAddr();

  if (pthrOn) {
    for (auto &uw : uws) {
      auto fdOrError = uw->getEfd();
      if (fdOrError.isError()) {
        NIXL_ERROR << "Failed to get worker fd: " << fdOrError.getError();
        initErr = true;
        return;
      }
      pollFds.push_back({fdOrError.takeValue(), POLLIN, 0});
    }
    pollFds.push_back({pthrControlPipe[0], POLLIN, 0});
  }

  // TODO: in case of UCX error handling is enabled, we can clean up AM based
  // connections error handling, if user requested disabled error handling, we
  // dont care about it.
  auto &uw = uws.front();
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
  progressThreadStop();
  if (pthrOn) {
    for (const auto pthr_control_pipe : pthrControlPipe) {
      if (pthr_control_pipe != 0) {
        close(pthr_control_pipe);
      }
    }
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

  const auto search = remoteConnMap.find(remote_agent);

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

  str = workerAddr;
  return nixl_status_t::NIXL_SUCCESS;
}

ucs_status_t nixlUcxEngine::connectionCheckAmCb(
    void *arg, const void *header, size_t header_length, void *data,
    size_t length, const ucp_am_recv_param_t *param) {
  std::string remote_agent((char *)data, length);
  nixlUcxEngine *engine = (nixlUcxEngine *)arg;

  assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
  assert(header_length == 0);

  if (engine->checkConn(remote_agent) != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR << "Received connect AM from agent we don't recognize: "
               << remote_agent;
    return UCS_OK;
  }

  return UCS_OK;
}

ucs_status_t nixlUcxEngine::connectionTermAmCb(
    void *arg, const void *header, size_t header_length, void *data,
    size_t length, const ucp_am_recv_param_t *param) {
  std::string remote_agent((char *)data, length);

  assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
  assert(header_length == 0);

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

  if (remote_agent == localAgent)
    return loadRemoteConnInfo(remote_agent, workerAddr);

  const auto search = remoteConnMap.find(remote_agent);

  if (search == remoteConnMap.end()) {
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  bool error = false;
  nixl_status_t ret = nixl_status_t::NIXL_SUCCESS;
  std::vector<nixlUcxReq> reqs;
  for (size_t i = 0; i < uws.size(); i++) {
    reqs.emplace_back();
    ret = search->second->getEp(i).sendAm(
        to_underlying(UcxCallbackOp::CONN_CHECK), nullptr, 0,
        (void *)localAgent.data(), localAgent.size(), UCP_AM_SEND_FLAG_EAGER,
        reqs.back());
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

  if (remote_agent != localAgent) {
    const auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }

    nixl_status_t ret = nixl_status_t::NIXL_SUCCESS;
    for (size_t i = 0; i < uws.size(); i++) {
      if (search->second->getEp(i).checkTxState() ==
          nixl_status_t::NIXL_SUCCESS) {
        nixlUcxReq req;
        ret = search->second->getEp(i).sendAm(
            to_underlying(UcxCallbackOp::DISCONNECT), nullptr, 0,
            (void *)localAgent.data(), localAgent.size(),
            UCP_AM_SEND_FLAG_EAGER, req);
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

  std::unique_ptr<nixlUcxPrivateMetadata> priv =
      std::make_unique<nixlUcxPrivateMetadata>();

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
  const int ret = uc->memReg((void *)mem.addr, mem.len, priv->mem, nixl_mem);
  if (ret) {
    return nixl_status_t::NIXL_ERR_BACKEND;
  }
  priv->rkeyStr = uc->packRkey(priv->mem);

  if (priv->rkeyStr.empty()) {
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

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

  const auto search = remoteConnMap.find(agent);
  if (search == remoteConnMap.end()) {
    // TODO: err: remote connection not found
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }
  md->conn = search->second;

  std::vector<char> addr(size);
  // Create a temporary string from blob_ref for nixlSerDes
  std::string blobStr(static_cast<const char *>(blob.data), blob.size);
  nixlSerDes::_stringToBytes(addr.data(), blobStr, size);

  for (size_t wid = 0; wid < uws.size(); wid++) {
    nixl_status_t status = md->addRkey(md->conn->getEp(wid), addr.data());
    if (status != nixl_status_t::NIXL_SUCCESS) {
      return nixl_status_t::NIXL_ERR_BACKEND;
    }
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

  // Track active handle for progress thread monitoring
  {
    std::lock_guard<std::mutex> lock(inflightSendXfersMtx);
    inflightSendXfers.insert((nixlBackendReqH *)intHandle);
  }

  return ret;
}

// This is a private method. Use checkXfer() instead unless you know for sure
// that you do not want to grab the lock.
nixl_status_t
nixlUcxEngine::checkXferWithoutLock(nixlBackendReqH *handle) const {
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
    // Sends notifs to receiver when transfer is complete.
    status = notifSendPriv(notif->agent, notif->payload, req, workerId);
    notif.reset();
    if (_retHelper(status, intHandle, req) != nixl_status_t::NIXL_SUCCESS) {
      return status;
    }

    status = intHandle->status();
  }

  return status;
}

nixl_status_t nixlUcxEngine::checkXfer(nixlBackendReqH *handle) const {
  std::lock_guard<std::mutex> lock(inflightSendXfersMtx);
  return checkXferWithoutLock(handle);
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH *handle) const {
  auto deviceCtxScope = activateContext();
  if (deviceCtxScope && deviceCtxScope->err().isError()) {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }

  nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
  nixl_status_t status = intHandle->release();

  // Remove from active handles
  {
    std::lock_guard<std::mutex> lock(inflightSendXfersMtx);
    inflightSendXfers.erase(intHandle);
  }

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
  nixl_status_t ret;

  const auto search = remoteConnMap.find(remote_agent);

  if (search == remoteConnMap.end()) {
    // TODO: err: remote connection not found
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  ser_des.addStr("name", localAgent);
  ser_des.addStr("msg", msg);
  // TODO: replace with mpool for performance
  std::string buffer = ser_des.exportStr();

  ret = search->second->getEp(worker_id).sendAm(
      to_underlying(UcxCallbackOp::NOTIF_STR), nullptr, 0,
      (void *)buffer.data(), buffer.size(), UCP_AM_SEND_FLAG_EAGER, req);

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
  nixlSerDes ser_des;

  nixlUcxEngine *engine = (nixlUcxEngine *)arg;

  // send_am should be forcing EAGER protocol
  assert(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
  assert(header_length == 0);

  ser_des.importBlob(blob_ref{data, length});
  std::string remote_name = ser_des.getStr("name");
  std::string msg = ser_des.getStr("msg");

  if (engine->isProgressThread()) {
    /* Append to the private list to allow batching */
    engine->notifPthrPriv.push_back(
        std::make_pair(std::move(remote_name), std::move(msg)));
  } else {
    engine->notifMainList.push_back(
        std::make_pair(std::move(remote_name), std::move(msg)));
  }

  return UCS_OK;
}

void nixlUcxEngine::notifProgressCombineHelper(notif_list_t &src,
                                               notif_list_t &tgt) {
  const std::lock_guard<std::mutex> lock(notifMtx);
  moveNotifList(src, tgt);
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

  moveNotifList(notifMainList, notif_list);
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
