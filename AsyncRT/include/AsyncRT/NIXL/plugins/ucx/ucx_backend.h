//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_PLUGINS_UCX_UCX_BACKEND_H
#define ASYNCRT_NIXL_PLUGINS_UCX_UCX_BACKEND_H

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/NIXL/backend/backend_engine.h"
#include "AsyncRT/NIXL/backend/backend_plugin.h"
#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/utils/common/list_elem.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/common/nixl_time.h"
#include "AsyncRT/NIXL/utils/common/str_tools.h"
#include "AsyncRT/NIXL/utils/ucx/rkey.h"
#include "AsyncRT/NIXL/utils/ucx/ucx_utils.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <poll.h>
#include <set>
#include <string_view>
#include <thread>
#include <vector>

namespace M::AsyncRT::NIXL {

class nixlUcxConnection : public nixlBackendConnMD {
private:
  std::string remoteAgent;
  std::vector<std::unique_ptr<nixlUcxEp>> eps;

public:
  [[nodiscard]] nixlUcxEp &getEp(size_t ep_id) const { return *eps[ep_id]; }
  // Extra information required for UCX connections

  friend class nixlUcxEngine;
};

using ucx_connection_ptr_t = std::shared_ptr<nixlUcxConnection>;

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
private:
  nixlUcxMem mem;
  nixl_blob_t rkeyStr;

public:
  nixlUcxPrivateMetadata() : nixlBackendMD(true) {}

  ~nixlUcxPrivateMetadata() override {}

  [[nodiscard]] const std::string &get() const { return rkeyStr; }

  friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {
public:
  nixlUcxPublicMetadata() : nixlBackendMD(false) {}

  [[nodiscard]] const UCX::Rkey &getRkey(size_t id) const {
    return *rkeys_[id];
  }

  nixl_status_t addRkey(const nixlUcxEp &ep, const void *rkeyBuffer) {
    auto rkeyOrError = UCX::Rkey::create(ep, rkeyBuffer);
    if (rkeyOrError.isError()) {
      NIXL_ERROR << "Failed to create rkey: " << rkeyOrError.getError();
      return nixl_status_t::NIXL_ERR_BACKEND;
    }
    rkeys_.emplace_back(rkeyOrError.takeValue());
    return nixl_status_t::NIXL_SUCCESS;
  }

  ucx_connection_ptr_t conn;

private:
  std::vector<std::unique_ptr<UCX::Rkey>> rkeys_;
};

class nixlUcxEngine : public nixlBackendEngine {
public:
  static std::unique_ptr<nixlUcxEngine>
  create(const nixlBackendInitParams &init_params);

  ~nixlUcxEngine();

  bool supportsRemote() const override { return true; }
  bool supportsLocal() const override { return true; }
  bool supportsNotif() const override { return true; }
  virtual bool supportsProgTh() const override { return false; }

  nixl_mem_list_t getSupportedMems() const override;

  /* Object management */
  nixl_status_t getPublicData(const nixlBackendMD *meta,
                              std::string &str) const override;
  nixl_status_t getConnInfo(std::string &str) const override;

  /// Establishes a UCX endpoint to a remote agent and caches the connection
  /// metadata.
  nixl_status_t
  loadRemoteConnInfo(const std::string &remote_agent,
                     const std::string &remote_conn_info) override;

  nixl_status_t connect(const std::string &remote_agent) override;
  nixl_status_t disconnect(const std::string &remote_agent) override;

  nixl_status_t registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) override;
  nixl_status_t deregisterMem(nixlBackendMD *meta) override;

  nixl_status_t loadLocalMD(nixlBackendMD *input,
                            nixlBackendMD *&output) override;

  nixl_status_t loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) override;
  nixl_status_t unloadMD(nixlBackendMD *input) override;

  // Data transfer
  nixl_status_t
  prepXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
           nixlBackendReqH *&handle,
           const nixl_opt_b_args_t *opt_args = nullptr) const override;

  nixl_status_t
  postXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
           nixlBackendReqH *&handle,
           const nixl_opt_b_args_t *opt_args = nullptr) const override;

  nixl_status_t checkXfer(nixlBackendReqH *handle) const override;
  nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

  int progress() override;

  virtual nixl_status_t getNotifs(notif_list_t &notif_list) override;
  nixl_status_t genNotif(const std::string &remote_agent,
                         const std::string_view msg) const override;

  // public function for UCX worker to mark connections as connected
  nixl_status_t checkConn(const std::string &remote_agent);
  nixl_status_t endConn(const std::string &remote_agent);

protected:
  nixlUcxEngine(const nixlBackendInitParams &init_params);

  const std::vector<std::unique_ptr<nixlUcxWorker>> &getWorkers() const {
    return uws;
  }

  const std::unique_ptr<nixlUcxWorker> &getWorker(size_t worker_id) const {
    return uws[worker_id];
  }

  virtual size_t getWorkerId() const {
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) %
           uws.size();
  }

  void getNotifsImpl(notif_list_t &notif_list);

  virtual int vramApplyCtx();

  virtual void appendNotif(std::string remote_name, std::string msg);

  // Check if the device pointer is associated with the expected device.
  M::LogicalResult checkDevicePointerContext(void *address, int addrDeviceId);

  // If we have a device context, set it as active within the scope of the
  // returned object. Otherwise do nothing.
  [[nodiscard]] std::unique_ptr<DeviceContextScope> activateContext() const {
    if (deviceCtx)
      return std::make_unique<DeviceContextScope>(*deviceCtx);
    return {};
  }

private:
  void vramInitCtx();
  void vramFiniCtx();
  int vramUpdateCtx(void *address, uint64_t dev_id, bool &restart_reqd);

  // Connection helper
  static ucs_status_t connectionCheckAmCb(void *arg, const void *header,
                                          size_t header_length, void *data,
                                          size_t length,
                                          const ucp_am_recv_param_t *param);

  static ucs_status_t connectionTermAmCb(void *arg, const void *header,
                                         size_t header_length, void *data,
                                         size_t length,
                                         const ucp_am_recv_param_t *param);

  // Memory management helpers
  nixl_status_t internalMDHelper(const blob_ref blob, const std::string &agent,
                                 nixlBackendMD *&output);

  // Notifications
  nixl_status_t checkXferWithoutLock(nixlBackendReqH *handle) const;
  static ucs_status_t notifAmCb(void *arg, const void *header,
                                size_t header_length, void *data, size_t length,
                                const ucp_am_recv_param_t *param);
  nixl_status_t notifSendPriv(const std::string &remote_agent,
                              const std::string_view msg, nixlUcxReq &req,
                              size_t worker_id) const;

  /* UCX data */
  std::unique_ptr<nixlUcxContext> uc;
  std::vector<std::unique_ptr<nixlUcxWorker>> uws;
  std::string workerAddr;

  /* CUDA data*/
  DeviceContextRef deviceCtx;

  /* Notifications */
  notif_list_t notifMainList;

  /* Inflight send transfers */
  // These are mutable as they are accessed in const methods.
  mutable std::set<nixlBackendReqH *> inflightSendXfers;
  mutable std::mutex inflightSendXfersMtx;

  // Map of agent name to saved nixlUcxConnection info
  std::unordered_map<std::string, ucx_connection_ptr_t, std::hash<std::string>,
                     strEqual>
      remoteConnMap;
};

class nixlUcxThread;

/**
 * Represents an engine with a single progress thread for all shared workers
 */
class nixlUcxThreadEngine : public nixlUcxEngine {
public:
  nixlUcxThreadEngine(const nixlBackendInitParams &init_params);
  ~nixlUcxThreadEngine();

  bool supportsProgTh() const override { return true; }

  nixl_status_t getNotifs(notif_list_t &notif_list) override;

protected:
  int vramApplyCtx() override;

  void appendNotif(std::string remote_name, std::string msg) override;

private:
  std::unique_ptr<nixlUcxThread> thread_;
  std::mutex notifMtx_;
  notif_list_t notifPthr_;
};

nixlBackendPlugin *createStaticUcxPlugin();

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_PLUGINS_UCX_UCX_BACKEND_H
