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
#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/NIXL/utils/common/str_tools.h"

// Local includes
#include "AsyncRT/NIXL/backend/backend_plugin.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/utils/common/list_elem.h"
#include "AsyncRT/NIXL/utils/common/nixl_time.h"
#include "AsyncRT/NIXL/utils/ucx/ucx_utils.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

namespace M::AsyncRT::NIXL {

class nixlUcxConnection : public nixlBackendConnMD {
private:
  std::string remoteAgent;
  // NOTE: nixlUcxEp object must be alive as long as ucp_ep_h is alive
  //       since the object is passed to ucp_ep_create_nbx as err_cb
  //       argument.
  std::shared_ptr<nixlUcxEp> ep = std::make_shared<nixlUcxEp>();

public:
  nixlUcxEp &getEp() { return *ep; }
  // Extra information required for UCX connections

  friend class nixlUcxEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlUcxPrivateMetadata : public nixlBackendMD {
private:
  nixlUcxMem mem;
  nixl_blob_t rkeyStr;

public:
  nixlUcxPrivateMetadata() : nixlBackendMD(true) {}

  ~nixlUcxPrivateMetadata() override {}

  std::string get() const { return rkeyStr; }

  friend class nixlUcxEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlUcxPublicMetadata : public nixlBackendMD {

public:
  nixlUcxRkey rkey;
  nixlUcxConnection conn;

  nixlUcxPublicMetadata() : nixlBackendMD(false) {}

  ~nixlUcxPublicMetadata() {}
};

class nixlUcxEngine : public nixlBackendEngine {
private:
  /* UCX data */
  std::unique_ptr<nixlUcxContext> uc;
  std::unique_ptr<nixlUcxWorker> uw;
  std::unique_ptr<char[]> workerAddr;
  size_t workerSize;

  /* Progress thread data */
  volatile bool pthrStop, pthrActive, pthrOn;
  int noSyncIters;
  std::thread pthr;
  nixlTime::us_t pthrDelay;

  /* CUDA data*/
  DeviceContextRef deviceCtx;

  /* Notifications */
  notif_list_t notifMainList;
  std::mutex notifMtx;
  notif_list_t notifPthrPriv, notifPthr;

  // Map of agent name to saved nixlUcxConnection info
  std::unordered_map<std::string, nixlUcxConnection, std::hash<std::string>,
                     strEqual>
      remoteConnMap;

  // Check if the device pointer is associated with the expected device.
  M::LogicalResult checkDevicePointerContext(void *address, int addrDeviceId);

  // If we have a device context, set it as active within the scope of the
  // returned object. Otherwise do nothing.
  [[nodiscard]] std::unique_ptr<DeviceContextScope> activateContext() const {
    if (deviceCtx)
      return std::make_unique<DeviceContextScope>(*deviceCtx);
    return {};
  }

  // Threading infrastructure
  //   TODO: move the thread management one outside of NIXL common infra
  void progressFunc();
  void progressThreadStart();
  void progressThreadStop();
  void progressThreadRestart();
  bool isProgressThread() {
    return (std::this_thread::get_id() == pthr.get_id());
  }

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
  static ucs_status_t notifAmCb(void *arg, const void *header,
                                size_t header_length, void *data, size_t length,
                                const ucp_am_recv_param_t *param);
  nixl_status_t notifSendPriv(const std::string &remote_agent,
                              const std::string_view msg, nixlUcxReq &req);
  void notifProgress();
  void notifCombineHelper(notif_list_t &src, notif_list_t &tgt);
  void notifProgressCombineHelper(notif_list_t &src, notif_list_t &tgt);

public:
  nixlUcxEngine(const nixlBackendInitParams *init_params);
  ~nixlUcxEngine();

  bool supportsRemote() const override { return true; }
  bool supportsLocal() const override { return true; }
  bool supportsNotif() const override { return true; }
  bool supportsProgTh() const override { return pthrOn; }

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
  nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args = nullptr) override;

  nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args = nullptr) override;

  nixl_status_t checkXfer(nixlBackendReqH *handle) override;
  nixl_status_t releaseReqH(nixlBackendReqH *handle) override;

  int progress() override;

  nixl_status_t getNotifs(notif_list_t &notif_list) override;
  nixl_status_t genNotif(const std::string &remote_agent,
                         const std::string_view msg) override;

  // public function for UCX worker to mark connections as connected
  nixl_status_t checkConn(const std::string &remote_agent);
  nixl_status_t endConn(const std::string &remote_agent);
};

nixlBackendPlugin *createStaticUcxPlugin();

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_PLUGINS_UCX_UCX_BACKEND_H
