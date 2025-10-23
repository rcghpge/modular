//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/agent_data.h"
#include "AsyncRT/NIXL/backend/backend_engine.h"
#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/plugin_manager.h"
#include "AsyncRT/NIXL/transfer_request.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/common/operators.h"
#include "AsyncRT/NIXL/utils/serdes/serdes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <iostream>

using namespace M::AsyncRT::NIXL;
using namespace std::string_view_literals;

/* ERROR log prefixed with current function name and a colon */
#define NIXL_ERROR_FUNC NIXL_ERROR << __func__ << ": "

#define SERIALIZE_ERR(ret)                                                     \
  do {                                                                         \
    NIXL_ERROR_FUNC << "serialization failed";                                 \
    return ret;                                                                \
  } while (0)

#define DESERIALIZE_ERR()                                                      \
  do {                                                                         \
    NIXL_ERROR_FUNC << "deserialization failed";                               \
    return nixl_status_t::NIXL_ERR_MISMATCH;                                   \
  } while (0)

/*** nixlEnumStrings namespace implementation in API ***/
std::string_view nixlEnumStrings::memTypeStr(nixl_mem_t mem) {
  constexpr std::array<std::string_view, to_underlying(nixl_mem_t::COUNT)>
      nixl_mem_str = {"DRAM_SEG"sv, "VRAM_SEG"sv, "BLK_SEG"sv, "OBJ_SEG"sv,
                      "FILE_SEG"sv};
  if (mem < nixl_mem_t::DRAM_SEG || mem > nixl_mem_t::FILE_SEG)
    return "BAD_SEG"sv;
  return nixl_mem_str[to_underlying(mem)];
}

std::string_view nixlEnumStrings::xferOpStr(nixl_xfer_op_t op) {
  constexpr std::array<std::string_view, 2> nixl_op_str = {"READ"sv, "WRITE"sv};
  if (op < nixl_xfer_op_t::NIXL_READ || op > nixl_xfer_op_t::NIXL_WRITE)
    return "BAD_OP"sv;
  return nixl_op_str[to_underlying(op)];
}

const char *nixlEnumStrings::statusStr(nixl_status_t status) {
  switch (status) {
  case nixl_status_t::NIXL_IN_PROG:
    return "NIXL_IN_PROG";
  case nixl_status_t::NIXL_SUCCESS:
    return "NIXL_SUCCESS";
  case nixl_status_t::NIXL_ERR_NOT_POSTED:
    return "NIXL_ERR_NOT_POSTED";
  case nixl_status_t::NIXL_ERR_INVALID_PARAM:
    return "NIXL_ERR_INVALID_PARAM";
  case nixl_status_t::NIXL_ERR_BACKEND:
    return "NIXL_ERR_BACKEND";
  case nixl_status_t::NIXL_ERR_NOT_FOUND:
    return "NIXL_ERR_NOT_FOUND";
  case nixl_status_t::NIXL_ERR_MISMATCH:
    return "NIXL_ERR_MISMATCH";
  case nixl_status_t::NIXL_ERR_NOT_ALLOWED:
    return "NIXL_ERR_NOT_ALLOWED";
  case nixl_status_t::NIXL_ERR_REPOST_ACTIVE:
    return "NIXL_ERR_REPOST_ACTIVE";
  case nixl_status_t::NIXL_ERR_UNKNOWN:
    return "NIXL_ERR_UNKNOWN";
  case nixl_status_t::NIXL_ERR_NOT_SUPPORTED:
    return "NIXL_ERR_NOT_SUPPORTED";
  case nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT:
    return "NIXL_ERR_REMOTE_DISCONNECT";
  }

  llvm_unreachable("bad status");
}

/*** nixlAgentData constructor/destructor, as part of nixlAgent's ***/
// TODO(GENAI-170): use fallible constructor idiom and move to header.
nixlAgentData::nixlAgentData(std::string agentName, nixlAgentConfig cfg)
    : name(std::move(agentName)), config(std::move(cfg)),
      lock(config.syncMode) {
  if (name.size() == 0) {
    // TODO(bduke): return ErrorOr here using fallible constructor idiom.
    NIXL_FATAL << "Agent needs a name";
    std::abort();
  }

  memorySection = new nixlLocalSection();
}

nixlAgentData::~nixlAgentData() {
  delete memorySection;

  for (auto &elm : remoteSections)
    delete elm.second;

  for (auto &elm : backendEngines) {
    auto &plugin_manager = nixlPluginManager::getInstance();
    auto plugin_handle = plugin_manager.getPlugin(elm.second->getType());

    if (plugin_handle) {
      // If we have a plugin handle, use it to destroy the engine
      plugin_handle->destroyEngine(elm.second);
    }
  }

  for (auto &elm : backendHandles)
    delete elm.second;
}

/*** nixlAgent implementation ***/
nixlAgent::nixlAgent(std::string name, nixlAgentConfig cfg)
    : data(std::make_unique<nixlAgentData>(std::move(name), std::move(cfg))) {
  if (cfg.useListenThread) {
    int my_port = cfg.listenPort;
    if (my_port == 0)
      my_port = default_comm_port;
    data->listener = new nixlMDStreamListener(my_port);
    data->listener->setupListener();
  }

  if (cfg.useListenThread) {
    data->commThreadStop = false;
    data->commThread =
        std::thread(&nixlAgentData::commWorker, data.get(), this);
  }
}

nixlAgent::~nixlAgent() {
  if (data->config.useListenThread) {
    data->commThreadStop = true;
    if (data->commThread.joinable())
      data->commThread.join();

    // Close remaining connections from comm thread
    for (auto &[remote, fd] : data->remoteSockets) {
      shutdown(fd, SHUT_RDWR);
      close(fd);
    }

    if (data->listener)
      delete data->listener;
  }
}

nixl_status_t nixlAgent::getAvailPlugins(std::vector<nixl_backend_t> &plugins) {
  auto &plugin_manager = nixlPluginManager::getInstance();
  plugins = plugin_manager.getLoadedPluginNames();
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::getPluginParams(const nixl_backend_t &type,
                                         nixl_mem_list_t &mems,
                                         nixl_b_params_t &params) const {
  // Normalize plugin key: trim whitespace, lowercase everything.
  std::string canonicalType = StringRef(type).trim().lower();

  // First try to get options from a loaded plugin
  auto &pluginManager = nixlPluginManager::getInstance();
  std::shared_ptr<const nixlPluginHandle> pluginHandle =
      pluginManager.getPlugin(canonicalType);

  if (pluginHandle) {
    // If the plugin is already loaded, get options directly
    params = pluginHandle->getBackendOptions();
    mems = pluginHandle->getBackendMems();
    return nixl_status_t::NIXL_SUCCESS;
  }

  // If plugin isn't loaded yet, try to load it temporarily
  pluginHandle = pluginManager.loadPlugin(canonicalType);
  if (pluginHandle) {
    params = pluginHandle->getBackendOptions();
    mems = pluginHandle->getBackendMems();

    NIXL_LOCK_GUARD(data->lock);

    // We don't keep the plugin loaded if we didn't have it before
    if (data->backendEngines.count(canonicalType) == 0)
      pluginManager.unloadPlugin(canonicalType);

    return nixl_status_t::NIXL_SUCCESS;
  }

  return nixl_status_t::NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlAgent::getBackendParams(const nixlBackendH *backend,
                                          nixl_mem_list_t &mems,
                                          nixl_b_params_t &params) const {
  if (!backend)
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;

  NIXL_LOCK_GUARD(data->lock);
  mems = backend->engine->getSupportedMems();
  params = backend->engine->getCustomParams();
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::createBackend(const nixl_backend_t &type,
                                       const nixl_b_params_t &params,
                                       nixlBackendH *&bknd_hndl) {
  nixlBackendEngine *backend = nullptr;
  nixlBackendInitParams init_params;
  nixl_mem_list_t mems;
  nixl_status_t ret;
  std::string str;
  backend_list_t *backend_list;

  NIXL_LOCK_GUARD(data->lock);
  // Registering same type of backend is not supported, unlikely and prob error
  if (data->backendEngines.count(type) != 0) {
    NIXL_ERROR_FUNC << "backend already created for type '" << type << "'";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  init_params.localAgent = data->name;
  init_params.type = type;
  init_params.customParams = const_cast<nixl_b_params_t *>(&params);
  init_params.enableProgTh = data->config.useProgThread;
  init_params.pthrDelay = data->config.pthrDelay;
  init_params.syncMode = data->config.syncMode;

  // First, try to load the backend as a plugin
  auto &plugin_manager = nixlPluginManager::getInstance();
  auto plugin_handle = plugin_manager.loadPlugin(type);

  if (plugin_handle) {
    // Plugin found, use it to create the backend
    backend = plugin_handle->createEngine(&init_params);
  } else {
    NIXL_ERROR_FUNC << "unsupported backend '" << type << "'";
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  if (backend) {
    if (backend->getInitErr()) {
      delete backend;
      NIXL_ERROR_FUNC << "backend initialization error for '" << type << "'";
      return nixl_status_t::NIXL_ERR_BACKEND;
    }

    // Enforce that any backend supporting remote also supports notifications
    if (backend->supportsRemote() && !backend->supportsNotif()) {
      delete backend;
      NIXL_ERROR_FUNC << "backend '" << type
                      << "' supportsRemote but not notifications";
      return nixl_status_t::NIXL_ERR_BACKEND;
    }

    if (backend->supportsRemote()) {
      ret = backend->getConnInfo(str);
      if (ret != nixl_status_t::NIXL_SUCCESS) {
        delete backend;
        NIXL_ERROR_FUNC << "failed to get connection info for '" << type
                        << "' with status " << ret;
        return ret;
      }
      data->connMD[type] = str;
    }

    if (backend->supportsLocal()) {
      ret = backend->connect(data->name);

      if (nixl_status_t::NIXL_SUCCESS != ret) {
        delete backend;
        NIXL_ERROR_FUNC << "backend '" << type
                        << "' encountered error during intra-agent transfer "
                           "setup with status "
                        << ret;
        return ret;
      }
    }

    bknd_hndl = new nixlBackendH(backend);
    if (!bknd_hndl) {
      delete backend;
      NIXL_ERROR_FUNC << "allocation of backend handle failed for '" << type
                      << "'";
      return nixl_status_t::NIXL_ERR_BACKEND;
    }

    data->backendEngines[type] = backend;
    data->backendHandles[type] = bknd_hndl;
    mems = backend->getSupportedMems();
    for (auto &elm : mems) {
      backend_list = &data->memToBackend[to_underlying(elm)];
      // First time creating this backend handle, so unique
      // The order of creation sets the preference order
      backend_list->push_back(backend);
    }

    if (backend->supportsRemote())
      data->notifEngines.push_back(backend);

    // TODO: Check if backend supports ProgThread
    //       when threading is in agent

    NIXL_DEBUG << "Created backend: " << type;
    return nixl_status_t::NIXL_SUCCESS;
  }

  NIXL_ERROR_FUNC << "backend creation failed for '" << type << "'";
  return nixl_status_t::NIXL_ERR_BACKEND;
}

nixl_status_t nixlAgent::registerMemOnBackends(nixlAgentData &data,
                                               const nixl_reg_dlist_t &descs,
                                               backend_list_t &backends) {
  int successCount = 0;

  for (nixlBackendEngine *backend : backends) {
    nixl_sec_dlist_t secDescs(descs.getType(), /*sorted=*/false);

    nixl_status_t ret =
        data.memorySection->addDescList(descs, *backend, secDescs);
    if (ret != nixl_status_t::NIXL_SUCCESS) {
      // Soft fail: try next.
      continue;
    }

    if (backend->supportsLocal()) {
      auto [iter, inserted] = data.remoteSections.emplace(data.name, nullptr);
      if (inserted) {
        // If newly inserted, the key didn't exist: create the remote section.
        iter->second = new nixlRemoteSection(data.name);
      }

      // Load descriptors to the new or existing section.
      ret = iter->second->loadLocalData(secDescs, backend);
      if (ret != nixl_status_t::NIXL_SUCCESS) {
        data.memorySection->remDescList(descs, *backend);
        continue;
      }
    }

    ++successCount;
  }

  return successCount ? nixl_status_t::NIXL_SUCCESS
                      : nixl_status_t::NIXL_ERR_BACKEND;
}

nixl_status_t nixlAgent::registerMem(const nixl_reg_dlist_t &descs,
                                     const nixl_opt_args_t *extra) {
  NIXL_LOCK_GUARD(data->lock);

  if (!extra || extra->backends.empty()) {
    backend_list_t *borrowedBackends =
        &data->memToBackend[to_underlying(descs.getType())];
    if (borrowedBackends->empty()) {
      NIXL_ERROR_FUNC << "no available backends for mem type '"
                      << descs.getType() << "'";
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }

    return registerMemOnBackends(*data, descs, *borrowedBackends);
  } else {
    std::unique_ptr<backend_list_t> ownedBackends =
        std::make_unique<backend_list_t>();
    for (const nixlBackendH *optBackend : extra->backends)
      ownedBackends->push_back(optBackend->engine);

    return registerMemOnBackends(*data, descs, *ownedBackends);
  }
}

nixl_status_t nixlAgent::deregisterMem(const nixl_reg_dlist_t &descs,
                                       const nixl_opt_args_t *extra_params) {

  backend_set_t backend_set;
  nixl_status_t ret, bad_ret = nixl_status_t::NIXL_SUCCESS;

  NIXL_LOCK_GUARD(data->lock);
  if (!extra_params || extra_params->backends.size() == 0) {
    backend_set_t *avail_backends;
    avail_backends = data->memorySection->queryBackends(descs.getType());
    if (!avail_backends || avail_backends->empty()) {
      NIXL_ERROR_FUNC << "no available backends for mem type '"
                      << descs.getType() << "'";
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }
    // Make a copy as we might change it in remDescList
    backend_set = *avail_backends;
  } else {
    for (auto &elm : extra_params->backends)
      backend_set.insert(elm->engine);
  }

  // Doing best effort, and returning err if any
  for (nixlBackendEngine *backend : backend_set) {
    ret = data->memorySection->remDescList(descs, *backend);
    if (ret != nixl_status_t::NIXL_SUCCESS)
      bad_ret = ret;
  }

  if (bad_ret != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC
        << "deregistration failed on at least one backend with status "
        << bad_ret;
  }

  return bad_ret;
}

nixl_status_t nixlAgent::queryMem(const nixl_reg_dlist_t &descs,
                                  std::vector<nixl_query_resp_t> &resp,
                                  const nixl_opt_args_t *extra_params) const {

  if (!extra_params || extra_params->backends.size() != 1) {
    NIXL_ERROR_FUNC << "this method requires exactly one backend to be passed";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  return extra_params->backends[0]->engine->queryMem(descs, resp);
}

nixl_status_t nixlAgent::makeConnection(const std::string &remote_agent,
                                        const nixl_opt_args_t *extra_params) {
  nixl_status_t ret;
  std::set<nixl_backend_t> backend_set;
  int count = 0;

  NIXL_LOCK_GUARD(data->lock);
  const auto &remoteBackends = data->remoteBackends;
  auto remoteIter = remoteBackends.find(remote_agent);
  if (remoteIter == remoteBackends.end()) {
    NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent
                    << "' not found";
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }
  const auto &agentBackend = remoteIter->second;

  if (!extra_params || extra_params->backends.size() == 0) {
    if (agentBackend.empty()) {
      NIXL_ERROR_FUNC << "no backends are found in metadata for remote agent '"
                      << remote_agent << "'";
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }
    for (auto &[r_bknd, conn_info] : agentBackend)
      backend_set.insert(r_bknd);
  } else {
    for (auto &elm : extra_params->backends)
      backend_set.insert(elm->engine->getType());
  }

  // For now trying to make all the connections, can become best effort,
  for (auto &backend : backend_set) {
    const backend_map_t &backendEngines = data->backendEngines;
    if (auto engineIter = backendEngines.find(backend);
        engineIter != backendEngines.end()) {
      nixlBackendEngine *eng = engineIter->second;
      ret = eng->connect(remote_agent);
      if (ret != nixl_status_t::NIXL_SUCCESS) {
        NIXL_ERROR_FUNC << "connect('" << remote_agent
                        << "') failed on backend '" << eng->getType()
                        << "' with status " << ret;
        break;
      }
      count++;
    }
  }

  if (ret != nixl_status_t::NIXL_SUCCESS) // Error is already logged
    return ret;

  if (count == 0) { // No common backend
    NIXL_ERROR_FUNC << "no common backend to connect with '" << remote_agent
                    << "'";
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::prepXferDlist(
    const std::string &agent_name, const nixl_xfer_dlist_t &descs,
    nixlDlistH *&dlist_hndl, const nixl_opt_args_t *extra_params) const {
  // Using a set as order is not important to revert the operation
  // TODO(GENAI-170): Refactor to remove conditional ownership.
  backend_set_t *backend_set;
  nixl_status_t ret;
  int count = 0;
  bool init_side = (agent_name == NIXL_INIT_AGENT);

  NIXL_LOCK_GUARD(data->lock);
  // When central KV is supported, still it should return error,
  // just we can add a call to fetchRemoteMD for next time
  const auto &remoteSections = data->remoteSections;
  auto sectionIter = remoteSections.find(agent_name);
  if (!init_side && (sectionIter == remoteSections.end())) {
    NIXL_ERROR_FUNC << "metadata for remote agent '" << agent_name
                    << "' not found";
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }
  nixlRemoteSection *agentSection = sectionIter->second;

  if (!extra_params || extra_params->backends.size() == 0) {
    if (!init_side)
      backend_set = agentSection->queryBackends(descs.getType());
    else
      backend_set = data->memorySection->queryBackends(descs.getType());

    if (!backend_set || backend_set->empty()) {
      NIXL_ERROR_FUNC << "no available backends for mem type '"
                      << descs.getType() << "'";
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }
  } else {
    backend_set = new backend_set_t();
    for (auto &elm : extra_params->backends)
      backend_set->insert(elm->engine);
  }

  // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

  nixlDlistH *handle = new nixlDlistH;
  if (init_side) {
    handle->isLocal = true;
    handle->remoteAgent = "";
  } else {
    handle->isLocal = false;
    handle->remoteAgent = agent_name;
  }

  for (auto &backend : *backend_set) {
    handle->descs[backend] =
        new nixl_meta_dlist_t(descs.getType(), descs.isSorted());
    if (init_side)
      ret = data->memorySection->populate(descs, backend,
                                          *(handle->descs[backend]));
    else
      ret = agentSection->populate(descs, backend, *(handle->descs[backend]));
    if (ret == nixl_status_t::NIXL_SUCCESS) {
      count++;
    } else {
      delete handle->descs[backend];
      handle->descs.erase(backend);
    }
  }

  if (extra_params && extra_params->backends.size() > 0)
    delete backend_set;

  if (count == 0) {
    delete handle;
    dlist_hndl = nullptr;
    NIXL_ERROR_FUNC << "failed to prepare the descriptors for any of "
                       "the specified or potential backends for agent '"
                    << agent_name << "'";
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  } else {
    dlist_hndl = handle;
    return nixl_status_t::NIXL_SUCCESS;
  }
}

nixl_status_t nixlAgent::makeXferReq(
    const nixl_xfer_op_t &operation, const nixlDlistH *local_side,
    const std::vector<int> &local_indices, const nixlDlistH *remote_side,
    const std::vector<int> &remote_indices, nixlXferReqH *&req_hndl,
    const nixl_opt_args_t *extra_params) const {

  nixl_opt_b_args_t opt_args;
  nixl_status_t ret;
  int desc_count = (int)local_indices.size();
  nixlBackendEngine *backend = nullptr;

  req_hndl = nullptr;

  if (!local_side || !remote_side) {
    NIXL_ERROR_FUNC << "local or remote side handle is null";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  if ((!local_side->isLocal) || (remote_side->isLocal)) {
    NIXL_ERROR_FUNC
        << "invalid sides (local must be local, remote must be remote)";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  NIXL_LOCK_GUARD(data->lock);
  // The remote was invalidated in between prepXferDlist and this call
  if (data->remoteSections.count(remote_side->remoteAgent) == 0) {
    NIXL_ERROR_FUNC
        << "remote agent '" << remote_side->remoteAgent
        << "' was invalidated in between prepXferDlist and this call";
    delete req_hndl;
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  if (extra_params && extra_params->backends.size() > 0) {
    for (auto &elm : extra_params->backends) {
      if ((local_side->descs.count(elm->engine) > 0) &&
          (remote_side->descs.count(elm->engine) > 0)) {
        backend = elm->engine;
        break;
      }
    }
  } else {
    for (auto &loc_bknd : local_side->descs) {
      for (auto &rem_bknd : remote_side->descs) {
        if (loc_bknd.first == rem_bknd.first) {
          backend = loc_bknd.first;
          break;
        }
      }
      if (backend)
        break;
    }
  }

  if (!backend) {
    NIXL_ERROR_FUNC << "could not find a common backend in the specified or "
                       "available list of backends for the prepped Dlists";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  nixl_meta_dlist_t *local_descs = local_side->descs.at(backend);
  nixl_meta_dlist_t *remote_descs = remote_side->descs.at(backend);

  if ((desc_count == 0) || (remote_indices.size() == 0) ||
      (desc_count != (int)remote_indices.size())) {
    NIXL_ERROR_FUNC << "different number of indices for local (" << desc_count
                    << "), remote (" << remote_indices.size() << ")";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  for (int i = 0; i < desc_count; ++i) {
    if ((local_indices[i] >= local_descs->descCount()) ||
        (local_indices[i] < 0)) {
      NIXL_ERROR_FUNC << "local index out of range at index " << i
                      << " with value " << local_indices[i];
      return nixl_status_t::NIXL_ERR_INVALID_PARAM;
    }
    if ((remote_indices[i] >= remote_descs->descCount()) ||
        (remote_indices[i] < 0)) {
      NIXL_ERROR_FUNC << "remote index out of range at index " << i
                      << " with value " << remote_indices[i];
      return nixl_status_t::NIXL_ERR_INVALID_PARAM;
    }
    if ((*local_descs)[local_indices[i]].len !=
        (*remote_descs)[remote_indices[i]].len) {
      NIXL_ERROR_FUNC << "length mismatch at index pair " << i
                      << " with local index " << local_indices[i]
                      << " and remote index " << remote_indices[i];
      return nixl_status_t::NIXL_ERR_INVALID_PARAM;
    }
  }

  if (extra_params && extra_params->hasNotif) {
    opt_args.notifMsg = extra_params->notifMsg;
    opt_args.hasNotif = true;
  }

  if ((opt_args.hasNotif) && (!backend->supportsNotif())) {
    NIXL_ERROR_FUNC << "the selected backend '" << backend->getType()
                    << "' does not support notifications";
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  // Populate has been already done, no benefit in having sorted descriptors
  // which will be overwritten by [] assignment operator.
  nixlXferReqH *handle = new nixlXferReqH;
  handle->initiatorDescs =
      new nixl_meta_dlist_t(local_descs->getType(), false, desc_count);

  handle->targetDescs =
      new nixl_meta_dlist_t(remote_descs->getType(), false, desc_count);

  if (extra_params && extra_params->skipDescMerge) {
    for (int i = 0; i < desc_count; ++i) {
      (*handle->initiatorDescs)[i] = (*local_descs)[local_indices[i]];
      (*handle->targetDescs)[i] = (*remote_descs)[remote_indices[i]];
    }
  } else {
    int i = 0, j = 0; // final list size
    while (i < (desc_count)) {
      nixlMetaDesc local_desc1 = (*local_descs)[local_indices[i]];
      nixlMetaDesc remote_desc1 = (*remote_descs)[remote_indices[i]];

      if (i != (desc_count - 1)) {
        nixlMetaDesc *local_desc2 = &((*local_descs)[local_indices[i + 1]]);
        nixlMetaDesc *remote_desc2 = &((*remote_descs)[remote_indices[i + 1]]);

        while (((local_desc1.addr + local_desc1.len) == local_desc2->addr) &&
               ((remote_desc1.addr + remote_desc1.len) == remote_desc2->addr) &&
               (local_desc1.metadataP == local_desc2->metadataP) &&
               (remote_desc1.metadataP == remote_desc2->metadataP) &&
               (local_desc1.devId == local_desc2->devId) &&
               (remote_desc1.devId == remote_desc2->devId)) {

          local_desc1.len += local_desc2->len;
          remote_desc1.len += remote_desc2->len;

          i++;
          if (i == (desc_count - 1))
            break;

          local_desc2 = &((*local_descs)[local_indices[i + 1]]);
          remote_desc2 = &((*remote_descs)[remote_indices[i + 1]]);
        }
      }

      (*handle->initiatorDescs)[j] = local_desc1;
      (*handle->targetDescs)[j] = remote_desc1;
      j++;
      i++;
    }
    NIXL_DEBUG << "reqH descList size down to " << j;
    handle->initiatorDescs->resize(j);
    handle->targetDescs->resize(j);
  }

  handle->engine = backend;
  handle->remoteAgent = remote_side->remoteAgent;
  handle->notifMsg = opt_args.notifMsg;
  handle->hasNotif = opt_args.hasNotif;
  handle->backendOp = operation;
  handle->status = nixl_status_t::NIXL_ERR_NOT_POSTED;

  ret = handle->engine->prepXfer(handle->backendOp, *handle->initiatorDescs,
                                 *handle->targetDescs, handle->remoteAgent,
                                 handle->backendHandle, &opt_args);
  if (ret != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC << "backend '" << backend->getType()
                    << "' failed to prepare the transfer request with status "
                    << ret;
    delete handle;
    return ret;
  }

  req_hndl = handle;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::createXferReq(
    const nixl_xfer_op_t &operation, const nixl_xfer_dlist_t &local_descs,
    const nixl_xfer_dlist_t &remote_descs, const std::string &remote_agent,
    nixlXferReqH *&req_hndl, const nixl_opt_args_t *extra_params) const {
  nixl_status_t ret1, ret2;
  nixl_opt_b_args_t opt_args;
  nixlXferReqH *handle{nullptr};
  {
    backend_set_t backendSet;

    req_hndl = nullptr;

    NIXL_LOCK_GUARD(data->lock);
    if (data->remoteSections.count(remote_agent) == 0) {
      NIXL_ERROR_FUNC << "metadata for remote agent '" << remote_agent
                      << "' not found";
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }

    // Check the correspondence between descriptor lists
    if (local_descs.descCount() != remote_descs.descCount()) {
      NIXL_ERROR_FUNC << "different descriptor list sizes (local="
                      << local_descs.descCount()
                      << ", remote=" << remote_descs.descCount() << ")";
      return nixl_status_t::NIXL_ERR_INVALID_PARAM;
    }
    for (int i = 0; i < local_descs.descCount(); ++i) {
      size_t local_len = local_descs[i].len;
      size_t remote_len = remote_descs[i].len;
      if (local_len != remote_len) {
        NIXL_ERROR_FUNC << "length mismatch at index " << i;
        return nixl_status_t::NIXL_ERR_INVALID_PARAM;
      }
    }

    if (!extra_params || extra_params->backends.size() == 0) {
      // Finding backends that support the corresponding memories
      // locally and remotely, and find the common ones.
      backend_set_t *local_set =
          data->memorySection->queryBackends(local_descs.getType());
      backend_set_t *remote_set =
          data->remoteSections[remote_agent]->queryBackends(
              remote_descs.getType());

      if (!local_set || !remote_set) {
        NIXL_ERROR_FUNC << "no backends found for local or remote for their "
                           "corresponding memory type";
        return nixl_status_t::NIXL_ERR_NOT_FOUND;
      }

      for (auto &elm : *local_set)
        if (remote_set->count(elm) != 0)
          backendSet.insert(elm);

      if (backendSet.empty()) {
        NIXL_ERROR_FUNC
            << "no potential backend found to be able to do the transfer";
        return nixl_status_t::NIXL_ERR_NOT_FOUND;
      }
    } else {
      for (auto &elm : extra_params->backends)
        backendSet.insert(elm->engine);
    }

    // TODO: when central KV is supported, add a call to fetchRemoteMD
    // TODO: merge descriptors back to back in memory (like makeXferReq).
    // TODO [Perf]: Avoid heap allocation on the datapath, maybe use a mem pool

    handle = new nixlXferReqH;
    handle->initiatorDescs =
        new nixl_meta_dlist_t(local_descs.getType(), local_descs.isSorted());

    handle->targetDescs =
        new nixl_meta_dlist_t(remote_descs.getType(), remote_descs.isSorted());

    // Currently we loop through and find first local match. Can use a
    // preference list or more exhaustive search.
    for (auto &backend : backendSet) {
      // If populate fails, it clears the resp before return
      ret1 = data->memorySection->populate(local_descs, backend,
                                           *handle->initiatorDescs);
      ret2 = data->remoteSections[remote_agent]->populate(remote_descs, backend,
                                                          *handle->targetDescs);

      if ((ret1 == nixl_status_t::NIXL_SUCCESS) &&
          (ret2 == nixl_status_t::NIXL_SUCCESS)) {
        handle->engine = backend;
        break;
      }
    }
  }

  if (!handle->engine) {
    NIXL_ERROR_FUNC << "no specified or potential backend had the required "
                       "registrations to be able to do the transfer";
    delete handle;
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  if (extra_params && extra_params->hasNotif) {
    opt_args.notifMsg = extra_params->notifMsg;
    opt_args.hasNotif = true;
  }

  if (opt_args.hasNotif && (!handle->engine->supportsNotif())) {
    NIXL_ERROR_FUNC << "the selected backend '" << handle->engine->getType()
                    << "' does not support notifications";
    delete handle;
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  handle->remoteAgent = remote_agent;
  handle->backendOp = operation;
  handle->status = nixl_status_t::NIXL_ERR_NOT_POSTED;
  handle->notifMsg = opt_args.notifMsg;
  handle->hasNotif = opt_args.hasNotif;

  ret1 = handle->engine->prepXfer(handle->backendOp, *handle->initiatorDescs,
                                  *handle->targetDescs, handle->remoteAgent,
                                  handle->backendHandle, &opt_args);
  if (ret1 != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC << "backend '" << handle->engine->getType()
                    << "' failed to prepare the transfer request with status "
                    << ret1;
    delete handle;
    return ret1;
  }

  req_hndl = handle;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::postXferReq(nixlXferReqH *req_hndl,
                       const nixl_opt_args_t *extra_params) const {
  nixl_status_t ret;
  nixl_opt_b_args_t opt_args;

  opt_args.hasNotif = false;

  if (!req_hndl)
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;

  NIXL_LOCK_GUARD(data->lock);
  // Check if the remote was invalidated before post/repost
  if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
    delete req_hndl;
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }

  // We can't repost while a request is in progress
  if (req_hndl->status == nixl_status_t::NIXL_IN_PROG) {
    req_hndl->status = req_hndl->engine->checkXfer(req_hndl->backendHandle);
    if (req_hndl->status == nixl_status_t::NIXL_IN_PROG) {
      delete req_hndl;
      return nixl_status_t::NIXL_ERR_REPOST_ACTIVE;
    }
  }

  // Carrying over notification from xfer handle creation time
  if (req_hndl->hasNotif) {
    opt_args.notifMsg = req_hndl->notifMsg;
    opt_args.hasNotif = true;
  }

  // Updating the notification based on opt_args
  if (extra_params) {
    if (extra_params->hasNotif) {
      req_hndl->notifMsg = extra_params->notifMsg;
      opt_args.notifMsg = extra_params->notifMsg;
      req_hndl->hasNotif = true;
      opt_args.hasNotif = true;
    } else {
      req_hndl->hasNotif = false;
      opt_args.hasNotif = false;
    }
  }

  if (opt_args.hasNotif && (!req_hndl->engine->supportsNotif())) {
    delete req_hndl;
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  // If status is not NIXL_IN_PROG we can repost,
  ret = req_hndl->engine->postXfer(
      req_hndl->backendOp, *req_hndl->initiatorDescs, *req_hndl->targetDescs,
      req_hndl->remoteAgent, req_hndl->backendHandle, &opt_args);
  req_hndl->status = ret;
  return ret;
}

nixl_status_t nixlAgent::getXferStatus(nixlXferReqH *req_hndl) const {

  NIXL_LOCK_GUARD(data->lock);
  // If the status is done, no need to recheck.
  if (req_hndl->status == nixl_status_t::NIXL_IN_PROG) {
    // Check if the remote was invalidated before completion
    if (data->remoteSections.count(req_hndl->remoteAgent) == 0) {
      delete req_hndl;
      return nixl_status_t::NIXL_ERR_NOT_FOUND;
    }
    req_hndl->status = req_hndl->engine->checkXfer(req_hndl->backendHandle);
  }

  return req_hndl->status;
}

nixl_status_t nixlAgent::queryXferBackend(const nixlXferReqH *req_hndl,
                                          nixlBackendH *&backend) const {
  NIXL_LOCK_GUARD(data->lock);
  backend = data->backendHandles[req_hndl->engine->getType()];
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::releaseXferReq(nixlXferReqH *req_hndl) const {

  NIXL_LOCK_GUARD(data->lock);
  // attempt to cancel request
  if (req_hndl->status == nixl_status_t::NIXL_IN_PROG) {
    req_hndl->status = req_hndl->engine->checkXfer(req_hndl->backendHandle);

    if (req_hndl->status == nixl_status_t::NIXL_IN_PROG) {

      req_hndl->status = req_hndl->engine->releaseReqH(req_hndl->backendHandle);

      if (isErrorStatus(req_hndl->status))
        return nixl_status_t::NIXL_ERR_REPOST_ACTIVE;

      // just in case the backend doesn't set to NULL on success
      // this will prevent calling releaseReqH again in destructor
      req_hndl->backendHandle = nullptr;
    }
  }
  delete req_hndl;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::releasedDlistH(nixlDlistH *dlist_hndl) const {
  NIXL_LOCK_GUARD(data->lock);
  delete dlist_hndl;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::getNotifs(nixl_notifs_t &notif_map,
                                   const nixl_opt_args_t *extra_params) {
  const auto getNotifsImpl = [&notif_map](backend_list_t &backend_list) {
    if (backend_list.empty())
      return nixl_status_t::NIXL_ERR_BACKEND;

    notif_list_t backendNotifList;
    nixl_status_t badRet = nixl_status_t::NIXL_SUCCESS;

    // Doing best effort, if any backend errors out we return
    // error but proceed with the rest. We can add metadata about
    // the backend to the msg, but user could put it themselves.
    for (nixlBackendEngine *&eng : backend_list) {
      backendNotifList.clear();

      if (const nixl_status_t ret = eng->getNotifs(backendNotifList);
          isErrorStatus(ret))
        badRet = ret;

      // If the above fails then this list is empty, this becomes a no-op.
      for (auto &[agent, notif] : backendNotifList)
        notif_map[agent].push_back(std::move(notif));
    }
    return badRet;
  };

  NIXL_LOCK_GUARD(data->lock);
  if (!extra_params || extra_params->backends.size() == 0) {
    return getNotifsImpl(data->notifEngines);
  } else {
    backend_list_t backendList;
    for (auto &elm : extra_params->backends)
      if (elm->engine->supportsNotif())
        backendList.push_back(elm->engine);
    return getNotifsImpl(backendList);
  }
}

nixl_status_t nixlAgent::genNotif(const std::string &remote_agent,
                                  const std::string_view msg,
                                  const nixl_opt_args_t *extra_params) const {

  backend_list_t backend_list_value;
  backend_list_t *backend_list;

  if (!extra_params || extra_params->backends.empty()) {
    backend_list = &data->notifEngines;
  } else {
    backend_list = &backend_list_value;
    for (auto &elm : extra_params->backends) {
      if (elm->engine->supportsNotif()) {
        backend_list->push_back(elm->engine);
      }
    }
  }

  if (backend_list->empty()) {
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  NIXL_SHARED_LOCK_GUARD(data->lock);

  if (data->name == remote_agent) {
    for (const auto &eng : *backend_list) {
      if (eng->supportsLocal()) {
        return eng->genNotif(remote_agent, msg);
      }
    }
    return nixl_status_t::NIXL_ERR_NOT_FOUND;
  }
  const auto iter = data->remoteBackends.find(remote_agent);

  if (iter != data->remoteBackends.end()) {
    for (const auto &eng : *backend_list) {
      if (iter->second.count(eng->getType()) != 0) {
        return eng->genNotif(remote_agent, msg);
      }
    }
  }
  return nixl_status_t::NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlAgent::getLocalMD(nixl_blob_t &str) const {
  NIXL_LOCK_GUARD(data->lock);
  // data->connMD was populated when the backend was created
  const size_t conn_cnt = data->connMD.size();

  if (conn_cnt == 0) { // Error, no backend supports remote
    NIXL_ERROR_FUNC << "no backends support remote operations";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  nixlSerDes sd;
  sd.addStr("Agent", data->name);
  sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));

  for (const auto &[backend, metadata] : data->connMD) {
    sd.addStr("t", backend);
    sd.addStr("c", metadata);
  }

  sd.addStr("", "MemSection");

  data->memorySection->serialize(sd);

  str = sd.exportStr();
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t
nixlAgent::getLocalPartialMD(const nixl_reg_dlist_t &descs, nixl_blob_t &str,
                             const nixl_opt_args_t *extra_params) const {
  backend_list_t tmp_list;
  backend_list_t *backend_list;
  nixl_status_t ret;

  NIXL_LOCK_GUARD(data->lock);

  if (!extra_params || extra_params->backends.size() == 0) {
    if (descs.descCount() != 0) {
      // Non-empty dlist, return backends that support the memory type
      backend_list = &data->memToBackend[to_underlying(descs.getType())];
      if (backend_list->empty()) {
        NIXL_ERROR_FUNC << "no available backends for mem type '"
                        << descs.getType() << "'";
        return nixl_status_t::NIXL_ERR_NOT_FOUND;
      }
    } else {
      // Empty dlist, return all backends
      backend_list = &tmp_list;
      for (const auto &elm : data->backendEngines)
        backend_list->push_back(elm.second);
    }
  } else {
    backend_list = &tmp_list;
    for (const auto &elm : extra_params->backends)
      backend_list->push_back(elm->engine);
  }

  // First find all relevant engines and their conn info.
  // Best effort, ignore if no conn info (meaning backend doesn't support
  // remote).
  backend_set_t selected_engines;
  std::vector<typename decltype(data->connMD)::iterator> found_iters;
  for (const auto &backend : *backend_list) {
    auto it = data->connMD.find(backend->getType());
    if (it == data->connMD.end())
      continue;
    found_iters.push_back(it);
    selected_engines.insert(backend);
  }

  nixlSerDes sd;
  sd.addStr("Agent", data->name);

  // Only add connection info if requested via extra_params or empty dlist
  size_t conn_cnt = ((extra_params && extra_params->includeConnInfo) ||
                     descs.descCount() == 0)
                        ? found_iters.size()
                        : 0;
  sd.addBuf("Conns", &conn_cnt, sizeof(conn_cnt));

  for (size_t i = 0; i < conn_cnt; i++) {
    sd.addStr("t", found_iters[i]->first);
    sd.addStr("c", found_iters[i]->second);
  }

  // No engines found, but there are descs, this is an error
  if (selected_engines.size() == 0 && descs.descCount() > 0) {
    NIXL_ERROR_FUNC << "no backends support the requested descriptors";
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  sd.addStr("", "MemSection");

  ret = data->memorySection->serializePartial(sd, selected_engines, descs);
  if (ret != nixl_status_t::NIXL_SUCCESS)
    return ret;

  str = sd.exportStr();
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::loadRemoteMD(const blob_ref remote_metadata,
                                      std::string &agent_name) {
  int count = 0;
  nixlSerDes sd;
  size_t conn_cnt;
  nixl_blob_t conn_info;
  nixl_backend_t nixl_backend;
  nixlBackendEngine *eng;
  nixl_status_t ret;

  NIXL_LOCK_GUARD(data->lock);
  ret = sd.importBlob(remote_metadata);
  if (ret != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC << "failed to deserialize remote metadata";
    return ret;
  }

  std::string remote_agent = sd.getStr("Agent");
  if (remote_agent.size() == 0) {
    NIXL_ERROR_FUNC << "error in deserializing remote agent name";
    return nixl_status_t::NIXL_ERR_MISMATCH;
  }

  if (remote_agent == data->name) {
    NIXL_ERROR_FUNC << "remote agent name same as local agent, "
                       "no need to load metadata";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

  ret = sd.getBuf("Conns", &conn_cnt, sizeof(conn_cnt));
  if (ret != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC << "error getting connection count: " << ret;
    return ret;
  }

  for (size_t i = 0; i < conn_cnt; ++i) {
    nixl_backend = sd.getStr("t");
    if (nixl_backend.size() == 0)
      return nixl_status_t::NIXL_ERR_MISMATCH;
    conn_info = sd.getStr("c");
    if (conn_info.size() == 0)
      return nixl_status_t::NIXL_ERR_MISMATCH;

    // Current agent might not support a remote backend
    if (data->backendEngines.count(nixl_backend) != 0) {

      // No need to reload same conn info, error if it changed
      if (data->remoteBackends.count(remote_agent) != 0 &&
          data->remoteBackends[remote_agent].count(nixl_backend) != 0) {
        if (data->remoteBackends[remote_agent][nixl_backend] != conn_info)
          return nixl_status_t::NIXL_ERR_NOT_ALLOWED;
        count++;
        continue;
      }

      eng = data->backendEngines[nixl_backend];
      if (eng->supportsRemote()) {
        ret = eng->loadRemoteConnInfo(remote_agent, conn_info);
        if (ret != nixl_status_t::NIXL_SUCCESS) {
          NIXL_ERROR_FUNC << "error loading connection info for backend '"
                          << nixl_backend << "' with status " << ret;
          return ret; // Error in load
        }
        count++;
        data->remoteBackends[remote_agent].emplace(nixl_backend, conn_info);
      } else {
        // If there was an issue and we return error while some connections
        // are loaded, they will be deleted in the backend destructor.
        return nixl_status_t::NIXL_ERR_UNKNOWN; // This is an erroneous case
      }
    }
  }

  // No common backend, no point in loading the rest, unexpected
  if (count == 0 && conn_cnt > 0) {
    NIXL_ERROR_FUNC << "no common backend found";
    return nixl_status_t::NIXL_ERR_BACKEND;
  }

  if (sd.getStr("") != "MemSection")
    DESERIALIZE_ERR();

  if (data->remoteSections.count(remote_agent) == 0)
    data->remoteSections[remote_agent] = new nixlRemoteSection(remote_agent);

  ret = data->remoteSections[remote_agent]->loadRemoteData(
      &sd, data->backendEngines);

  // TODO: can be more graceful, if just the new MD blob was improper
  if (ret != nixl_status_t::NIXL_SUCCESS) {
    NIXL_ERROR_FUNC << "error loading remote metadata for agent '"
                    << remote_agent << "' with status " << ret;
    delete data->remoteSections[remote_agent];
    data->remoteSections.erase(remote_agent);
    data->remoteBackends.erase(remote_agent);
    return ret;
  }

  agent_name = remote_agent;
  return nixl_status_t::NIXL_SUCCESS;
}

nixl_status_t nixlAgent::invalidateRemoteMD(const std::string &remote_agent) {
  NIXL_LOCK_GUARD(data->lock);

  if (remote_agent == data->name) {
    NIXL_ERROR_FUNC
        << "remote agent same as local agent, cannot invalidate local metadata";
    return nixl_status_t::NIXL_ERR_INVALID_PARAM;
  }

  nixl_status_t ret = nixl_status_t::NIXL_ERR_NOT_FOUND;
  if (data->remoteSections.count(remote_agent) != 0) {
    delete data->remoteSections[remote_agent];
    data->remoteSections.erase(remote_agent);
    ret = nixl_status_t::NIXL_SUCCESS;
  }

  if (data->remoteBackends.count(remote_agent) != 0) {
    for (auto &it : data->remoteBackends[remote_agent])
      data->backendEngines[it.first]->disconnect(remote_agent);
    data->remoteBackends.erase(remote_agent);
    ret = nixl_status_t::NIXL_SUCCESS;
  }

  if (ret != nixl_status_t::NIXL_SUCCESS)
    NIXL_ERROR_FUNC << "error invalidating remote metadata for agent '"
                    << remote_agent << "' with status " << ret;
  return ret;
}

nixl_status_t
nixlAgent::sendLocalMD(const nixl_opt_args_t *extra_params) const {
  nixl_blob_t myMD;
  nixl_status_t ret = getLocalMD(myMD);
  if (isErrorStatus(ret)) {
    NIXL_ERROR_FUNC << "error getting local metadata with status " << ret;
    return ret;
  }

  // If IP is provided, use socket-based communication
  if (extra_params && !extra_params->ipAddr.empty()) {
    data->enqueueCommWork(std::make_tuple(nixl_comm_t::SOCK_SEND,
                                          extra_params->ipAddr,
                                          extra_params->port, std::move(myMD)));
    return nixl_status_t::NIXL_SUCCESS;
  }

  NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not "
                     "provided either";
  return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlAgent::sendLocalPartialMD(const nixl_reg_dlist_t &descs,
                              const nixl_opt_args_t *extra_params) const {
  nixl_blob_t myMD;
  nixl_status_t ret = getLocalPartialMD(descs, myMD, extra_params);
  if (isErrorStatus(ret)) {
    NIXL_ERROR_FUNC << "error getting local partial metadata with status "
                    << ret;
    return ret;
  }

  // If IP is provided, use socket-based communication
  if (extra_params && !extra_params->ipAddr.empty()) {
    data->enqueueCommWork(std::make_tuple(nixl_comm_t::SOCK_SEND,
                                          extra_params->ipAddr,
                                          extra_params->port, std::move(myMD)));
    return nixl_status_t::NIXL_SUCCESS;
  }

  NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not "
                     "provided either";
  return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t nixlAgent::fetchRemoteMD(const std::string &remote_name,
                                       const nixl_opt_args_t *extra_params) {
  // If IP is provided, use socket-based communication
  if (extra_params && !extra_params->ipAddr.empty()) {
    data->enqueueCommWork(std::make_tuple(
        nixl_comm_t::SOCK_FETCH, extra_params->ipAddr, extra_params->port, ""));
    return nixl_status_t::NIXL_SUCCESS;
  }

  NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not "
                     "provided either";
  return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t
nixlAgent::invalidateLocalMD(const nixl_opt_args_t *extra_params) const {
  // If IP is provided, use socket-based communication
  if (extra_params && !extra_params->ipAddr.empty()) {
    data->enqueueCommWork(std::make_tuple(
        nixl_comm_t::SOCK_INVAL, extra_params->ipAddr, extra_params->port, ""));
    return nixl_status_t::NIXL_SUCCESS;
  }

  NIXL_ERROR_FUNC << "ETCD is not supported and socket information was not "
                     "provided either";
  return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t nixlAgent::checkRemoteMD(const std::string &remote_name,
                                       const nixl_xfer_dlist_t &descs) const {
  NIXL_LOCK_GUARD(data->lock);
  if (data->remoteSections.count(remote_name) != 0) {
    if (descs.descCount() == 0) {
      return nixl_status_t::NIXL_SUCCESS;
    } else {
      nixl_meta_dlist_t dummy(descs.getType(), descs.isSorted());
      for (const auto &[backend, conn_info] : data->remoteBackends[remote_name])
        if (data->remoteSections[remote_name]->populate(
                descs, data->backendEngines[backend], dummy) ==
            nixl_status_t::NIXL_SUCCESS)
          return nixl_status_t::NIXL_SUCCESS;
      dummy.clear();
    }
  }
  return nixl_status_t::NIXL_ERR_NOT_FOUND;
}
