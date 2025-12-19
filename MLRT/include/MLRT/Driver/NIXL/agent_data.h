//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_AGENT_DATA_H
#define ASYNCRT_NIXL_AGENT_DATA_H

#include "MLRT/Driver/NIXL/mem_section.h"
#include "MLRT/Driver/NIXL/sync.h"
#include "MLRT/Driver/NIXL/utils/common/str_tools.h"
#include "MLRT/Driver/NIXL/utils/stream/metadata_stream.h"

namespace M::AsyncRT::NIXL {

// Internal typedef to define metadata communication request types
// To be extended with ETCD operations
enum class nixl_comm_t {
  SOCK_SEND,
  SOCK_FETCH,
  SOCK_INVAL,
  SOCK_MAX,
};

// Command to be sent to listener thread from NIXL API
//  1) Command type
//  2) IP Address
//  3) Port
//  4) Metadata to send (for sendLocalMD calls)
using nixl_comm_req_t = std::tuple<nixl_comm_t, std::string, int, nixl_blob_t>;

using nixl_socket_peer_t = std::pair<std::string, int>;

class nixlAgentData {
public:
  nixlAgentData(std::string agentName, nixlAgentConfig cfg);
  ~nixlAgentData();
  nixlAgentData(const nixlAgentData &) = delete;
  nixlAgentData &operator=(const nixlAgentData &) = delete;
  nixlAgentData(nixlAgentData &&) = delete;
  nixlAgentData &operator=(nixlAgentData &&) = delete;

private:
  std::string name;
  nixlAgentConfig config;
  nixlLock lock;

  // some handle that can be used to instantiate an object from the lib
  std::map<std::string, void *> backendLibs;

  // Bookkeeping from backend type and memory type to backend engine
  backend_list_t notifEngines;
  backend_map_t backendEngines;
  std::array<backend_list_t, to_underlying(nixl_mem_t::COUNT)> memToBackend;

  // Bookkeeping for local connection metadata and user handles per backend
  std::unordered_map<nixl_backend_t, nixlBackendH *> backendHandles;
  std::unordered_map<nixl_backend_t, nixl_blob_t> connMD;

  // Local section, and Remote sections and their available common backends
  nixlLocalSection *memorySection;

  std::unordered_map<std::string,
                     std::unordered_map<nixl_backend_t, nixl_blob_t>,
                     std::hash<std::string>, strEqual>
      remoteBackends;
  std::unordered_map<std::string, nixlRemoteSection *, std::hash<std::string>,
                     strEqual>
      remoteSections;

  // State/methods for listener thread
  nixlMDStreamListener *listener;
  std::map<nixl_socket_peer_t, int> remoteSockets;
  std::thread commThread;
  std::vector<nixl_comm_req_t> commQueue;
  std::mutex commLock;
  volatile bool commThreadStop = false;

  void commWorker(nixlAgent *myAgent);
  void enqueueCommWork(nixl_comm_req_t request);
  void getCommWork(std::vector<nixl_comm_req_t> &req_list);

  friend class nixlAgent;
};

class nixlBackendEngine;
// This class hides away the nixlBackendEngine from user of the Agent API
class nixlBackendH {
private:
  nixlBackendEngine *engine;

  // TODO(GENAI-170): nixlBackendH(nixlBackendEngine &engine)
  nixlBackendH(nixlBackendEngine *engine) : engine{engine} {}

public:
  nixl_backend_t getType() const { return engine->getType(); }

  bool supportsRemote() const { return engine->supportsRemote(); }
  bool supportsLocal() const { return engine->supportsLocal(); }
  bool supportsNotif() const { return engine->supportsNotif(); }
  bool supportsProgTh() const { return engine->supportsProgTh(); }

  friend class nixlAgentData;
  friend class nixlAgent;
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_AGENT_DATA_H
