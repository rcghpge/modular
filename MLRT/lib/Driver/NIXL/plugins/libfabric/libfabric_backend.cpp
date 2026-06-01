//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/plugins/libfabric/libfabric_backend.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_topology.h"
#include "MLRT/Driver/NIXL/utils/serdes/serdes.h"

#include <charconv>
#include <cstring>
#include <limits>

#include <iomanip>
#include <numeric>

namespace M::MLRT {
using enum nixl_mem_t;
using enum nixl_status_t;
using enum nixl_xfer_op_t;

class LibfabricGpuContextCache {
public:
  M::ErrorOrSuccess getOrCreate(uint64_t devId,
                                M::Driver::DeviceContext *&context_out) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = contexts_.find(devId);
    if (it != contexts_.end()) {
      context_out = it->second.getPointer();
      return M::success();
    }

    auto device = M::Driver::Device::create(M::Driver::Device::gpuAPI,
                                            static_cast<int>(devId));
    if (device.isError()) {
      return device.takeError();
    }

    auto context = device.get()->createContext();
    if (context.isError()) {
      return context.takeError();
    }

    context_out = context.get().getPointer();
    contexts_.emplace(devId, context.takeValue());
    return M::success();
  }

  bool hasAnyGpuDevice() const {
    return M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI) > 0;
  }

private:
  mutable std::mutex mu_;
  std::unordered_map<uint64_t, M::Driver::DeviceContextRef> contexts_;
};

M::ErrorOrSuccess createGpuContextScope(
    LibfabricGpuContextCache &gpuContexts, uint64_t devId,
    std::unique_ptr<M::Driver::DeviceContextScope> &scope_out) {
  M::Driver::DeviceContext *context = nullptr;
  if (auto err = gpuContexts.getOrCreate(devId, context);
      const char *error = err.getError()) {
    return M::Error(error);
  }

  auto scope = std::make_unique<M::Driver::DeviceContextScope>(*context);
  auto err = scope->err();
  if (const char *error = err.getError()) {
    return M::Error(error);
  }

  scope_out = std::move(scope);
  return M::success();
}

/****************************************
 * Request Management
 *****************************************/

nixlLibfabricBackendH::nixlLibfabricBackendH()
    : completed_requests_(0), total_requests_used_(0), ref_count_(1) {
  NIXL_DEBUG << "constructor called, this: " << this
             << " total_requests_used=" << total_requests_used_.load();
}

nixlLibfabricBackendH::~nixlLibfabricBackendH() {
  NIXL_DEBUG << "destructor called, this: " << this;
}

// Multi-request completion tracking methods
void nixlLibfabricBackendH::init_request_tracking(size_t num_requests) {
  total_requests_used_.store(num_requests);
  completed_requests_.store(0);
  NIXL_DEBUG << "Initialized request tracking for " << num_requests
             << " requests";
}

void nixlLibfabricBackendH::increment_completed_requests() {
  // Snapshot ``total_requests_used_`` BEFORE the fetch_add so the log line
  // that follows does not dereference *this after the increment. After
  // fetch_add, the closure that called us proceeds to release(); the
  // matching retain() done at submit time is still live, so *this is
  // guaranteed alive for the NIXL_DEBUG that follows. Reading total before
  // the fetch_add isn't racy: total is set by init_request_tracking +
  // adjust_total_requests, both of which happen on the owner thread before
  // any callback can fire.
  size_t total_before = total_requests_used_.load(std::memory_order_acquire);
  size_t prev = completed_requests_.fetch_add(1, std::memory_order_acq_rel);
  if (prev + 1 == total_before) {
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
    completion_time_ns_.store(ns, std::memory_order_release);
  }
  NIXL_DEBUG << "Request completed, total completed: " << (prev + 1) << "/"
             << total_before;
}

int64_t nixlLibfabricBackendH::completion_time_ns() const {
  return completion_time_ns_.load(std::memory_order_acquire);
}

void nixlLibfabricBackendH::set_notif_xfer_id_overflow(size_t dropped,
                                                       size_t submitted) {
  notif_xfer_ids_submitted_.store(submitted, std::memory_order_release);
  notif_xfer_ids_dropped_.store(dropped, std::memory_order_release);
}

size_t nixlLibfabricBackendH::get_notif_xfer_ids_submitted() const {
  return notif_xfer_ids_submitted_.load(std::memory_order_acquire);
}

size_t nixlLibfabricBackendH::get_notif_xfer_ids_dropped() const {
  return notif_xfer_ids_dropped_.load(std::memory_order_acquire);
}

size_t nixlLibfabricBackendH::get_completed_requests_count() const {
  return completed_requests_.load();
}

size_t nixlLibfabricBackendH::get_total_requests_used() const {
  return total_requests_used_.load();
}

void nixlLibfabricBackendH::adjust_total_requests(size_t actual_count) {
  total_requests_used_.store(actual_count);
  NIXL_DEBUG << "Adjusted total requests to actual count: " << actual_count;
}

bool nixlLibfabricBackendH::is_completed() const {
  // Transfer is completed when at least every submitted request has
  // completed. ``>=`` (not strict ``==``) so that any bookkeeping
  // mismatch where ``completed_requests_`` overshoots
  // ``total_requests_used_`` doesn't wedge the sender into an
  // unrecoverable poll loop. The submission path is still expected to
  // set ``total_requests_used_`` to the exact count of in-flight RDMAs;
  // this is defense-in-depth.
  return completed_requests_.load() >= total_requests_used_.load();
}

void nixlLibfabricBackendH::retain() {
  ref_count_.fetch_add(1, std::memory_order_relaxed);
}

void nixlLibfabricBackendH::release() {
  // The ``acq_rel`` ordering on the decrement publishes any prior writes
  // to *this (e.g. completed_requests_ updates from the callback we just
  // ran) to whichever thread happens to drop the last reference. After
  // ``fetch_sub`` returns 1, the caller owns the only remaining reference
  // and ``delete this`` is the unique transition; no other thread can be
  // mid-method on *this because that would imply they hold a reference
  // they have not yet released, contradicting our seeing 1.
  if (ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete this;
  }
}

/****************************************
 * Constructor/Destructor
 *****************************************/

nixlLibfabricEngine::nixlLibfabricEngine(
    const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params), cm_thread_stop_(false),
      progress_thread_enabled_(init_params->enableProgTh),
      progress_thread_delay_(
          std::chrono::microseconds(init_params->pthrDelay)) {

  NIXL_DEBUG << "Initializing Libfabric Backend with GPU Support";
  gpu_contexts_ = std::make_unique<LibfabricGpuContextCache>();

  // Parse striping threshold parameter
  std::string threshold_str;
  striping_threshold_ = NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD;

  if (getInitParam("striping_threshold", threshold_str) == NIXL_SUCCESS) {
    uint64_t parsed_threshold = 0;
    auto *begin = threshold_str.data();
    auto *end = begin + threshold_str.size();
    auto parse_result = std::from_chars(begin, end, parsed_threshold);
    if (parse_result.ec == std::errc() && parse_result.ptr == end) {
      striping_threshold_ = parsed_threshold;
      NIXL_DEBUG << "Using custom striping threshold: " << striping_threshold_
                 << " bytes";
    } else {
      NIXL_WARN << "Invalid striping_threshold value '" << threshold_str
                << "', using default: " << striping_threshold_ << " bytes";
    }
  } else {
    NIXL_DEBUG << "Using default striping threshold: " << striping_threshold_
               << " bytes";
  }

  // Initialize Rail Manager which will discover the topology and create all
  // rails. Only request FI_HMEM when GPU devices are present — dKV and other
  // CPU-only agents must not request FI_HMEM because it causes EFA's fi_getinfo
  // to require an active CUDA context, making initialization fail in containers
  // without GPU device access.
  const bool enable_hmem = gpu_contexts_->hasAnyGpuDevice();
  NIXL_DEBUG << "Initializing rail manager with FI_HMEM "
             << (enable_hmem ? "enabled" : "disabled")
             << " (GPU devices present: " << enable_hmem << ")";
  auto rail_manager_or_error =
      nixlLibfabricRailManager::create(striping_threshold_, enable_hmem);
  if (rail_manager_or_error.isError()) {
    NIXL_ERROR << "Failed to create libfabric rail manager: "
               << rail_manager_or_error.getError();
    initErr = true;
    cleanup();
    return;
  }
  rail_manager = rail_manager_or_error.takeValue();
  NIXL_DEBUG << "Rail Manager created with " << rail_manager->getNumDataRails()
             << " data rails and " << rail_manager->getNumControlRails()
             << " control rails";

  // Set up callbacks on each rail using Engine's static callback functions
  size_t control_rail_id = 0;
  NIXL_DEBUG << "Set notification processor for control rail 0";
  rail_manager->getControlRail(control_rail_id)
      .setNotificationCallback([this](const std::string &serialized_notif) {
        processNotification(serialized_notif);
      });

  // Set up connection state callbacks for control rails
  NIXL_DEBUG << "Set connection state processor for CM rail 0";

  rail_manager->getControlRail(control_rail_id)
      .setConnectionAckCallback([this](const uint16_t agent_idx,
                                       nixlLibfabricConnection *conn_info,
                                       ConnectionState state) {
        processConnectionAck(agent_idx, conn_info, state);
      });

  // Set up connection request callback for control rails
  rail_manager->getControlRail(control_rail_id)
      .setConnectionReqCallback(
          [this](const uint16_t agent_idx, const std::string &serialized_data,
                 nixlLibfabricRail *rail) -> nixl_status_t {
            return processConnectionRequest(agent_idx, serialized_data, rail);
          });

  // Set up XFER_ID tracking callbacks for all data rails
  NIXL_DEBUG << "Setting up XFER_ID tracking callbacks for "
             << rail_manager->getNumDataRails() << " data rails";
  for (size_t data_rail_id = 0; data_rail_id < rail_manager->getNumDataRails();
       ++data_rail_id) {
    rail_manager->getDataRail(data_rail_id)
        .setXferIdCallback(
            [this](uint32_t xfer_id) { addReceivedXferId(xfer_id); });
    NIXL_DEBUG << "Set XFER_ID callback for data rail " << data_rail_id;
  }

  // Create self-connection
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints(
      rail_manager->getNumDataRails());
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> control_endpoints(
      rail_manager->getNumControlRails());
  for (size_t rail_id = 0; rail_id < rail_manager->getNumDataRails();
       ++rail_id) {
    std::memcpy(data_endpoints[rail_id].data(),
                rail_manager->getDataRail(rail_id).ep_name,
                sizeof(rail_manager->getDataRail(rail_id).ep_name));
  }
  for (size_t rail_id = 0; rail_id < rail_manager->getNumControlRails();
       ++rail_id) {
    std::memcpy(control_endpoints[rail_id].data(),
                rail_manager->getControlRail(rail_id).ep_name,
                sizeof(rail_manager->getControlRail(rail_id).ep_name));
  }
  nixl_status_t conn_status =
      createAgentConnection(localAgent, data_endpoints, control_endpoints);
  if (conn_status != NIXL_SUCCESS) {
    NIXL_ERROR
        << "createAgentConnection failed for self-connection with status: "
        << conn_status;
    initErr = true;
    cleanup();
    return;
  }

  NIXL_DEBUG << "Created self-connection for agent: " << localAgent << " on "
             << rail_manager->getNumDataRails() << " data rails and "
             << rail_manager->getNumControlRails() << " control rails";

  NIXL_DEBUG << "Starting CM thread";
  cm_thread_ = std::thread(&nixlLibfabricEngine::cmThread, this);
  if (!cm_thread_.joinable()) {
    NIXL_ERROR << "Failed to start CM thread";
    initErr = true;
    cleanup();
    return;
  }
  NIXL_DEBUG << "ConnectionManagement thread started successfully";

  if (progress_thread_enabled_) {
    NIXL_DEBUG << "Starting Progress thread for data rails with delay: "
               << progress_thread_delay_.count() << " microseconds";
    progress_thread_stop_ = false;
    progress_thread_ = std::thread(&nixlLibfabricEngine::progressThread, this);

    if (!progress_thread_.joinable()) {
      NIXL_ERROR << "Failed to start Progress thread";
      cm_thread_stop_.store(true);
      postShutdownCompletion();
      if (cm_thread_.joinable()) {
        cm_thread_.join();
      }
      initErr = true;
      cleanup();
      return;
    }
    NIXL_DEBUG << "Progress thread started successfully";
  } else {
    NIXL_DEBUG << "Progress thread disabled, using manual progress in "
                  "checkXfer/getNotifs";
  }
}

nixlLibfabricEngine::~nixlLibfabricEngine() {
  NIXL_DEBUG << "Destructor starting, stopping all threads FIRST to prevent "
                "timing report interruption";

  // STOP ALL THREADS FIRST to prevent any interference with timing report
  cm_thread_stop_.store(true);

  if (progress_thread_enabled_) {
    progress_thread_stop_.store(true);
  }

  // Post dummy completion to wake up blocking threads
  postShutdownCompletion();

  if (cm_thread_.joinable()) {
    NIXL_DEBUG << "Waiting for CM thread to exit";
    cm_thread_.join();
    NIXL_DEBUG << "CM thread joined successfully";
  }
  if (progress_thread_enabled_ && progress_thread_.joinable()) {
    NIXL_DEBUG << "Waiting for Progress thread to exit";
    progress_thread_.join();
    NIXL_DEBUG << "Progress thread joined successfully";
  } else if (!progress_thread_enabled_) {
    NIXL_DEBUG << "Progress thread was not running";
  }
  NIXL_DEBUG << "All threads stopped, now cleaning up resources";
  cleanup();
}

/****************************************
 * Connection management
 *****************************************/

nixl_status_t nixlLibfabricEngine::getConnInfo(std::string &str) const {
  // Verify all rail endpoints are initialized
  for (size_t rail_id = 0; rail_id < rail_manager->getNumDataRails();
       ++rail_id) {
    if (!rail_manager->getDataRail(rail_id).endpoint) {
      NIXL_ERROR << "Rail " << rail_id << " endpoint not initialized";
      return NIXL_ERR_BACKEND;
    }
  }

  NIXL_DEBUG << "Retrieving local endpoint addresses for all "
             << rail_manager->getNumDataRails() << " rails";

  // Use Rail Manager's connection SerDes method with "dest" prefix for remote
  // consumption
  nixl_status_t status = rail_manager->serializeConnectionInfo("dest", str);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail Manager serializeConnectionInfo failed";
    return status;
  }

  NIXL_DEBUG << "Rail Manager serialized connection info for "
             << rail_manager->getNumDataRails() << " rails, "
             << rail_manager->getNumControlRails() << " control rails, "
             << "total size: " << str.length();

  return NIXL_SUCCESS;
}

nixl_status_t
nixlLibfabricEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                        const std::string &remote_conn_info) {
  std::lock_guard<std::mutex> lock(connection_state_mutex_);

  NIXL_DEBUG << "Loading remote info for agent: " << remote_agent
             << ", info length: " << remote_conn_info.length()
             << ", info (hex): "
             << LibfabricUtils::hexdump(remote_conn_info.data());

  if (remote_conn_info.empty()) {
    NIXL_ERROR << "Empty remote connection info received";
    return NIXL_ERR_INVALID_PARAM;
  }

  NIXL_DEBUG << "Processing " << rail_manager->getNumDataRails()
             << " data rails and " << rail_manager->getNumControlRails()
             << " control rails for agent: " << remote_agent;

  // Use Rail Manager's connection SerDes method with "dest" prefix (remote is
  // sending us their endpoints as "dest")
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints;
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> control_endpoints;
  nixl_status_t status = rail_manager->deserializeConnectionInfo(
      "dest", remote_conn_info, data_endpoints, control_endpoints);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail Manager deserializeConnectionInfo failed";
    return status;
  }
  // Create connection to remote agent
  nixl_status_t conn_status =
      createAgentConnection(remote_agent, data_endpoints, control_endpoints);
  if (conn_status != NIXL_SUCCESS) {
    NIXL_ERROR << "createAgentConnection failed with status: " << conn_status;
    return conn_status;
  }

  NIXL_DEBUG << "Successfully stored multirail connection for " << remote_agent
             << " on " << rail_manager->getNumDataRails() << " rails";
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::connect(const std::string &remote_agent) {
  std::lock_guard<std::mutex> lock(connection_state_mutex_);

  NIXL_DEBUG << "Connecting to agent: " << remote_agent
             << ", connections_ size: " << connections_.size();

  // Check if connection is already established
  auto it = connections_.find(remote_agent);
  if (it != connections_.end() &&
      it->second->overall_state_ == ConnectionState::CONNECTED) {
    NIXL_DEBUG << "Connection already established for " << remote_agent
               << ", fi_addr: " << it->second->rail_remote_addr_list_[0];
    return NIXL_SUCCESS;
  }

  // Connection exists but not established - trigger establishConnection()
  NIXL_DEBUG << "Connection exists but not established, triggering "
                "establishConnection for "
             << remote_agent;

  // Release the lock before calling establishConnection since it acquires the
  // same mutex
  lock.~lock_guard();

  nixl_status_t status = establishConnection(remote_agent);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to establish connection with " << remote_agent;
    return status;
  }

  it = connections_.find(remote_agent);
  if (it == connections_.end()) {
    NIXL_DEBUG << "Connect failed. No metadata connection info for "
               << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  NIXL_DEBUG << "Successfully established connection for " << remote_agent;
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::disconnect(const std::string &remote_agent) {
  std::lock_guard<std::mutex> lock(connection_state_mutex_);
  auto it = connections_.find(remote_agent);
  if (it == connections_.end()) {
    NIXL_ERROR << "Disconnect failed. No metadata connection info for "
               << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }
  // Connection exists - check if already disconnected
  if (it->second->overall_state_ == ConnectionState::DISCONNECTED) {
    NIXL_DEBUG << "Connection already established for " << remote_agent
               << ", fi_addr: " << it->second->rail_remote_addr_list_[0];
    return NIXL_SUCCESS;
  }

  NIXL_DEBUG << "Disconnecting from agent: " << remote_agent;

  if (remote_agent != localAgent) {
    // Send disconnect control message to remote peer - fire and forget
    // semantics
    NIXL_DEBUG << "Sending disconnect notification to remote agent: "
               << remote_agent;

    // Use rail manager's serialization method with "src" prefix (we are sending
    // our source endpoints)
    std::string serialized_conn_info;
    nixl_status_t serialize_status =
        rail_manager->serializeConnectionInfo("src", serialized_conn_info);
    if (serialize_status != NIXL_SUCCESS) {
      NIXL_ERROR << "Rail manager serializeConnectionInfo failed";
      return serialize_status;
    }

    // Allocate control request
    const size_t control_rail_id = 0;
    const size_t buffer_size =
        (rail_manager->getNumDataRails() + rail_manager->getNumControlRails()) *
            LF_EP_NAME_MAX_LEN +
        1024;
    nixlLibfabricReq *control_request =
        rail_manager->getControlRail(control_rail_id)
            .allocateControlRequest(buffer_size);
    if (!control_request) {
      NIXL_ERROR << "Failed to allocate control request for disconnect";
      return NIXL_ERR_BACKEND;
    }

    memcpy(control_request->buffer, serialized_conn_info.data(),
           serialized_conn_info.length());

    // Set the actual size of serialized data
    control_request->buffer_size = serialized_conn_info.length();

    nixl_status_t status = rail_manager->postControlMessage(
        nixlLibfabricRailManager::ControlMessageType::DISCONNECT_REQ,
        control_request,
        it->second->control_rail_remote_addr_list_[0], // Use control rail 0
        it->second->agent_index_);

    if (status != NIXL_SUCCESS) {
      NIXL_WARN << "Failed to send disconnect notification to " << remote_agent
                << ", proceeding with local cleanup anyway";
      // Continue with cleanup even if notification failed
    } else {
      NIXL_DEBUG << "Disconnect notification sent successfully to "
                 << remote_agent;
    }
  } else {
    NIXL_DEBUG << "Skipping disconnect notification for self-connection";
  }

  // Clean up libfabric resources (AV entries) via rail manager
  NIXL_DEBUG << "Cleaning up libfabric resources for agent: " << remote_agent;
  // Clean up libfabric per-connection AV cleanup for both data and control
  // rails
  nixl_status_t data_cleanup_status =
      rail_manager->cleanupConnection(nixlLibfabricRailManager::RailType::DATA,
                                      it->second->rail_remote_addr_list_);
  if (data_cleanup_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to clean up data rail resources for agent: "
               << remote_agent << " with status: " << data_cleanup_status;
    return data_cleanup_status;
  }

  nixl_status_t control_cleanup_status = rail_manager->cleanupConnection(
      nixlLibfabricRailManager::RailType::CONTROL,
      it->second->control_rail_remote_addr_list_);
  if (control_cleanup_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to clean up control rail resources for agent: "
               << remote_agent << " with status: " << control_cleanup_status;
    return control_cleanup_status;
  }

  NIXL_DEBUG << "Successfully cleaned up libfabric resources for agent: "
             << remote_agent;

  // Update connection state to DISCONNECTED before removing
  it->second->overall_state_ = ConnectionState::DISCONNECTED;

  // Remove connection from map
  connections_.erase(remote_agent);
  NIXL_DEBUG << "Connection erased from the connection map for agent: "
             << remote_agent;

  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::createAgentConnection(
    const std::string &agent_name,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>
        &data_rail_endpoints,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>
        &control_rail_endpoints) {

  NIXL_DEBUG << "Creating connection for agent: " << agent_name;

  // Validate input parameters
  if (data_rail_endpoints.size() != rail_manager->getNumDataRails()) {
    NIXL_ERROR << "Expected " << rail_manager->getNumDataRails()
               << " data rail endpoints, got " << data_rail_endpoints.size();
    return NIXL_ERR_INVALID_PARAM;
  }

  if (control_rail_endpoints.size() != rail_manager->getNumControlRails()) {
    NIXL_ERROR << "Expected " << rail_manager->getNumControlRails()
               << " control rail endpoints, got "
               << control_rail_endpoints.size();
    return NIXL_ERR_INVALID_PARAM;
  }

  // Create connection object
  auto conn = std::make_shared<nixlLibfabricConnection>();
  if (!conn) {
    NIXL_ERROR << "Failed to allocate connection object";
    return NIXL_ERR_BACKEND;
  }

  conn->remoteAgent_ = agent_name;
  conn->overall_state_ = ConnectionState::DISCONNECTED;
  conn->rail_remote_addr_list_.reserve(rail_manager->getNumDataRails());
  conn->control_rail_remote_addr_list_.reserve(
      rail_manager->getNumControlRails());

  // Process all data rails in one operation
  nixl_status_t data_status = rail_manager->insertAllAddresses(
      nixlLibfabricRailManager::RailType::DATA, data_rail_endpoints,
      conn->rail_remote_addr_list_, conn->src_ep_names_);
  if (data_status != NIXL_SUCCESS) {
    NIXL_ERROR << "insertAllAddresses failed for data rails with status: "
               << data_status;
    return NIXL_ERR_BACKEND;
  }

  // Process all control rails in one operation
  nixl_status_t control_status = rail_manager->insertAllAddresses(
      nixlLibfabricRailManager::RailType::CONTROL, control_rail_endpoints,
      conn->control_rail_remote_addr_list_, conn->control_ep_names_);
  if (control_status != NIXL_SUCCESS) {
    NIXL_ERROR << "insertAllAddresses failed for control rails with status: "
               << control_status;
    return NIXL_ERR_BACKEND;
  }

  // Manage agent names and index
  agent_names_.push_back(agent_name);
  int index = 0;
  std::for_each(agent_names_.begin(), agent_names_.end(),
                [&index](const std::string &name) {
                  NIXL_DEBUG << "Index " << index << ": " << name;
                  index++;
                });
  conn->agent_index_ = agent_names_.size() - 1;

  // Store connection
  connections_[agent_name] = conn;

  NIXL_DEBUG << "Successfully created connection for agent: " << agent_name
             << " on " << rail_manager->getNumDataRails() << " data rails and "
             << rail_manager->getNumControlRails() << " control rails";

  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::establishConnection(
    const std::string &remote_agent) const {
  // Use existing connection_state_mutex_ to serialize connection establishment
  std::lock_guard<std::mutex> lock(connection_state_mutex_);

  // Check if another thread already established the connection
  auto it = connections_.find(remote_agent);
  if (it != connections_.end() &&
      it->second->overall_state_ == ConnectionState::CONNECTED) {
    NIXL_DEBUG << "Connection already established by another thread for "
               << remote_agent;
    return NIXL_SUCCESS;
  }

  if (it == connections_.end()) {
    NIXL_ERROR << "No connection found for agent: " << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  // Verify we have addresses for all data rails
  if (it->second->rail_remote_addr_list_.size() !=
      rail_manager->getNumDataRails()) {
    NIXL_ERROR << "Remote connection has "
               << it->second->rail_remote_addr_list_.size()
               << " data rails, expected " << rail_manager->getNumDataRails();
    return NIXL_ERR_BACKEND;
  }

  NIXL_DEBUG
      << "Establishing connections_ on control rails and data rails for agent: "
      << remote_agent;

  // Use single "Communicator" for CM
  auto *conn_info =
      reinterpret_cast<nixlLibfabricConnection *>(it->second.get());
  if (!conn_info) {
    NIXL_ERROR << "Connection info for agent " << remote_agent << " is null";
    return NIXL_ERR_BACKEND;
  }

  NIXL_DEBUG << "Using connection info with " << conn_info->src_ep_names_.size()
             << " data endpoints and " << conn_info->control_ep_names_.size()
             << " control endpoints for agent index: "
             << it->second->agent_index_;

  // Allocate control request
  const size_t control_rail_id = 0;

  // Serialize connection info
  std::string serialized_conn_info;
  nixl_status_t serialize_status =
      rail_manager->serializeConnectionInfo("src", serialized_conn_info);
  if (serialize_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail manager serializeConnectionInfo failed";
    return serialize_status;
  }

  nixlLibfabricReq *control_request =
      rail_manager->getControlRail(control_rail_id)
          .allocateControlRequest(serialized_conn_info.length());
  if (!control_request) {
    NIXL_ERROR
        << "Failed to allocate control request for connection establishment";
    return NIXL_ERR_BACKEND;
  }

  // Copy serialized data to control request buffer
  memcpy(control_request->buffer, serialized_conn_info.data(),
         serialized_conn_info.length());
  control_request->buffer_size = serialized_conn_info.length();

  nixl_status_t status = rail_manager->postControlMessage(
      nixlLibfabricRailManager::ControlMessageType::CONNECTION_REQ,
      control_request,
      conn_info->control_rail_remote_addr_list_[0], // Always use control rail 0
      it->second->agent_index_ // agent_index is only used in the ACK back from
                               // remote, to match connection request
  );
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "postSend failed on rail " << 0;
    // TODO, wrap req info into a nixlLibfabricRequestHandle and add retry logic
    return NIXL_ERR_BACKEND;
  }
  // Register the connection state tracker with the CM thread
  // Wait for the CM thread to establish the connection
  // TODO: Currently blocking, update to timeout and return NIXL_IN_PROG
  {
    std::unique_lock<std::mutex> lock(conn_info->conn_state_mutex_);
    NIXL_DEBUG << "Waiting for connection to be established for agent: "
               << remote_agent;
    conn_info->cv_.wait(lock, [conn_info] {
      return conn_info->overall_state_ == ConnectionState::CONNECTED ||
             conn_info->overall_state_ == ConnectionState::FAILED;
    });
    NIXL_DEBUG << "Connection state for agent " << remote_agent << " is now "
               << conn_info->overall_state_;

    if (conn_info->overall_state_ == ConnectionState::FAILED) {
      NIXL_ERROR << "Connection failed on control rail 0";
      return NIXL_ERR_BACKEND;
    }
  }

  NIXL_DEBUG << "Connection already established for agent: " << remote_agent;
  return NIXL_SUCCESS;
}

/****************************************
 * Memory management
 *****************************************/

nixl_mem_list_t nixlLibfabricEngine::getSupportedMems() const {
  nixl_mem_list_t mems;
  mems.push_back(DRAM_SEG);
  if (gpu_contexts_ && gpu_contexts_->hasAnyGpuDevice()) {
    mems.push_back(VRAM_SEG);
  }
  return mems;
}

nixl_status_t nixlLibfabricEngine::registerMem(const nixlBlobDesc &mem,
                                               const nixl_mem_t &nixl_mem,
                                               nixlBackendMD *&out) {
  auto priv = std::make_unique<nixlLibfabricPrivateMetadata>();
  std::unique_ptr<M::Driver::DeviceContextScope> gpu_scope;

  priv->buffer_ = (void *)mem.addr;
  priv->length_ = mem.len;
  priv->gpu_device_id_ =
      (nixl_mem == VRAM_SEG) ? static_cast<int>(mem.devId) : -1;

  if (nixl_mem == VRAM_SEG) {
    if (mem.devId == std::numeric_limits<uint64_t>::max()) {
      NIXL_ERROR << "VRAM registration requires a valid device ID";
      return NIXL_ERR_INVALID_PARAM;
    }

    auto gpu_scope_err =
        createGpuContextScope(*gpu_contexts_, mem.devId, gpu_scope);
    if (const char *error = gpu_scope_err.getError()) {
      NIXL_ERROR << "Failed to create GPU device context for device "
                 << mem.devId << ": " << error;
      return NIXL_ERR_NOT_SUPPORTED;
    }
  }

  // Initialize vectors to accommodate all possible rails (for indexing
  // consistency)
  priv->rail_mr_list_.resize(rail_manager->getNumDataRails(), nullptr);
  priv->rail_key_list_.resize(rail_manager->getNumDataRails(), 0);

  // Use Rail Manager for centralized memory registration with GPU Direct RDMA
  // support
  nixl_status_t status = rail_manager->registerMemory(
      (void *)mem.addr, mem.len, nixl_mem, mem.devId, priv->rail_mr_list_,
      priv->rail_key_list_, priv->selected_rails_);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail Manager registerMemory failed";
    return status;
  }

  NIXL_DEBUG << "Rail Manager successfully registered "
             << (nixl_mem == VRAM_SEG ? "VRAM" : "DRAM") << " memory on "
             << priv->selected_rails_.size() << " rails"
             << (nixl_mem == VRAM_SEG ? " with GPU Direct RDMA support" : "");

  NIXL_DEBUG << "Successfully registered memory on "
             << priv->selected_rails_.size() << " rails for "
             << (nixl_mem == VRAM_SEG ? "GPU" : "CPU") << " " << mem.devId;
  out = priv.release();
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::deregisterMem(nixlBackendMD *meta) {
  auto *priv = static_cast<nixlLibfabricPrivateMetadata *>(meta);
  // Use Rail Manager for centralized memory deregistration
  nixl_status_t status = rail_manager->deregisterMemory(priv->selected_rails_,
                                                        priv->rail_mr_list_);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail Manager deregisterMemory failed";
    // Continue with cleanup even if deregistration failed
  }

  delete priv;
  return status;
}

nixl_status_t nixlLibfabricEngine::getPublicData(const nixlBackendMD *meta,
                                                 std::string &str) const {
  const nixlLibfabricPrivateMetadata *priv =
      static_cast<const nixlLibfabricPrivateMetadata *>(meta);

  return rail_manager->serializeMemoryKeys(priv->rail_key_list_, priv->buffer_,
                                           str);
}

nixl_status_t nixlLibfabricEngine::loadLocalMD(nixlBackendMD *input,
                                               nixlBackendMD *&output) {
  nixlLibfabricPrivateMetadata *input_md =
      static_cast<nixlLibfabricPrivateMetadata *>(input);
  auto pub_md = std::make_unique<nixlLibfabricPublicMetadata>();
  // Store all rail keys instead of just the first one
  pub_md->rail_remote_key_list_.reserve(input_md->rail_key_list_.size());
  for (size_t rail_id = 0; rail_id < input_md->rail_key_list_.size();
       ++rail_id) {
    pub_md->rail_remote_key_list_.push_back(input_md->rail_key_list_[rail_id]);
    NIXL_DEBUG << "Added rail " << rail_id
               << " key: " << input_md->rail_key_list_[rail_id];
  }

  pub_md->remote_buf_addr_ = reinterpret_cast<uint64_t>(input_md->buffer_);
  pub_md->conn_ = connections_[localAgent];

  output = pub_md.release();
  NIXL_DEBUG << "Loading Local MD with " << input_md->rail_key_list_.size()
             << " rail keys";
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::loadRemoteMD(const nixlBlobDesc &input,
                                                const nixl_mem_t &nixl_mem,
                                                const std::string &remote_agent,
                                                nixlBackendMD *&output) {
  NIXL_DEBUG << "Loading remote metadata for agent: " << remote_agent;

  auto conn_it = connections_.find(remote_agent);
  if (conn_it == connections_.end()) {
    NIXL_ERROR << "Could not find connection for agent: " << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  // Delegate to Rail Manager for SerDes operations (returns raw data)
  std::vector<uint64_t> remote_keys;
  uint64_t remote_addr;
  nixl_status_t status = rail_manager->deserializeMemoryKeys(
      input.metaInfo, remote_keys, remote_addr);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Rail Manager deserializeMemoryKeys failed";
    return status;
  }

  // Engine handles connection management and metadata object creation
  auto pub_md = std::make_unique<nixlLibfabricPublicMetadata>();
  pub_md->conn_ = conn_it->second;
  pub_md->rail_remote_key_list_ = std::move(remote_keys);
  pub_md->remote_buf_addr_ = remote_addr;
  NIXL_DEBUG << "Remote metadata loaded with"
             << " Remote addr: " << (void *)pub_md->remote_buf_addr_
             << " Remote keys for " << pub_md->rail_remote_key_list_.size()
             << " rails"
             << " Remote fi_addr: " << pub_md->conn_->rail_remote_addr_list_[0];

  output = pub_md.release();
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::unloadMD(nixlBackendMD *input) {
  delete input;
  return NIXL_SUCCESS;
}

/****************************************
 * Data movement
 *****************************************/

nixl_status_t nixlLibfabricEngine::prepXfer(
    const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote, const std::string &remote_agent,
    nixlBackendReqH *&handle, const nixl_opt_b_args_t *opt_args) const {
  NIXL_DEBUG << "Preparing transfer for remote_agent: " << remote_agent;

  auto conn_it = connections_.find(remote_agent);
  if (conn_it == connections_.end() || !conn_it->second) {
    NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  auto backend_handle = new nixlLibfabricBackendH();
  if (!backend_handle) {
    NIXL_ERROR << "Failed to allocate nixlLibfabricBackendH";
    return NIXL_ERR_BACKEND;
  }
  handle = backend_handle; // Assign to base class pointer

  NIXL_DEBUG << "Transfer preparation complete, handle address: " << handle;
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::estimateXferCost(
    const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote, const std::string &remote_agent,
    nixlBackendReqH *const &handle, std::chrono::microseconds &duration,
    std::chrono::microseconds &err_margin, nixl_cost_t &method,
    const nixl_opt_args_t *opt_args) const {
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::postXfer(
    const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
    const nixl_meta_dlist_t &remote, const std::string &remote_agent,
    nixlBackendReqH *&handle, const nixl_opt_b_args_t *opt_args) const {

  // Validate connection
  auto conn_it = connections_.find(remote_agent);
  if (conn_it == connections_.end() || !conn_it->second) {
    NIXL_ERROR << "No valid connection found for agent: " << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  if (conn_it->second->overall_state_ == ConnectionState::DISCONNECTED) {
    NIXL_DEBUG << "No existing connection for " << remote_agent
               << ", establishing new connection";
    nixl_status_t status = this->establishConnection(remote_agent);
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to establish connection with " << remote_agent;
      return status;
    }
    NIXL_DEBUG << "Established new connection with remote_agent: "
               << remote_agent;
  }

  NIXL_DEBUG << "Posting transfer for remote_agent: " << remote_agent
             << ", handle address: " << handle;

  auto backend_handle = static_cast<nixlLibfabricBackendH *>(handle);
  if (!backend_handle) {
    NIXL_ERROR << "Failed to cast handle to nixlLibfabricBackendH";
    return NIXL_ERR_INVALID_PARAM;
  }

  // Allocate a new notification request at the start of each postXfer
  const size_t control_rail_id = 0;
  nixlLibfabricReq *control_request =
      rail_manager->getControlRail(control_rail_id)
          .allocateControlRequest(sizeof(BinaryNotification));
  if (!control_request) {
    NIXL_ERROR << "Failed to allocate control request for notification";
    return NIXL_ERR_BACKEND;
  }

  // GEX-3606: control_request is allocated up front and used as scratch
  // for binary_notif while submitting the data WRITEs. Ownership only
  // transfers off our hands when notifSendPriv succeeds — every other
  // exit (the error returns below, plus the no-notification success
  // path) must release it back to the pool, otherwise the 256-slot
  // control rail leaks one slot per call until allocation fails and
  // every subsequent transfer fails fast.
  auto release_control_request = [&]() {
    rail_manager->getControlRail(control_rail_id)
        .releaseRequest(control_request);
  };

  // Create BinaryNotification directly in the control request buffer
  BinaryNotification *binary_notif =
      reinterpret_cast<BinaryNotification *>(control_request->buffer);
  binary_notif->clear();

  nixlLibfabricReq::OpType op_type;
  int desc_count = local.descCount();

  NIXL_DEBUG << "Processing " << desc_count
             << " descriptors using optimized single-pass approach";

  op_type = (operation == NIXL_WRITE) ? nixlLibfabricReq::WRITE
                                      : nixlLibfabricReq::READ;

  // Set initial request count to maximum possible requests
  size_t max_possible_requests = desc_count * rail_manager->getNumDataRails();
  backend_handle->init_request_tracking(max_possible_requests);

  // Track the actual number of RDMAs submitted across every call to
  // ``prepareAndSubmitTransfer`` below. We can't infer this from
  // ``binary_notif->xfer_id_count``: that field is capped by the
  // notification's fixed ``NIXL_LIBFABRIC_MAX_XFER_IDS`` array, so on
  // large batches the notification count silently lags the true
  // submission count and the backend's completion-tracker would never
  // converge.
  size_t submitted_count = 0;

  // Core transfer submission to process each descriptor with direct
  // submission. On every error return below this lambda collapses the
  // initial ``init_request_tracking`` overestimate down to whatever was
  // actually submitted before the failure. Handle lifetime is owned by
  // the atomic refcount in retain()/release(); this adjustment is purely
  // for ``is_completed()`` status reporting via ``checkXfer``, so a
  // dropped transfer is reported "complete" once its in-flight requests
  // drain instead of staying IN_PROG forever waiting on submits that
  // never happened.
  auto adjust_for_error_return = [&] {
    backend_handle->adjust_total_requests(submitted_count);
  };
  for (int desc_idx = 0; desc_idx < desc_count; ++desc_idx) {
    auto *local_md =
        static_cast<nixlLibfabricPrivateMetadata *>(local[desc_idx].metadataP);
    auto *remote_md =
        static_cast<nixlLibfabricPublicMetadata *>(remote[desc_idx].metadataP);
    if (!local_md || !remote_md || !remote_md->conn_) {
      NIXL_ERROR << "Invalid metadata pointers for descriptor " << desc_idx;
      adjust_for_error_return();
      release_control_request();
      return NIXL_ERR_INVALID_PARAM;
    }

    // Validate connection for this descriptor
    if (remote_md->conn_ != conn_it->second) {
      NIXL_ERROR << "Connection mismatch for descriptor " << desc_idx;
      adjust_for_error_return();
      release_control_request();
      return NIXL_ERR_MISMATCH;
    }
    // Get transfer info for THIS descriptor
    void *transfer_addr = (void *)local[desc_idx].addr;
    size_t transfer_size = local[desc_idx].len;
    int gpu_id = local[desc_idx].devId;

    NIXL_DEBUG << "Processing descriptor " << desc_idx << " GPU " << gpu_id
               << " addr: " << transfer_addr << " size: " << transfer_size;

    // Prepare and submit transfer (use per-descriptor remote address, not
    // the metadata base address, so each descriptor targets the correct
    // offset within the remote buffer).
    //
    // The retain/release/completion triple anchors the lifetime of
    // ``backend_handle`` against in-flight progress-thread callbacks.
    // ``prepareAndSubmitTransfer`` calls ``retain_callback`` before each
    // libfabric post, so the count is bumped before the CQ entry can
    // fire the matching ``release()`` from inside the completion
    // closure. On post failure the function calls ``release_callback``
    // synchronously to balance the retain it just did.
    nixl_status_t status = rail_manager->prepareAndSubmitTransfer(
        op_type, transfer_addr, transfer_size, remote[desc_idx].addr,
        local_md->selected_rails_, local_md->rail_mr_list_,
        remote_md->rail_remote_key_list_,
        conn_it->second->rail_remote_addr_list_, conn_it->second->agent_index_,
        [backend_handle]() { backend_handle->retain(); },
        [backend_handle]() { backend_handle->release(); },
        [backend_handle]() {
          backend_handle->increment_completed_requests();
          backend_handle->release();
        },
        binary_notif, submitted_count);

    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "prepareAndSubmitTransfer failed for descriptor "
                 << desc_idx << " GPU " << gpu_id;
      adjust_for_error_return();
      release_control_request();
      return status;
    }

    NIXL_DEBUG << "Successfully processed descriptor " << desc_idx << " with "
               << submitted_count << " requests submitted";
  }

  NIXL_DEBUG << "Processing complete: submitted " << submitted_count
             << " requests from " << desc_count << " descriptors"
             << " with " << binary_notif->xfer_id_count << " total XFER_IDs";

  // Record BinaryNotification overflow on the backend handle so the
  // agent / Python transfer-engine layer can surface a structured
  // warning. The receiver still completes (data lands via RDMA WRITE
  // imm_data; the receiver only waits for the xfer_ids it was told
  // about), but its tracking set keeps the unaccounted arrivals as
  // stragglers. Recorded once per postXfer rather than per-xfer_id to
  // keep the cost trivial on large batches. `dropped == 0` is the
  // common no-overflow case.
  const size_t notif_dropped =
      (submitted_count > binary_notif->xfer_id_count)
          ? submitted_count - binary_notif->xfer_id_count
          : 0;
  backend_handle->set_notif_xfer_id_overflow(notif_dropped, submitted_count);

  // Adjust to actual request count after all submissions complete. Use
  // ``submitted_count`` (the true RDMA count) rather than the
  // notification's xfer_id_count, which is capped by
  // ``NIXL_LIBFABRIC_MAX_XFER_IDS`` and would otherwise leave the
  // backend's completion-tracker waiting on phantom missing requests.
  backend_handle->adjust_total_requests(submitted_count);

  // Send notification immediately after successful request submission
  if (opt_args && opt_args->hasNotif) {
    NIXL_DEBUG
        << "Sending immediate notification after successful request submission";

    // Set agent name and message in the BinaryNotification
    binary_notif->setAgentName(localAgent);
    binary_notif->setMessage(opt_args->notifMsg);

    nixl_status_t notif_status = notifSendPriv(remote_agent, control_request);
    if (notif_status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to send immediate notification";
      // GEX-3606: release control_request on error path. notifSendPriv
      // hands the request to the control rail's send queue on success;
      // on failure we still own it.
      release_control_request();
      return notif_status;
    }
    NIXL_DEBUG << "Immediate notification sent successfully with "
               << binary_notif->xfer_id_count << " XFER_IDs";
  } else {
    // No notification requested by the caller. The control_request was
    // allocated up front and used as scratch space for binary_notif
    // while submitting the data WRITEs, but it was never handed to
    // notifSendPriv, which is the only path that would otherwise
    // transfer ownership to the control rail's send queue and have its
    // CQ completion release it back to the pool. Release it here so
    // the control pool stays bounded across many postXfer calls
    // regardless of how the caller drives the API.
    release_control_request();
  }

  // Progress data rails to kick off transfers
  if (!progress_thread_enabled_) {
    nixl_status_t progress_status = rail_manager->progressActiveDataRails();
    if (progress_status == NIXL_IN_PROG) {
      return NIXL_IN_PROG;
    }
  }

  // For very small transfers we can check for local completions immediately.
  if (backend_handle->is_completed()) {
    return NIXL_SUCCESS;
  }

  return NIXL_IN_PROG;
}

nixl_status_t nixlLibfabricEngine::checkXfer(nixlBackendReqH *handle) const {
  auto backend_handle = static_cast<nixlLibfabricBackendH *>(handle);

  if (!progress_thread_enabled_) {
    nixl_status_t progress_status = rail_manager->progressActiveDataRails();
    if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
      NIXL_ERROR << "Failed to progress data rails in checkXfer";
      return progress_status;
    }
  }
  // Then check for completions after processing any pending completions
  if (backend_handle->is_completed()) {
    NIXL_DEBUG << "Data transfer completed successfully";
    return NIXL_SUCCESS;
  }
  return NIXL_IN_PROG;
}

nixl_status_t
nixlLibfabricEngine::getCompletionTime(const nixlBackendReqH *handle,
                                       chrono_point_t &out) const {
  const auto *backend_handle =
      static_cast<const nixlLibfabricBackendH *>(handle);
  int64_t ns = backend_handle->completion_time_ns();
  if (ns == 0) {
    return NIXL_IN_PROG;
  }
  out = chrono_point_t(std::chrono::duration_cast<chrono_point_t::duration>(
      std::chrono::nanoseconds(ns)));
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::getXferNotifOverflow(
    const nixlBackendReqH *handle, size_t &dropped, size_t &submitted) const {
  if (!handle) {
    return NIXL_ERR_INVALID_PARAM;
  }
  const auto *backend_handle =
      static_cast<const nixlLibfabricBackendH *>(handle);
  submitted = backend_handle->get_notif_xfer_ids_submitted();
  dropped = backend_handle->get_notif_xfer_ids_dropped();
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::releaseReqH(nixlBackendReqH *handle) const {
  // The NIXL agent layer (releaseXferReq + ~nixlXferReqH) calls into here
  // exactly once per ``nixlLibfabricBackendH*`` allocated by ``prepXfer``,
  // but does not itself ``delete`` the handle -- that's the backend's
  // responsibility. The previous stub left this as a no-op with a stale
  // "Let NIXL framework handle the deletion" comment, leaking one
  // ``nixlLibfabricBackendH`` per call.
  //
  // ``release()`` drops the owner reference seeded in the handle's
  // constructor; the progress-thread callbacks installed by
  // ``prepareAndSubmitTransfer`` each hold their own reference (taken via
  // ``retain_callback`` before each successful post) and drop it from the
  // matching ``release()`` inside the completion closure. ``delete this``
  // happens from whichever party drops the last reference -- typically the
  // last completion callback, but releaseReqH itself when no requests are
  // in flight (e.g. the rail-filter all-or-nothing rejection path).
  // Conditional checks like ``is_completed()`` are intentionally NOT used
  // here: an external decision based on ``completed_requests_`` would race
  // with the very callbacks that bump it, leading to either UAF (delete
  // while a callback is mid-method) or permanent leak (defer delete after
  // all callbacks already fired). The atomic refcount in ``release()``
  // resolves both at once.
  if (!handle) {
    return NIXL_SUCCESS;
  }
  static_cast<nixlLibfabricBackendH *>(handle)->release();
  NIXL_DEBUG << "releaseReqH dropped owner reference on backend handle";
  return NIXL_SUCCESS;
}

// notifSendPriv that accept control request
nixl_status_t
nixlLibfabricEngine::notifSendPriv(const std::string &remote_agent,
                                   nixlLibfabricReq *control_request) const {
  auto it = connections_.find(remote_agent);
  if (it == connections_.end()) {
    NIXL_ERROR << "No connection found for agent: " << remote_agent;
    return NIXL_ERR_NOT_FOUND;
  }

  auto connection = it->second;
  const size_t control_rail_id = 0; // Only use control rail 0 for notifications

  // Set the correct buffer size for the notification
  control_request->buffer_size = sizeof(BinaryNotification);

  // Get BinaryNotification from control request buffer for logging
  BinaryNotification *binary_notif =
      reinterpret_cast<BinaryNotification *>(control_request->buffer);

  NIXL_DEBUG << "Sending binary notification control request"
             << " Message: " << binary_notif->getMessage()
             << " xfer_id_count: " << binary_notif->xfer_id_count;
  nixl_status_t status = rail_manager->postControlMessage(
      nixlLibfabricRailManager::ControlMessageType::NOTIFICATION,
      control_request,
      connection->control_rail_remote_addr_list_[control_rail_id],
      connection->agent_index_);

  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "postControlMessage failed on control rail "
               << control_rail_id;
    return NIXL_ERR_BACKEND;
  }

  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricEngine::genNotif(const std::string &remote_agent,
                                            const std::string_view msg) const {
  // For regular notifications, we need to allocate a temporary control request
  // since we don't have a pre-allocated one from prepXfer
  const size_t control_rail_id = 0;
  nixlLibfabricReq *control_req =
      rail_manager->getControlRail(control_rail_id)
          .allocateControlRequest(sizeof(BinaryNotification));
  if (!control_req) {
    NIXL_ERROR << "Failed to allocate temporary control request for genNotif";
    return NIXL_ERR_BACKEND;
  }

  // Create BinaryNotification directly in the control buffer
  BinaryNotification *binary_notif =
      reinterpret_cast<BinaryNotification *>(control_req->buffer);
  binary_notif->clear();
  binary_notif->setAgentName(localAgent);
  binary_notif->setMessage(std::string(msg));

  return notifSendPriv(remote_agent, control_req);
}

nixl_status_t nixlLibfabricEngine::getNotifs(notif_list_t &notif_list) {
  if (!progress_thread_enabled_) {
    nixl_status_t progress_status = rail_manager->progressActiveDataRails();
    if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
      NIXL_ERROR << "Failed to progress data rails in getNotifs";
      return progress_status;
    }
  }

  // Then check for available notifications after processing completions
  // Thread-safe access to internal notification list
  {
    std::lock_guard<std::mutex> lock(notif_mutex_);

    // Move all notifications from internal list to user's list
    notif_list.insert(notif_list.end(), notifMainList_.begin(),
                      notifMainList_.end());

    if (!notifMainList_.empty()) {
      NIXL_DEBUG << "Retrieved " << notifMainList_.size() << " notifications";
      // Clear the internal list after copying
      notifMainList_.clear();
      return NIXL_SUCCESS;
    }

    // Clear the internal list after copying (even if empty)
    notifMainList_.clear();
  }

  return NIXL_IN_PROG;
}

/****************************************
 * ConnectionManagement Thread Function
 *****************************************/

// Background progress function that continuously processes completions on all
// rails
nixl_status_t nixlLibfabricEngine::cmThread() {
  NIXL_DEBUG << "ConnectionManagement thread started successfully";
  NIXL_DEBUG << "Initial receives already posted in main thread, entering "
                "progress loop";

  // Main progress loop - continuously process completions on all rails.
  // progressAllControlRails uses blocking fi_cq_sread when the hardware
  // supports it, falling back to non-blocking fi_cq_read otherwise.
  while (!cm_thread_stop_.load()) {
    rail_manager->progressAllControlRails();
    if (!rail_manager->getControlRail(0).blocking_cq_sread_supported) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(10));
    }
  }
  NIXL_DEBUG << "ConnectionManagement thread exiting cleanly";
  return NIXL_SUCCESS;
}

/****************************************
 * Progress Thread Function (Data Rails Only)
 *****************************************/

// Progress thread that continuously processes completions only on data rails
nixl_status_t nixlLibfabricEngine::progressThread() {
  NIXL_DEBUG << "Progress thread started successfully for data rails only";
  // Main progress loop - continuously process completions only on data rails
  while (!progress_thread_stop_.load()) {
    // Process completions only on data rails (non-blocking)
    bool any_completions = false;
    nixl_status_t status = rail_manager->progressActiveDataRails();
    if (status == NIXL_SUCCESS) {
      any_completions = true;
      NIXL_DEBUG << "Processed completions on data rails";
    } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to process completions on data rails";
      // Don't return error, continue for robustness
    }
    if (!any_completions) {
      std::this_thread::sleep_for(progress_thread_delay_);
    }
  }
  NIXL_DEBUG << "Progress thread exiting cleanly";
  return NIXL_SUCCESS;
}

void nixlLibfabricEngine::postShutdownCompletion() {
  NIXL_DEBUG << "Posting shutdown signal to wake up background thread";
  // Send shutdown message to self on rail 0 if self-connection exists
  auto self_conn_it = connections_.find(localAgent);
  if (self_conn_it != connections_.end() && self_conn_it->second &&
      rail_manager->getNumDataRails() > 0) {
    const size_t rail_id = 0; // Use rail 0 for shutdown signal

    // Allocate control request
    const size_t control_rail_id = 0;
    const size_t shutdown_msg_len = 8; // "SHUTDOWN" length
    nixlLibfabricReq *control_request =
        rail_manager->getControlRail(control_rail_id)
            .allocateControlRequest(shutdown_msg_len);
    if (!control_request) {
      NIXL_ERROR << "Failed to allocate control request for shutdown";
      return;
    }

    // Copy shutdown message to the control request buffer
    std::strcpy(static_cast<char *>(control_request->buffer), "SHUTDOWN");
    control_request->buffer_size = shutdown_msg_len;

    nixl_status_t status = rail_manager->postControlMessage(
        nixlLibfabricRailManager::ControlMessageType::DISCONNECT_REQ,
        control_request, self_conn_it->second->rail_remote_addr_list_[rail_id],
        self_conn_it->second->agent_index_);

    if (status == NIXL_SUCCESS) {
      NIXL_DEBUG << "Shutdown signal posted successfully on rail " << rail_id;
    } else {
      NIXL_ERROR << "Failed to post shutdown signal on rail " << rail_id;
    }
  } else {
    NIXL_ERROR << "Could not find self-connection or rails not initialized";
  }
}

/****************************************
 * Static Callback Functions
 *****************************************/

void nixlLibfabricEngine::processNotification(
    const std::string &serialized_notif) {
  // Only handle binary notification format
  // Check if this is a binary notification (fixed size)
  NIXL_DEBUG << "Received notification size: " << serialized_notif.size()
             << ", sizeof(BinaryNotification): " << sizeof(BinaryNotification);

  if (serialized_notif.size() != sizeof(BinaryNotification)) {
    NIXL_ERROR << "Invalid notification size: " << serialized_notif.size()
               << ", expected: " << sizeof(BinaryNotification);
    return;
  }

  // Process binary notification format
  const BinaryNotification *binary_notif =
      reinterpret_cast<const BinaryNotification *>(serialized_notif.data());

  std::string remote_name = binary_notif->getAgentName();
  std::string msg = binary_notif->getMessage();
  std::unordered_set<uint32_t> expected_xfer_ids = binary_notif->getXferIds();

  NIXL_TRACE << "Received binary notification from " << remote_name
             << " msg: " << msg
             << " xfer_id_count: " << binary_notif->xfer_id_count;

  // Check if this is a transfer notification that needs queuing
  if (!expected_xfer_ids.empty()) {
    std::stringstream xfer_ids_log;
    xfer_ids_log << "Expected XFER_IDs from binary notification: [";
    bool first = true;
    for (uint32_t xfer_id : expected_xfer_ids) {
      if (!first)
        xfer_ids_log << ", ";
      xfer_ids_log << xfer_id;
      first = false;
    }
    xfer_ids_log << "] (total: " << expected_xfer_ids.size() << ")";
    NIXL_TRACE << xfer_ids_log.str();

    // Count xfer_ids already in received_remote_writes_
    // BEFORE consuming them. Any pre-existing entry is a straggler from a
    // previous request that was reused (LIFO pool) for this new request.
    std::unordered_set<uint32_t> pre_existing_ids;
    {
      std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
      for (uint32_t xfer_id : expected_xfer_ids) {
        if (received_remote_writes_.count(xfer_id)) {
          pre_existing_ids.insert(xfer_id);
        }
      }
    }
    // Check if all expected XFER_IDs have already arrived. consume* erases
    // them on match so received_remote_writes_ does not grow unboundedly.
    if (consumeReceivedXferIds(expected_xfer_ids)) {
      NIXL_TRACE << "All XFER_IDs already received, processing binary "
                    "notification immediately";
      std::lock_guard<std::mutex> lock(notif_mutex_);
      notifMainList_.push_back({remote_name, msg});
      NIXL_DEBUG << "Binary notification processed immediately: " << msg;
    } else {
      NIXL_TRACE
          << "Not all XFER_IDs received yet, queuing binary notification";
      std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
      pending_notifications_.emplace_back(remote_name, msg, expected_xfer_ids,
                                          pre_existing_ids);
      NIXL_TRACE << "Binary notification queued for later processing: " << msg;
    }
  } else {
    // Regular notification without XFER_IDs - process immediately
    NIXL_TRACE
        << "Regular binary notification (no XFER_IDs), processing immediately";
    std::lock_guard<std::mutex> lock(notif_mutex_);
    notifMainList_.push_back({remote_name, msg});
    NIXL_TRACE << "Regular binary notification processed immediately: " << msg;
  }
}

void nixlLibfabricEngine::processConnectionAck(
    uint16_t agent_idx, nixlLibfabricConnection *conn_info,
    ConnectionState state) {
  std::string remote_agent_name = agent_names_[agent_idx];
  NIXL_DEBUG << "Connection state callback for agent " << remote_agent_name
             << " agent_idx: " << agent_idx;
  std::lock_guard<std::mutex> lock(
      connections_[remote_agent_name]->conn_state_mutex_);
  connections_[remote_agent_name]->overall_state_ = ConnectionState::CONNECTED;
  connections_[remote_agent_name]->cv_.notify_all();
  NIXL_DEBUG << "Connection state updated to CONNECTED";
}

nixl_status_t nixlLibfabricEngine::processConnectionRequest(
    uint16_t agent_idx, const std::string &serialized_data,
    nixlLibfabricRail *rail) {
  NIXL_DEBUG << "Processing connection request from agent " << agent_idx
             << " on rail " << rail->rail_id;

  // Use rail manager to deserialize ALL endpoints at once with "src" prefix
  // (connection request contains source endpoints)
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> data_endpoints;
  std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> control_endpoints;
  nixl_status_t status = rail_manager->deserializeConnectionInfo(
      "src", serialized_data, data_endpoints, control_endpoints);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to deserialize connection info";
    return status;
  }

  // Insert ALL data rail addresses at once
  std::vector<fi_addr_t> data_fi_addrs;
  std::vector<char *> data_ep_names;
  status = rail_manager->insertAllAddresses(
      nixlLibfabricRailManager::RailType::DATA, data_endpoints, data_fi_addrs,
      data_ep_names);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to insert data rail addresses";
    return status;
  }

  // Insert ALL control rail addresses at once
  std::vector<fi_addr_t> control_fi_addrs;
  std::vector<char *> control_ep_names;
  status = rail_manager->insertAllAddresses(
      nixlLibfabricRailManager::RailType::CONTROL, control_endpoints,
      control_fi_addrs, control_ep_names);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to insert control rail addresses";
    return status;
  }

  // Use the first control rail's fi_addr for ACK (same as before)
  fi_addr_t initiator_control_fi_addr = control_fi_addrs[0];

  NIXL_DEBUG << "Successfully inserted addresses for " << data_fi_addrs.size()
             << " data rails and " << control_fi_addrs.size()
             << " control rails"
             << ", initiator_control_fi_addr: " << initiator_control_fi_addr;

  // Send acknowledgement back to the initiator using the rail manager
  size_t ep_name_len = sizeof(rail->ep_name);

  // Allocate control request
  const size_t control_rail_id = 0;
  nixlLibfabricReq *control_request =
      rail_manager->getControlRail(control_rail_id)
          .allocateControlRequest(ep_name_len);
  if (!control_request) {
    NIXL_ERROR << "Failed to allocate control request for connection ACK";
    return NIXL_ERR_BACKEND;
  }

  // Copy endpoint name to control request buffer
  std::memcpy(control_request->buffer, rail->ep_name, ep_name_len);
  control_request->buffer_size = ep_name_len;

  nixl_status_t ack_status = rail_manager->postControlMessage(
      nixlLibfabricRailManager::ControlMessageType::CONNECTION_ACK,
      control_request, initiator_control_fi_addr, agent_idx);
  if (ack_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to send ACK via rail manager";
    return ack_status;
  }

  NIXL_DEBUG << "ACK sent successfully via rail manager";
  return NIXL_SUCCESS;
}

/****************************************
 * Receiver Side XFER_ID Tracking Helper Methods
 *****************************************/

void nixlLibfabricEngine::addReceivedXferId(uint32_t xfer_id) {
  size_t map_size;
  uint32_t new_count;
  {
    std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
    // [GEX-3718] Increment the per-xfer_id outstanding-write count. xfer_id
    // values are recycled (LIFO pool) across concurrent transfers, so multiple
    // in-flight transfers can hold the same xfer_id at the same time. A count,
    // rather than a set, lets us distinguish "still outstanding for some
    // transfer" from "fully consumed".
    new_count = ++received_remote_writes_[xfer_id];
    map_size = received_remote_writes_.size();
    NIXL_DEBUG << "Added received XFER_ID " << xfer_id
               << " to global tracking map (count: " << new_count
               << ", unique total: " << map_size << ")";
  }

  // Check if any pending notifications can now be processed
  checkPendingNotifications();
}

bool nixlLibfabricEngine::consumeReceivedXferIds(
    const std::unordered_set<uint32_t> &expected) {
  std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
  // Verify all expected XFER_IDs are present before mutating the map.
  for (uint32_t xfer_id : expected) {
    if (received_remote_writes_.find(xfer_id) ==
        received_remote_writes_.end()) {
      NIXL_TRACE << "XFER_ID " << xfer_id << " not yet received";
      return false;
    }
  }
  // [GEX-3718] Decrement each matched xfer_id's count. Only erase the entry
  // when the count reaches 0, so outstanding references held by other
  // concurrent transfers sharing the same xfer_id are preserved.
  for (uint32_t xfer_id : expected) {
    auto it = received_remote_writes_.find(xfer_id);
    if (it->second <= 1) {
      received_remote_writes_.erase(it);
    } else {
      --it->second;
    }
  }
  return true;
}

/****************************************
 * Notification Queuing Helper Methods
 *****************************************/

void nixlLibfabricEngine::checkPendingNotifications() {
  std::lock_guard<std::mutex> lock(receiver_tracking_mutex_);
  auto it = pending_notifications_.begin();
  while (it != pending_notifications_.end()) {
    // Check if all expected XFER_IDs for this notification have arrived
    bool all_received = true;
    for (uint32_t xfer_id : it->expected_xfer_ids) {
      if (received_remote_writes_.find(xfer_id) ==
          received_remote_writes_.end()) {
        all_received = false;
        break;
      }
    }

    if (all_received) {
      NIXL_TRACE
          << "All XFER_IDs received for queued notification, processing now";

      // Move notification to main list (need to acquire notif_mutex_)
      {
        std::lock_guard<std::mutex> notif_lock(notif_mutex_);
        notifMainList_.push_back({it->remote_agent, it->message});
      }

      // [GEX-3718] Decrement matched xfer_id counts. Only erase the entry
      // when the count reaches 0, preserving references held by other
      // concurrent transfers that share xfer_ids. Safe under the held
      // receiver_tracking_mutex_.
      for (uint32_t xfer_id : it->expected_xfer_ids) {
        auto rrw_it = received_remote_writes_.find(xfer_id);
        if (rrw_it->second <= 1) {
          received_remote_writes_.erase(rrw_it);
        } else {
          --rrw_it->second;
        }
      }
      NIXL_TRACE << "Processed queued notification: " << it->message;

      // Remove from pending list
      it = pending_notifications_.erase(it);
    } else {
      ++it;
    }
  }
}

void nixlLibfabricEngine::cleanup() {
  NIXL_DEBUG << "Cleaning up all resources";
  gpu_contexts_.reset();

  NIXL_DEBUG << "Cleanup all resources complete";
}

} // namespace M::MLRT
