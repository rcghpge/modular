//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_rail_manager.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_topology.h"
#include "MLRT/Driver/NIXL/utils/serdes/serdes.h"
#include "llvm/Support/ErrorHandling.h"

// Static round-robin counter for rail selection
namespace M::MLRT {
using enum nixl_mem_t;
using enum nixl_status_t;
using enum nixl_xfer_op_t;

static std::atomic<size_t> round_robin_counter{0};

M::ErrorOr<std::unique_ptr<nixlLibfabricRailManager>>
nixlLibfabricRailManager::create(size_t striping_threshold, bool enable_hmem) {
  auto rail_manager = std::unique_ptr<nixlLibfabricRailManager>(
      new nixlLibfabricRailManager(striping_threshold, enable_hmem));
  if (rail_manager->getInitStatus() != NIXL_SUCCESS) {
    return M::Error(rail_manager->getInitError().empty()
                        ? "Failed to initialize libfabric rail manager."
                        : rail_manager->getInitError());
  }
  return M::ErrorOr<std::unique_ptr<nixlLibfabricRailManager>>(
      std::move(rail_manager));
}

nixlLibfabricRailManager::nixlLibfabricRailManager(size_t striping_threshold,
                                                   bool enable_hmem)
    : striping_threshold_(striping_threshold), enable_hmem_(enable_hmem) {
  NIXL_DEBUG << "Creating rail manager with striping threshold: "
             << striping_threshold_ << " bytes";

  auto topology_or_error = nixlLibfabricTopology::create();
  if (topology_or_error.isError()) {
    init_status_ = NIXL_ERR_BACKEND;
    init_error_ = topology_or_error.getError();
    NIXL_ERROR << "Failed to discover system topology: " << init_error_;
    return;
  }
  topology = topology_or_error.takeValue();
  NIXL_DEBUG << "System topology discovered successfully";

  // Get EFA devices from topology and create rails automatically
  std::vector<std::string> all_efa_devices = topology->getAllEfaDevices();
  NIXL_DEBUG << "Got " << all_efa_devices.size()
             << " EFA devices from topology";

  // Create data rails
  nixl_status_t rail_status = createDataRails(all_efa_devices);
  if (rail_status != NIXL_SUCCESS) {
    init_status_ = rail_status;
    if (init_error_.empty()) {
      init_error_ = "Rail Manager failed to create data rails.";
    }
    return;
  }
  // Create control rails
  nixl_status_t control_rail_status =
      createControlRails(all_efa_devices, NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS);
  if (control_rail_status != NIXL_SUCCESS) {
    init_status_ = control_rail_status;
    if (init_error_.empty()) {
      init_error_ = "Rail Manager failed to create control rails.";
    }
    return;
  }
  NIXL_DEBUG << "Successfully created " << data_rails_.size()
             << " data rails and " << control_rails_.size() << " control rails";
}

nixlLibfabricRailManager::~nixlLibfabricRailManager() {
  NIXL_DEBUG << "Destroying rail manager";
}

nixl_status_t nixlLibfabricRailManager::createDataRails(
    const std::vector<std::string> &efa_devices) {
  num_data_rails_ = efa_devices.size();
  // Pre-allocate to ensure contiguous memory allocation
  data_rails_.reserve(num_data_rails_);

  // Build EFA device to rail index mapping for O(1) lookup
  efa_device_to_rail_map.reserve(num_data_rails_);

  data_rails_.clear();
  data_rails_.reserve(num_data_rails_);

  for (size_t i = 0; i < num_data_rails_; ++i) {
    auto rail_or_error = nixlLibfabricRail::create(
        efa_devices[i], static_cast<uint16_t>(i), enable_hmem_);
    if (rail_or_error.isError()) {
      init_error_ = rail_or_error.getError();
      NIXL_ERROR << "Failed to create data rail " << i << ": " << init_error_;
      return NIXL_ERR_BACKEND;
    }

    data_rails_.emplace_back(rail_or_error.takeValue());
    efa_device_to_rail_map[efa_devices[i]] = i;

    NIXL_DEBUG << "Created data rail " << i << " (device: " << efa_devices[i]
               << ")";
  }
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::createControlRails(
    const std::vector<std::string> &efa_devices, size_t num_control_rails) {
  // Pre-allocate to ensure contiguous memory allocation
  num_control_rails_ = num_control_rails;
  control_rails_.reserve(num_control_rails_);

  control_rails_.clear();
  control_rails_.reserve(num_control_rails_);

  for (size_t i = 0; i < num_control_rails_; ++i) {
    auto rail_or_error = nixlLibfabricRail::create(
        efa_devices[i], static_cast<uint16_t>(i), enable_hmem_);
    if (rail_or_error.isError()) {
      init_error_ = rail_or_error.getError();
      NIXL_ERROR << "Failed to create control rail " << i << ": "
                 << init_error_;
      return NIXL_ERR_BACKEND;
    }
    control_rails_.emplace_back(rail_or_error.takeValue());
    NIXL_DEBUG << "Created control rail " << i << " (device: " << efa_devices[i]
               << ")";
  }
  return NIXL_SUCCESS;
}

bool nixlLibfabricRailManager::shouldUseStriping(size_t transfer_size) const {
  return transfer_size >= striping_threshold_;
}

nixl_status_t nixlLibfabricRailManager::prepareAndSubmitTransfer(
    nixlLibfabricReq::OpType op_type, void *local_addr, size_t transfer_size,
    uint64_t remote_base_addr, const std::vector<size_t> &selected_rails,
    const std::vector<struct fid_mr *> &local_mrs,
    const std::vector<uint64_t> &remote_keys,
    const std::vector<fi_addr_t> &dest_addrs, uint16_t agent_idx,
    std::function<void()> retain_callback,
    std::function<void()> release_callback,
    std::function<void()> completion_callback, BinaryNotification *binary_notif,
    size_t &submitted_count) {
  if (selected_rails.empty()) {
    NIXL_ERROR << "No rails selected for transfer";
    return NIXL_ERR_INVALID_PARAM;
  }

  // Filter to rails that BOTH endpoints actually registered memory on.
  // VRAM picks rails by GPU<->EFA topology; DRAM picks by NUMA<->EFA
  // topology (see selectRailsForMemory). For heterogeneous transfers
  // (e.g. dKV server DRAM <-> MAX worker VRAM) those subsets can
  // diverge, leaving zero entries in remote_keys for some local rails.
  // Posting fi_writedata against rkey=0 produces an EFA NACK; GEX-3606's
  // CQ-error release handler then advances the transfer by faking
  // completion to break the cascade, but the data never arrives so the
  // receiver silently sees stale memory. Filtering up front keeps the
  // submitted set well-formed and avoids the corruption path entirely.
  std::vector<size_t> valid_rails;
  valid_rails.reserve(selected_rails.size());
  size_t dropped_oob = 0;
  size_t dropped_missing_local_mr = 0;
  size_t dropped_missing_remote_key = 0;
  for (size_t rail_id : selected_rails) {
    if (rail_id >= local_mrs.size() || rail_id >= remote_keys.size() ||
        rail_id >= dest_addrs.size()) {
      ++dropped_oob;
      continue;
    }
    if (local_mrs[rail_id] == nullptr) {
      ++dropped_missing_local_mr;
      continue;
    }
    if (remote_keys[rail_id] == 0) {
      ++dropped_missing_remote_key;
      continue;
    }
    valid_rails.push_back(rail_id);
  }

  if (valid_rails.empty()) {
    NIXL_ERROR << "No rails have both local mr and remote key registered:"
               << " local_selected=" << selected_rails.size()
               << " dropped_oob=" << dropped_oob
               << " dropped_missing_local_mr=" << dropped_missing_local_mr
               << " dropped_missing_remote_key=" << dropped_missing_remote_key
               << "; check VRAM (GPU topology) vs DRAM (NUMA topology) rail"
               << " selection across the two endpoints.";
    return NIXL_ERR_INVALID_PARAM;
  }

  if (valid_rails.size() < selected_rails.size()) {
    NIXL_DEBUG << "Filtered rails for transfer: " << selected_rails.size()
               << " -> " << valid_rails.size()
               << " (dropped: oob=" << dropped_oob
               << " missing_local_mr=" << dropped_missing_local_mr
               << " missing_remote_key=" << dropped_missing_remote_key << ")";
  }

  // Determine striping strategy against the filtered rail set so we don't
  // stripe across rails that can't carry the transfer.
  bool use_striping =
      shouldUseStriping(transfer_size) && valid_rails.size() > 1;

  // Allocate a data request, draining completed requests inline if the pool
  // is temporarily full. The calling thread can poll the data CQ directly
  // without cross-thread coordination.
  static constexpr auto kDrainTimeout = std::chrono::milliseconds(100);
  auto allocateWithDrain =
      [&](size_t rail_id, nixlLibfabricReq::OpType op) -> nixlLibfabricReq * {
    nixlLibfabricReq *req = data_rails_[rail_id]->allocateDataRequest(op);
    if (!req) {
      auto deadline = std::chrono::steady_clock::now() + kDrainTimeout;
      do {
        progressActiveDataRails();
        req = data_rails_[rail_id]->allocateDataRequest(op);
      } while (!req && std::chrono::steady_clock::now() < deadline);
    }
    return req;
  };

  if (!use_striping) {
    // Round-robin: use one rail for entire transfer
    size_t rail_idx = round_robin_counter.fetch_add(1) % valid_rails.size();
    size_t rail_id = valid_rails[rail_idx];
    // Allocate request, draining the CQ inline if the pool is temporarily full.
    nixlLibfabricReq *req = allocateWithDrain(rail_id, op_type);
    if (!req) {
      NIXL_ERROR << "Failed to allocate data request for rail " << rail_id
                 << " after " << kDrainTimeout.count() << "ms drain timeout";
      return NIXL_ERR_BACKEND;
    }
    // Retain BEFORE installing the completion callback or posting so the
    // caller's reference count is bumped before the progress thread can
    // observe a completion entry for this request and run the callback's
    // matching release. If post fails below, we release_callback() to
    // undo this retain.
    retain_callback();
    // Set callbacks and populate request
    req->completion_callback = completion_callback;
    req->chunk_offset = 0;
    req->chunk_size = transfer_size;
    req->local_addr = local_addr;
    req->remote_addr = remote_base_addr;
    req->local_mr = local_mrs[rail_id];
    req->remote_key = remote_keys[rail_id];
    req->rail_id = rail_id;
    // Submit immediately
    nixl_status_t status;
    if (op_type == nixlLibfabricReq::WRITE) {
      uint64_t imm_data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER,
                                             agent_idx, req->xfer_id);
      status = data_rails_[rail_id]->postWrite(
          req->local_addr, req->chunk_size, fi_mr_desc(req->local_mr), imm_data,
          dest_addrs[rail_id], req->remote_addr, req->remote_key, req);
    } else {
      status = data_rails_[rail_id]->postRead(
          req->local_addr, req->chunk_size, fi_mr_desc(req->local_mr),
          dest_addrs[rail_id], req->remote_addr, req->remote_key, req);
    }
    if (status != NIXL_SUCCESS) {
      // Post never reached libfabric, so the closure won't fire. Undo
      // the retain we just did and recycle the request.
      release_callback();
      data_rails_[rail_id]->releaseRequest(req);
      return status;
    }

    // Track the actual submission. ``binary_notif->addXferId`` may drop
    // the xfer_id when the notification's fixed-size array is full
    // (NIXL_LIBFABRIC_MAX_XFER_IDS); ``submitted_count`` is the source
    // of truth for the backend's completion accounting either way.
    // Drop is intentionally silent here; the backend logs a single
    // aggregate WARN at the end of postXfer when it observes
    // ``submitted_count > binary_notif->xfer_id_count``.
    ++submitted_count;
    if (binary_notif) {
      binary_notif->addXferId(req->xfer_id);
    }

    NIXL_DEBUG << "Round-robin: submitted single request on rail " << rail_id
               << " for " << transfer_size
               << " bytes, XFER_ID: " << req->xfer_id;

  } else {
    // Striping: distribute across multiple rails
    size_t num_rails = valid_rails.size();
    size_t chunk_size = transfer_size / num_rails;
    size_t remainder = transfer_size % num_rails;
    for (size_t i = 0; i < num_rails; ++i) {
      size_t rail_id = valid_rails[i];
      size_t current_chunk_size =
          chunk_size + (i == num_rails - 1 ? remainder : 0);
      if (current_chunk_size == 0)
        break;
      // Allocate request, draining the CQ inline if the pool is temporarily
      // full.
      nixlLibfabricReq *req = allocateWithDrain(rail_id, op_type);
      if (!req) {
        NIXL_ERROR << "Failed to allocate data request for rail " << rail_id
                   << " after " << kDrainTimeout.count() << "ms drain timeout";
        return NIXL_ERR_BACKEND;
      }

      // See round-robin branch for retain ordering rationale.
      retain_callback();
      req->completion_callback = completion_callback;

      // Calculate and populate chunk info
      size_t chunk_offset = i * chunk_size;
      req->chunk_offset = chunk_offset;
      req->chunk_size = current_chunk_size;
      req->local_addr = static_cast<char *>(local_addr) + chunk_offset;
      req->remote_addr = remote_base_addr + chunk_offset;
      req->local_mr = local_mrs[rail_id];
      req->remote_key = remote_keys[rail_id];
      req->rail_id = rail_id;
      nixl_status_t status;
      if (op_type == nixlLibfabricReq::WRITE) {
        uint64_t imm_data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER,
                                               agent_idx, req->xfer_id);
        status = data_rails_[rail_id]->postWrite(
            req->local_addr, req->chunk_size, fi_mr_desc(req->local_mr),
            imm_data, dest_addrs[rail_id], req->remote_addr, req->remote_key,
            req);
      } else {
        status = data_rails_[rail_id]->postRead(
            req->local_addr, req->chunk_size, fi_mr_desc(req->local_mr),
            dest_addrs[rail_id], req->remote_addr, req->remote_key, req);
      }
      if (status != NIXL_SUCCESS) {
        release_callback();
        data_rails_[rail_id]->releaseRequest(req);
        return status;
      }

      // See round-robin branch for the rationale on tracking
      // ``submitted_count`` independently of the notification capacity
      // and on leaving overflow logging to the postXfer aggregate.
      ++submitted_count;
      if (binary_notif) {
        binary_notif->addXferId(req->xfer_id);
      }
    }
    NIXL_DEBUG << "Striping: submitted " << submitted_count << " requests for "
               << transfer_size << " bytes";
  }

  NIXL_DEBUG << "Successfully submitted " << submitted_count << " requests for "
             << transfer_size << " bytes";

  return NIXL_SUCCESS;
}

std::vector<size_t> nixlLibfabricRailManager::selectRailsForMemory(
    void *mem_addr, nixl_mem_t mem_type, uint64_t dev_id) const {
  if (mem_type == VRAM_SEG) {
    // VRAM is physically reachable from every EFA NIC on the host (just at
    // varying PCIe-traversal cost depending on which root the NIC and GPU
    // share). Register on ALL rails so that a remote GPU endpoint (which
    // selects rails by its OWN GPU<->EFA topology) will always find a
    // valid rkey for the rail it chooses, even when the local and remote
    // GPU indices differ.
    //
    // This handles disaggregated-inference KV transfers where a prefill
    // replica on GPU N sends its KV to a decode replica on GPU M (N != M)
    // -- the proxy load-balancer pairs DP replicas independently of GPU
    // index. Without all-rail registration, prefill picks rail N (local)
    // but decode only registered rail M, leaving remote_keys[N]=0 and
    // failing the transfer with NIXL_ERR_INVALID_PARAM. Mirrors the DRAM
    // fix in PR #84921 (GEX-3676).
    //
    // Log the GPU-preferred rail subset for diagnostics, but always
    // return the full rail set.
    int gpu_id = static_cast<int>(dev_id);
    if (dev_id == std::numeric_limits<uint64_t>::max()) {
      NIXL_ERROR << "Missing device ID for VRAM memory " << mem_addr;
      return {};
    }
    if (gpu_id < 0) {
      NIXL_ERROR << "Invalid GPU ID " << gpu_id << " for VRAM memory "
                 << mem_addr;
      return {};
    }
    std::vector<std::string> gpu_efa_devices =
        topology->getEfaDevicesForGpu(gpu_id);
    if (!gpu_efa_devices.empty()) {
      std::string preferred_rails;
      for (const std::string &efa_device : gpu_efa_devices) {
        auto it = efa_device_to_rail_map.find(efa_device);
        if (it != efa_device_to_rail_map.end() &&
            it->second < data_rails_.size()) {
          preferred_rails += std::to_string(it->second) + " ";
        }
      }
      NIXL_DEBUG << "VRAM memory " << mem_addr << " on GPU " << gpu_id
                 << " GPU-preferred rails: [" << preferred_rails
                 << "] -- registering on all " << data_rails_.size()
                 << " rails for cross-GPU transfer compatibility";
    }

    std::vector<size_t> all_rails;
    all_rails.reserve(data_rails_.size());
    for (size_t i = 0; i < data_rails_.size(); ++i) {
      all_rails.push_back(i);
    }

    if (all_rails.empty()) {
      NIXL_ERROR << "No data rails available for VRAM memory " << mem_addr;
      return {};
    }

    NIXL_DEBUG << "VRAM memory " << mem_addr << " on GPU " << gpu_id
               << " will use all " << all_rails.size() << " rails";
    return all_rails;
  }
  if (mem_type == DRAM_SEG) {
    // DRAM is physically reachable from every EFA NIC on the host -- any
    // NIC can DMA into CPU memory regardless of NUMA affinity. Register on
    // ALL rails so that a remote GPU endpoint (which selects rails by
    // GPU<->EFA topology) will always find a valid rkey for the rail it
    // chooses. Restricting to NUMA-optimal rails causes a mismatch: the
    // GPU picks rail N, but the CPU only registered on NUMA-local rails
    // that may not include rail N, leaving remote_keys[N]=0 and making
    // every cross-device-type transfer fail with NIXL_ERR_INVALID_PARAM.
    //
    // Log the NUMA-preferred subset for diagnostics, but always return the
    // full rail set.
    int numa_node = topology->detectNumaNodeForMemory(mem_addr);
    if (numa_node >= 0) {
      std::vector<std::string> numa_efa_devices =
          topology->getEfaDevicesForNumaNode(numa_node);
      if (!numa_efa_devices.empty()) {
        std::string preferred_rails;
        for (const std::string &dev : numa_efa_devices) {
          auto it = efa_device_to_rail_map.find(dev);
          if (it != efa_device_to_rail_map.end()) {
            preferred_rails += std::to_string(it->second) + " ";
          }
        }
        NIXL_DEBUG << "DRAM memory " << mem_addr << " on NUMA node "
                   << numa_node << " NUMA-preferred rails: [" << preferred_rails
                   << "] -- registering on all " << data_rails_.size()
                   << " rails for cross-device-type transfer compatibility";
      }
    }

    std::vector<size_t> all_rails;
    all_rails.reserve(data_rails_.size());
    for (size_t i = 0; i < data_rails_.size(); ++i) {
      all_rails.push_back(i);
    }

    if (all_rails.empty()) {
      NIXL_ERROR << "No data rails available for DRAM memory " << mem_addr;
      return {};
    }

    NIXL_DEBUG << "DRAM memory " << mem_addr << " will use all "
               << all_rails.size() << " rails";
    return all_rails;
  }

  // For unsupported memory types, return empty vector
  NIXL_ERROR << "Unsupported memory type " << mem_type;
  return {};
}

nixl_status_t nixlLibfabricRailManager::registerMemory(
    void *buffer, size_t length, nixl_mem_t mem_type, uint64_t dev_id,
    std::vector<struct fid_mr *> &mr_list_out,
    std::vector<uint64_t> &key_list_out,
    std::vector<size_t> &selected_rails_out) {
  if (!buffer) {
    NIXL_ERROR << "Invalid buffer parameter";
    return NIXL_ERR_INVALID_PARAM;
  }

  // Use internal rail selection (moved from engine)
  std::vector<size_t> selected_rails =
      selectRailsForMemory(buffer, mem_type, dev_id);
  if (selected_rails.empty()) {
    NIXL_ERROR << "No rails selected for memory type " << mem_type;
    return NIXL_ERR_NOT_SUPPORTED;
  }

  // Resize output vectors to match all rails
  mr_list_out.resize(data_rails_.size(), nullptr);
  key_list_out.resize(data_rails_.size(), 0);
  selected_rails_out = selected_rails; // Return which rails were selected

  // Register memory on each selected rail
  for (size_t i = 0; i < selected_rails.size(); ++i) {
    size_t rail_idx = selected_rails[i];
    if (rail_idx >= data_rails_.size()) {
      NIXL_ERROR << "Invalid rail index " << rail_idx;
      // Cleanup already registered MRs
      for (size_t cleanup_idx : selected_rails) {
        if (cleanup_idx >= rail_idx)
          break; // Only cleanup what we've done so far
        if (mr_list_out[cleanup_idx]) {
          data_rails_[cleanup_idx]->deregisterMemory(mr_list_out[cleanup_idx]);
          mr_list_out[cleanup_idx] = nullptr;
        }
      }
      return NIXL_ERR_INVALID_PARAM;
    }

    struct fid_mr *mr;
    uint64_t key;
    nixl_status_t status = data_rails_[rail_idx]->registerMemory(
        buffer, length, mem_type, static_cast<int>(dev_id),
        FI_REMOTE_WRITE | FI_REMOTE_READ, &mr, &key);
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to register memory on rail " << rail_idx;
      // Cleanup already registered MRs
      for (size_t cleanup_idx : selected_rails) {
        if (cleanup_idx >= rail_idx)
          break; // Only cleanup what we've done so far
        if (mr_list_out[cleanup_idx]) {
          data_rails_[cleanup_idx]->deregisterMemory(mr_list_out[cleanup_idx]);
          mr_list_out[cleanup_idx] = nullptr;
        }
      }
      return status;
    }

    mr_list_out[rail_idx] = mr;
    key_list_out[rail_idx] = key;

    // Mark rail as active for progress tracking optimization
    markRailActive(rail_idx);

    NIXL_DEBUG << "Registered memory on rail " << rail_idx
               << " (mr: " << reinterpret_cast<uintptr_t>(mr)
               << ", key: " << key << ")";
  }

  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::deregisterMemory(
    const std::vector<size_t> &selected_rails,
    const std::vector<struct fid_mr *> &mr_list) {
  if (selected_rails.empty() || mr_list.size() != data_rails_.size()) {
    NIXL_ERROR << "Invalid parameters";
    return NIXL_ERR_INVALID_PARAM;
  }

  nixl_status_t overall_status = NIXL_SUCCESS;

  for (size_t i = 0; i < selected_rails.size(); ++i) {
    size_t rail_idx = selected_rails[i];
    if (rail_idx >= data_rails_.size()) {
      NIXL_ERROR << "Invalid rail index " << rail_idx;
      overall_status = NIXL_ERR_INVALID_PARAM;
      continue;
    }

    if (mr_list[rail_idx]) {
      nixl_status_t status =
          data_rails_[rail_idx]->deregisterMemory(mr_list[rail_idx]);
      if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to deregister memory on rail " << rail_idx;
        overall_status = status;
      }
      markRailInactive(rail_idx);
    }
  }

  return overall_status;
}

nixl_status_t nixlLibfabricRailManager::insertAllAddresses(
    RailType rail_type,
    const std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints,
    std::vector<fi_addr_t> &fi_addrs_out, std::vector<char *> &ep_names_out) {
  auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
  const char *rail_type_str =
      (rail_type == RailType::DATA) ? "data" : "control";

  if (endpoints.size() != rails.size()) {
    NIXL_ERROR << "Expected " << rails.size() << " " << rail_type_str
               << " endpoints, got " << endpoints.size();
    return NIXL_ERR_INVALID_PARAM;
  }

  fi_addrs_out.clear();
  ep_names_out.clear();
  fi_addrs_out.reserve(rails.size());
  ep_names_out.reserve(rails.size());

  // Process all rails in one operation
  for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
    fi_addr_t fi_addr;
    nixl_status_t status =
        rails[rail_id]->insertAddress(endpoints[rail_id].data(), &fi_addr);
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed for " << rail_type_str << " rail " << rail_id;
      return status;
    }

    fi_addrs_out.push_back(fi_addr);
    ep_names_out.push_back(
        rails[rail_id]->ep_name); // This is char[LF_EP_NAME_MAX_LEN], will be
                                  // converted to char*

    NIXL_DEBUG << "Processed " << rail_type_str << " rail " << rail_id
               << " (fi_addr: " << fi_addr << ")";
  }

  NIXL_DEBUG << "Successfully processed " << rails.size() << " "
             << rail_type_str << " rails";
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::cleanupConnection(
    RailType rail_type, const std::vector<fi_addr_t> &fi_addrs_to_remove) {
  auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
  const char *rail_type_str =
      (rail_type == RailType::DATA) ? "data" : "control";

  if (fi_addrs_to_remove.size() != rails.size()) {
    NIXL_ERROR << "Expected " << rails.size() << " " << rail_type_str
               << " fi_addrs, got " << fi_addrs_to_remove.size();
    return NIXL_ERR_INVALID_PARAM;
  }

  NIXL_DEBUG << "Cleaning up connection for " << rails.size() << " "
             << rail_type_str << " rails";
  // Remove addresses from all rails
  nixl_status_t overall_status = NIXL_SUCCESS;
  for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
    if (fi_addrs_to_remove[rail_id] != FI_ADDR_UNSPEC) {
      nixl_status_t status =
          rails[rail_id]->removeAddress(fi_addrs_to_remove[rail_id]);
      if (status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to remove address from " << rail_type_str
                   << " rail " << rail_id
                   << ", fi_addr: " << fi_addrs_to_remove[rail_id];
        overall_status = status;
        // Continue cleanup for other rails even if one fails
      } else {
        NIXL_DEBUG << "Successfully removed address from " << rail_type_str
                   << " rail " << rail_id
                   << ", fi_addr: " << fi_addrs_to_remove[rail_id];
      }
    } else {
      NIXL_DEBUG << "Skipping FI_ADDR_UNSPEC for " << rail_type_str << " rail "
                 << rail_id;
    }
  }
  NIXL_DEBUG << "Completed cleanup for " << rails.size() << " " << rail_type_str
             << " rails";
  return overall_status;
}

nixl_status_t nixlLibfabricRailManager::postControlMessage(
    ControlMessageType msg_type, nixlLibfabricReq *req, fi_addr_t dest_addr,
    uint16_t agent_idx, std::function<void()> completion_callback) {
  // Validation
  if (control_rails_.empty()) {
    NIXL_ERROR << "No control rails available";
    return NIXL_ERR_INVALID_PARAM;
  }

  if (!req) {
    NIXL_ERROR << "Pre-allocated request is null";
    return NIXL_ERR_INVALID_PARAM;
  }

  auto getMessageTypeValue = [](ControlMessageType message_type) {
    switch (message_type) {
    case ControlMessageType::NOTIFICATION:
      return uint64_t{NIXL_LIBFABRIC_MSG_NOTIFICATION};
    case ControlMessageType::CONNECTION_REQ:
      return uint64_t{NIXL_LIBFABRIC_MSG_CONNECT};
    case ControlMessageType::CONNECTION_ACK:
      return uint64_t{NIXL_LIBFABRIC_MSG_ACK};
    case ControlMessageType::DISCONNECT_REQ:
      return uint64_t{NIXL_LIBFABRIC_MSG_DISCONNECT};
    }

    llvm_unreachable("invalid control message type");
  };
  uint64_t msg_type_value = getMessageTypeValue(msg_type);
  size_t control_rail_id = 0;
  uint32_t xfer_id = req->xfer_id;
  uint64_t imm_data = NIXL_MAKE_IMM_DATA(msg_type_value, agent_idx, xfer_id);

  // Set completion callback if provided
  if (completion_callback) {
    req->completion_callback = completion_callback;
    NIXL_DEBUG << "Set completion callback for control message request "
               << req->xfer_id;
  }

  NIXL_DEBUG << "Sending control message type " << msg_type_value
             << " agent_idx=" << agent_idx << " XFER_ID=" << xfer_id
             << " imm_data=0x" << std::hex << imm_data << std::dec;

  // Rail postSend
  nixl_status_t status =
      control_rails_[control_rail_id]->postSend(imm_data, dest_addr, req);

  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to send control message type "
               << static_cast<int>(msg_type) << " on control rail "
               << control_rail_id;
    control_rails_[control_rail_id]->releaseRequest(req);
    return status;
  }
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::progressActiveDataRails() {
  std::vector<size_t> active_rails = snapshotActiveRails();

  if (active_rails.empty()) {
    return NIXL_IN_PROG; // No active rails to process
  }

  bool any_completions = false;

  for (size_t rail_id : active_rails) {
    if (rail_id >= data_rails_.size()) {
      NIXL_ERROR << "Invalid active rail ID: " << rail_id;
      continue;
    }
    // Process completions on active data rails
    nixl_status_t status = data_rails_[rail_id]->progressCompletionQueue(false);
    if (status == NIXL_SUCCESS) {
      any_completions = true;
      NIXL_DEBUG << "Processed completions on active data rail " << rail_id;
    } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to process completions on active data rail "
                 << rail_id;
      // Continue processing other active rails even if one fails
    }
  }

  if (any_completions) {
    NIXL_TRACE << "Processed " << active_rails.size()
               << " active rails, completions found";
  }

  return any_completions ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t nixlLibfabricRailManager::progressAllControlRails() {
  bool any_completions = false;
  for (size_t rail_id = 0; rail_id < num_control_rails_; ++rail_id) {
    nixl_status_t status =
        control_rails_[rail_id]->progressCompletionQueue(true);
    if (status == NIXL_SUCCESS) {
      any_completions = true;
      NIXL_DEBUG << "Processed completion on control rail " << rail_id;
    } else if (status != NIXL_IN_PROG && status != NIXL_SUCCESS) {
      NIXL_WARN << "Transient error on control rail " << rail_id
                << ", retrying";
    }
  }
  return any_completions ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t nixlLibfabricRailManager::validateAllRailsInitialized() {
  for (size_t rail_id = 0; rail_id < data_rails_.size(); ++rail_id) {
    if (!data_rails_[rail_id]->isProperlyInitialized()) {
      NIXL_ERROR << "Rail " << rail_id << " is not properly initialized";
      return NIXL_ERR_BACKEND;
    }
  }
  NIXL_DEBUG << "All " << data_rails_.size()
             << " rails are properly initialized";
  return NIXL_SUCCESS;
}

struct fid_mr *
nixlLibfabricRailManager::getMemoryDescriptor(size_t rail_id,
                                              struct fid_mr *mr) {
  if (rail_id >= data_rails_.size()) {
    NIXL_ERROR << "Invalid rail index " << rail_id;
    return nullptr;
  }
  return static_cast<struct fid_mr *>(
      data_rails_[rail_id]->getMemoryDescriptor(mr));
}

nixl_status_t nixlLibfabricRailManager::serializeMemoryKeys(
    const std::vector<uint64_t> &keys, void *buffer, std::string &str) const {
  nixlSerDes ser_des;
  // Serialize all rail keys instead of just the first one
  for (size_t rail_id = 0; rail_id < keys.size(); ++rail_id) {
    std::string key_name = "key_" + std::to_string(rail_id);
    ser_des.addBuf(key_name.c_str(), &keys[rail_id], sizeof(keys[rail_id]));
  }

  ser_des.addBuf("addr", &buffer, sizeof(buffer));
  str = ser_des.exportStr();

  NIXL_DEBUG << "Serialized memory keys for " << keys.size() << " rails"
             << " (buffer: " << buffer << ", size: " << str.length()
             << " bytes)";
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::deserializeMemoryKeys(
    const std::string &serialized_data, std::vector<uint64_t> &keys_out,
    uint64_t &remote_addr_out) const {
  nixlSerDes ser_des;
  if (ser_des.importBlob(blob_ref{serialized_data}) != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to deserialize remote metadata";
    return NIXL_ERR_BACKEND;
  }
  // Load all rail keys instead of just one
  keys_out.clear();
  keys_out.reserve(data_rails_.size());
  for (size_t rail_id = 0; rail_id < data_rails_.size(); ++rail_id) {
    std::string key_name = "key_" + std::to_string(rail_id);
    uint64_t remote_key;
    nixl_status_t status =
        ser_des.getBuf(key_name.c_str(), &remote_key, sizeof(remote_key));
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to get key " << key_name << " for rail " << rail_id;
      return NIXL_ERR_BACKEND;
    }
    keys_out.push_back(remote_key);
  }
  nixl_status_t addr_status =
      ser_des.getBuf("addr", &remote_addr_out, sizeof(remote_addr_out));
  if (addr_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to get remote address";
    return NIXL_ERR_BACKEND;
  }
  NIXL_DEBUG << "Deserialized memory keys for " << keys_out.size() << " rails"
             << " (remote addr: " << (void *)remote_addr_out << ")";
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::serializeConnectionInfo(
    const std::string &user_prefix, std::string &str) const {

  nixlSerDes ser_des;

  // Use user prefix with standard suffixes
  std::string data_prefix = user_prefix + "_data_ep_";
  std::string control_prefix = user_prefix + "_control_ep_";

  serializeRailEndpoints(ser_des, data_prefix, RailType::DATA);
  serializeRailEndpoints(ser_des, control_prefix, RailType::CONTROL);
  str = ser_des.exportStr();
  NIXL_DEBUG << "Connection info serialized with prefix " << user_prefix
             << ", size: " << str.length();
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRailManager::deserializeConnectionInfo(
    const std::string &user_prefix, const std::string &serialized_data,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &data_endpoints_out,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &control_endpoints_out)
    const {

  nixlSerDes ser_des;
  if (ser_des.importBlob(blob_ref{serialized_data}) != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to deserialize connection info";
    return NIXL_ERR_BACKEND;
  }

  // Use user prefix with standard suffixes
  std::string data_prefix = user_prefix + "_data_ep_";
  std::string control_prefix = user_prefix + "_control_ep_";
  nixl_status_t data_status = deserializeRailEndpoints(
      ser_des, data_prefix, data_rails_.size(), data_endpoints_out);
  if (data_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to deserialize data rail endpoints with prefix: "
               << data_prefix;
    return data_status;
  }
  nixl_status_t control_status = deserializeRailEndpoints(
      ser_des, control_prefix, control_rails_.size(), control_endpoints_out);
  if (control_status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to deserialize control rail endpoints with prefix: "
               << control_prefix;
    return control_status;
  }

  NIXL_DEBUG << "Connection info deserialized with prefix " << user_prefix
             << ": " << data_endpoints_out.size() << " data endpoints, "
             << control_endpoints_out.size() << " control endpoints";
  return NIXL_SUCCESS;
}

void nixlLibfabricRailManager::serializeRailEndpoints(
    nixlSerDes &ser_des, const std::string &key_prefix,
    RailType rail_type) const {
  auto &rails = (rail_type == RailType::DATA) ? data_rails_ : control_rails_;
  const char *rail_type_str =
      (rail_type == RailType::DATA) ? "data" : "control";

  for (size_t rail_id = 0; rail_id < rails.size(); ++rail_id) {
    std::string rail_key = key_prefix + std::to_string(rail_id);
    const char *ep_name = rails[rail_id]->ep_name;
    size_t ep_name_len = sizeof(rails[rail_id]->ep_name);

    ser_des.addBuf(rail_key.c_str(), ep_name, ep_name_len);
  }

  NIXL_DEBUG << "Serialized " << rails.size() << " " << rail_type_str
             << " rail endpoints";
}

nixl_status_t nixlLibfabricRailManager::deserializeRailEndpoints(
    nixlSerDes &ser_des, const std::string &key_prefix, size_t expected_count,
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> &endpoints_out) const {
  endpoints_out.resize(expected_count);

  for (size_t rail_id = 0; rail_id < expected_count; ++rail_id) {
    std::string rail_key = key_prefix + std::to_string(rail_id);

    // First check if the key exists and get its length
    ssize_t actual_len = ser_des.getBufLen(rail_key);
    if (actual_len <= 0) {
      NIXL_ERROR << "Key " << rail_key
                 << " not found or has invalid length: " << actual_len;
      return NIXL_ERR_BACKEND;
    }

    if (actual_len > (ssize_t)endpoints_out[rail_id].size()) {
      NIXL_ERROR << "Buffer too small for rail " << rail_id << ", need "
                 << actual_len << " bytes, have "
                 << endpoints_out[rail_id].size();
      return NIXL_ERR_BACKEND;
    }

    // Get the actual data
    nixl_status_t status =
        ser_des.getBuf(rail_key, endpoints_out[rail_id].data(), actual_len);
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to get endpoint address for rail " << rail_id
                 << " with key: " << rail_key << ", status: " << status;
      return NIXL_ERR_BACKEND;
    }
  }

  NIXL_DEBUG << "Successfully deserialized " << expected_count
             << " rail endpoints";
  return NIXL_SUCCESS;
}

void nixlLibfabricRailManager::markRailActive(size_t rail_id) {
  if (rail_id >= data_rails_.size()) {
    NIXL_ERROR << "Invalid rail ID for markRailActive: " << rail_id;
    return;
  }

  bool was_inserted = false;
  size_t active_count = 0;
  {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    was_inserted = active_rails_.insert(rail_id).second;
    active_count = active_rails_.size();
  }

  if (was_inserted) {
    NIXL_DEBUG << "Marked rail " << rail_id
               << " as active (total active: " << active_count << ")";
  } else {
    NIXL_TRACE << "Rail " << rail_id << " was already active";
  }
}

void nixlLibfabricRailManager::markRailInactive(size_t rail_id) {
  size_t erased = 0;
  size_t active_count = 0;
  {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    erased = active_rails_.erase(rail_id);
    active_count = active_rails_.size();
  }
  if (erased > 0) {
    NIXL_DEBUG << "Marked rail " << rail_id
               << " as inactive (total active: " << active_count << ")";
  } else {
    NIXL_TRACE << "Rail " << rail_id << " was not in active set";
  }
}

void nixlLibfabricRailManager::clearActiveRails() {
  size_t cleared_count = 0;
  {
    std::lock_guard<std::mutex> lock(active_rails_mutex_);
    cleared_count = active_rails_.size();
    active_rails_.clear();
  }
  NIXL_DEBUG << "Cleared " << cleared_count << " active rails";
}

size_t nixlLibfabricRailManager::getActiveRailCount() const {
  std::lock_guard<std::mutex> lock(active_rails_mutex_);
  return active_rails_.size();
}

std::vector<size_t> nixlLibfabricRailManager::snapshotActiveRails() const {
  std::lock_guard<std::mutex> lock(active_rails_mutex_);
  return {active_rails_.begin(), active_rails_.end()};
}

} // namespace M::MLRT
