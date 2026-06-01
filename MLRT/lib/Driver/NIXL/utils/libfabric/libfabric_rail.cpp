//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_rail.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <sys/uio.h>

// RequestPool Base Class Implementation

namespace M::MLRT {
using enum nixl_mem_t;
using enum nixl_status_t;
using enum nixl_xfer_op_t;

RequestPool::RequestPool(size_t pool_size, size_t rail_id) : rail_id_(rail_id) {
  requests_.resize(pool_size);

  for (size_t i = 0; i < pool_size; ++i) {
    requests_[i].rail_id = rail_id;
    requests_[i].in_use = false;
    free_indices_.push(i);
  }
}

void RequestPool::release(nixlLibfabricReq *req) {
  if (!req)
    return;

  std::lock_guard<std::mutex> lock(pool_mutex_);

  req->in_use = false;
  req->chunk_offset = 0;
  req->chunk_size = 0;
  req->completion_callback = nullptr;
  memset(&req->ctx, 0, sizeof(fi_context2));
  size_t idx = req - &requests_[0];
  free_indices_.push(idx);
}

nixlLibfabricReq *RequestPool::findByContext(void *context) const {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  return reinterpret_cast<nixlLibfabricReq *>(context);
}

size_t RequestPool::getActiveRequestCount() const {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  return requests_.size() - free_indices_.size();
}

// ControlRequestPool Implementation

ControlRequestPool::ControlRequestPool(size_t pool_size, size_t rail_id)
    : RequestPool(pool_size, rail_id), buffer_chunk_(nullptr),
      buffer_chunk_size_(0), buffer_mr_(nullptr) {}

ControlRequestPool::~ControlRequestPool() {
  // Cleanup should have been called explicitly before domain destruction
  // This is just a safety check
  cleanup();
}

void ControlRequestPool::cleanup() {
  if (buffer_mr_) {
    fi_close(&buffer_mr_->fid);
    buffer_mr_ = nullptr;
  }
  if (buffer_chunk_) {
    free(buffer_chunk_);
    buffer_chunk_ = nullptr;
  }
}

nixl_status_t ControlRequestPool::initializeWithBuffersAndXferIds(
    struct fid_domain *domain, const std::vector<uint32_t> &xfer_ids) {
  if (xfer_ids.size() != requests_.size()) {
    return NIXL_ERR_INVALID_PARAM;
  }

  // Allocate buffer chunk
  buffer_chunk_size_ = BUFFER_SIZE * requests_.size();

  buffer_chunk_ = malloc(buffer_chunk_size_);
  if (!buffer_chunk_) {
    NIXL_ERROR << "Standard allocation failed for control request pool on rail "
               << rail_id_;
    return NIXL_ERR_BACKEND;
  }

  NIXL_DEBUG << "Allocated " << buffer_chunk_size_
             << " bytes for control request pool on rail " << rail_id_;

  // Register buffer chunk with libfabric
  int ret = fi_mr_reg(domain, buffer_chunk_, buffer_chunk_size_,
                      FI_SEND | FI_RECV, 0, 0, 0, &buffer_mr_, NULL);
  if (ret) {
    free(buffer_chunk_);
    buffer_chunk_ = nullptr;
    return NIXL_ERR_BACKEND;
  }
  // Pre-assign buffers and XFER_IDs to requests
  for (size_t i = 0; i < requests_.size(); ++i) {
    requests_[i].xfer_id = xfer_ids[i];
    requests_[i].buffer =
        static_cast<char *>(buffer_chunk_) + (i * BUFFER_SIZE);
    requests_[i].mr = buffer_mr_;
    requests_[i].buffer_size = BUFFER_SIZE;
    requests_[i].operation_type = nixlLibfabricReq::SEND; // Default for control
  }
  return NIXL_SUCCESS;
}

nixlLibfabricReq *ControlRequestPool::allocate(size_t needed_size) {
  std::lock_guard<std::mutex> lock(pool_mutex_);

  if (free_indices_.empty()) {
    return nullptr; // No free requests
  }
  if (needed_size > BUFFER_SIZE) {
    return nullptr; // Size too large
  }

  size_t idx = free_indices_.top();
  free_indices_.pop();

  nixlLibfabricReq *req = &requests_[idx];
  req->in_use = true;

  return req;
}

// DataRequestPool Implementation

DataRequestPool::DataRequestPool(size_t pool_size, size_t rail_id)
    : RequestPool(pool_size, rail_id) {}

nixl_status_t
DataRequestPool::initializeWithXferIds(const std::vector<uint32_t> &xfer_ids) {
  if (xfer_ids.size() != requests_.size()) {
    return NIXL_ERR_INVALID_PARAM;
  }
  // Pre-assign XFER_IDs to requests
  for (size_t i = 0; i < requests_.size(); ++i) {
    requests_[i].xfer_id = xfer_ids[i];
    requests_[i].buffer = nullptr; // No buffers for data requests
    requests_[i].mr = nullptr;
    requests_[i].buffer_size = 0;
    requests_[i].operation_type = nixlLibfabricReq::WRITE; // Default for data
  }
  return NIXL_SUCCESS;
}

nixlLibfabricReq *DataRequestPool::allocate(nixlLibfabricReq::OpType op_type) {
  std::lock_guard<std::mutex> lock(pool_mutex_);

  if (free_indices_.empty()) {
    return nullptr;
  }

  size_t idx = free_indices_.top();
  free_indices_.pop();

  nixlLibfabricReq *req = &requests_[idx];
  req->in_use = true;
  req->operation_type = op_type;

  return req;
}

// Rail Class Implementation

M::ErrorOr<std::unique_ptr<nixlLibfabricRail>>
nixlLibfabricRail::create(const std::string &device, uint16_t id,
                          bool enable_hmem) {
  auto rail = std::unique_ptr<nixlLibfabricRail>(
      new nixlLibfabricRail(device, id, enable_hmem));
  if (rail->getInitStatus() != NIXL_SUCCESS) {
    return M::Error(rail->getInitError().empty()
                        ? "Failed to initialize libfabric rail."
                        : rail->getInitError());
  }
  return rail;
}

nixlLibfabricRail::nixlLibfabricRail(const std::string &device, uint16_t id,
                                     bool enable_hmem)
    : rail_id(id), device_name(device), blocking_cq_sread_supported(true),
      control_request_pool_(CONTROL_REQUESTS_PER_RAIL, id),
      data_request_pool_(DATA_REQUESTS_PER_RAIL, id) {
  // Initialize all pointers to nullptr
  info = nullptr;
  fabric = nullptr;
  domain = nullptr;
  endpoint = nullptr;
  cq = nullptr;
  av = nullptr;
  memset(ep_name, 0, sizeof(ep_name));

  // Initialize all Libfabric resources for this rail
  NIXL_TRACE << "Initializing rail " << rail_id
             << " with device: " << device_name;

  auto failInit = [&](std::string message) {
    init_status_ = NIXL_ERR_BACKEND;
    init_error_ = std::move(message);
    NIXL_ERROR << init_error_;
    cleanup();
  };

  // Initialize hints for this rail
  struct fi_info *hints = fi_allocinfo();
  if (!hints) {
    failInit("Failed to allocate fi_info for rail " + std::to_string(rail_id) +
             ".");
    return;
  }
  hints->caps = FI_MSG | FI_RMA;
  if (enable_hmem)
    hints->caps |= FI_HMEM;
  hints->caps |= FI_LOCAL_COMM | FI_REMOTE_COMM;
  hints->mode = FI_CONTEXT | FI_CONTEXT2;
  hints->ep_attr->type = FI_EP_RDM;
  hints->domain_attr->mr_mode =
      FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
  if (enable_hmem)
    hints->domain_attr->mr_mode |= FI_MR_HMEM;
  hints->domain_attr->mr_key_size = 2;
  hints->domain_attr->name = strdup(device_name.c_str());
  hints->domain_attr->threading = FI_THREAD_SAFE;
  // Request the "efa" fabric (not "efa-direct"). The efa-direct fabric
  // bypasses the SRD middleware and cannot handle large RDMA transfers.
  hints->fabric_attr->name = strdup("efa");
  // Get fabric info for this specific device
  int ret = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &info);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_getinfo failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  // Create fabric for this rail
  ret = fi_fabric(info->fabric_attr, &fabric, NULL);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_fabric failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }
  NIXL_TRACE << "fabric_attr->name " << info->fabric_attr->name;

  // Create domain for this rail
  ret = fi_domain(fabric, info, &domain, NULL);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_domain failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  // Create CQ for this rail
  struct fi_cq_attr cq_attr = {};
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.wait_obj = FI_WAIT_UNSPEC;
  cq_attr.size = 12288;
  ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
  if (ret) {
    NIXL_ERROR << "fi_cq_open failed for rail " << rail_id << ": "
               << fi_strerror(-ret) << " - trying FI_WAIT_NONE";
    if (ret == -FI_ENOSYS) {
      NIXL_TRACE << "FI_WAIT_UNSPEC not supported, falling back to "
                    "FI_WAIT_NONE for rail "
                 << rail_id;
      blocking_cq_sread_supported = false;
      cq_attr.wait_obj = FI_WAIT_NONE;
      ret = fi_cq_open(domain, &cq_attr, &cq, NULL);
      if (ret) {
        fi_freeinfo(hints);
        failInit("fi_cq_open with FI_WAIT_NONE failed for rail " +
                 std::to_string(rail_id) + ": " + fi_strerror(-ret) + ".");
        return;
      }
      NIXL_TRACE << "fi_cq_open with FI_WAIT_NONE succeeded for rail "
                 << rail_id;
    } else {
      fi_freeinfo(hints);
      failInit("fi_cq_open failed for rail " + std::to_string(rail_id) + ": " +
               fi_strerror(-ret) + ".");
      return;
    }
  }
  if (!cq) {
    fi_freeinfo(hints);
    failInit("CQ creation returned success but pointer is null for rail " +
             std::to_string(rail_id) + ".");
    return;
  }

  struct fi_av_attr av_attr = {};
  av_attr.type = FI_AV_TABLE;
  av_attr.count = 1024;
  ret = fi_av_open(domain, &av_attr, &av, NULL);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_av_open failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  ret = fi_endpoint(domain, info, &endpoint, NULL);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_endpoint failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  ret = fi_ep_bind(endpoint, &cq->fid, FI_TRANSMIT | FI_RECV);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_ep_bind cq failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  ret = fi_ep_bind(endpoint, &av->fid, 0);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_ep_bind av failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  // GEX-3606 (AWS suggestion): set FI_OPT_EFA_HOMOGENEOUS_PEERS=true to
  // bypass the EFA RDM provider's per-peer handshake that runs before the
  // first RMA write to that peer. Per fi_efa(7): "When set to true, it
  // indicates all peers are homogeneous, meaning they run on the same
  // platform, use the same software versions, and share identical
  // capabilities. It accelerates the initial communication setup as
  // interoperability between peers is guaranteed." Both peers in DI run
  // identical MAX/MLRT binaries against identical libfabric, so this is
  // always safe to enable. Must be called before fi_enable().
  //
  // Without this, the original GEX-3606 symptom returns: rail 0's first
  // RMA write to a never-before-seen peer can hit a silent EAGAIN spin
  // (the rail's CQ stops producing TX completions while sibling rails
  // drain normally), wedging the workload at WARN log level. Enabling
  // it bypasses the handshake and lets the data write proceed directly.
  {
    bool homogeneous = true;
    int hp_ret =
        fi_setopt(&endpoint->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_HOMOGENEOUS_PEERS,
                  &homogeneous, sizeof(homogeneous));
    if (hp_ret) {
      NIXL_WARN << "fi_setopt(FI_OPT_EFA_HOMOGENEOUS_PEERS=true) failed on "
                   "rail "
                << rail_id << ": " << fi_strerror(-hp_ret);
    } else {
      NIXL_DEBUG << "rail " << rail_id
                 << ": fi_setopt(FI_OPT_EFA_HOMOGENEOUS_PEERS=true) ok";
    }
  }

  ret = fi_enable(endpoint);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_enable failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  size_t ep_name_len = sizeof(ep_name);
  ret = fi_getname(&endpoint->fid, ep_name, &ep_name_len);
  if (ret) {
    fi_freeinfo(hints);
    failInit("fi_getname failed for rail " + std::to_string(rail_id) + ": " +
             fi_strerror(-ret) + ".");
    return;
  }

  std::vector<uint32_t> control_xfer_ids =
      LibfabricUtils::preallocateXferIds(CONTROL_REQUESTS_PER_RAIL);
  std::vector<uint32_t> data_xfer_ids =
      LibfabricUtils::preallocateXferIds(DATA_REQUESTS_PER_RAIL);

  nixl_status_t status = control_request_pool_.initializeWithBuffersAndXferIds(
      domain, control_xfer_ids);
  if (status != NIXL_SUCCESS) {
    fi_freeinfo(hints);
    failInit("Failed to initialize control request pool for rail " +
             std::to_string(rail_id) + ".");
    return;
  }

  status = data_request_pool_.initializeWithXferIds(data_xfer_ids);
  if (status != NIXL_SUCCESS) {
    fi_freeinfo(hints);
    failInit("Failed to initialize data request pool for rail " +
             std::to_string(rail_id) + ".");
    return;
  }

  NIXL_TRACE << "Initialized request pools: " << CONTROL_REQUESTS_PER_RAIL
             << " control requests, " << DATA_REQUESTS_PER_RAIL
             << " data requests for rail " << rail_id;

  nixlLibfabricReq *recv_req =
      allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);
  if (!recv_req) {
    fi_freeinfo(hints);
    failInit("Failed to allocate request for initial receive on rail " +
             std::to_string(rail_id) + ".");
    return;
  }

  status = postRecv(recv_req);
  if (status != NIXL_SUCCESS) {
    releaseRequest(recv_req);
    fi_freeinfo(hints);
    failInit("Failed to post initial receive on rail " +
             std::to_string(rail_id) + ".");
    return;
  }

  NIXL_TRACE << "Successfully initialized rail " << rail_id;
  fi_freeinfo(hints);
}

nixlLibfabricRail::~nixlLibfabricRail() { cleanup(); }

bool nixlLibfabricRail::isProperlyInitialized() const {
  return (cq != nullptr && endpoint != nullptr && domain != nullptr);
}

void nixlLibfabricRail::cleanup() {
  NIXL_TRACE << "Starting cleanup for rail " << rail_id;
  // STEP 1: Close endpoint first (it depends on CQ and AV)
  if (endpoint) {
    NIXL_TRACE << "Closing endpoint for rail " << rail_id;
    int ret = fi_close(&endpoint->fid);
    if (ret) {
      NIXL_WARN << "fi_close endpoint failed for rail " << rail_id << ": "
                << fi_strerror(-ret);
    }
    endpoint = nullptr;
  }
  // STEP 2: Close CQ and AV (they depend on domain)
  if (cq) {
    NIXL_TRACE << "Closing completion queue for rail " << rail_id;
    int ret = fi_close(&cq->fid);
    if (ret) {
      NIXL_WARN << "fi_close cq failed for rail " << rail_id << ": "
                << fi_strerror(-ret);
    }
    cq = nullptr;
  }
  if (av) {
    NIXL_TRACE << "Closing address vector for rail " << rail_id;
    int ret = fi_close(&av->fid);
    if (ret) {
      NIXL_WARN << "fi_close av failed for rail " << rail_id << ": "
                << fi_strerror(-ret);
    }
    av = nullptr;
  }

  // STEP 3: Clean up request pools while domain is still valid
  // This ensures all memory registrations (MRs) are properly deregistered
  // before domain closure
  NIXL_TRACE << "Cleaning up request pools for rail " << rail_id;
  control_request_pool_.cleanup();
  // STEP 4: Close domain AFTER all MRs, endpoint, CQ, AV are closed
  if (domain) {
    NIXL_TRACE << "Closing domain for rail " << rail_id;
    int ret = fi_close(&domain->fid);
    if (ret) {
      NIXL_WARN << "fi_close domain failed for rail " << rail_id << ": "
                << fi_strerror(-ret);
    }
    domain = nullptr;
  }
  // STEP 5: Close fabric
  if (fabric) {
    NIXL_TRACE << "Closing fabric for rail " << rail_id;
    int ret = fi_close(&fabric->fid);
    if (ret) {
      NIXL_WARN << "fi_close fabric failed for rail " << rail_id << ": "
                << fi_strerror(-ret);
    }
    fabric = nullptr;
  }
  // STEP 6: Free info structure
  if (info) {
    NIXL_INFO << "Freeing info structure for rail " << rail_id;
    fi_freeinfo(info);
    info = nullptr;
  }
  NIXL_TRACE << "Cleanup completed for rail " << rail_id;
}

void nixlLibfabricRail::setNotificationCallback(
    std::function<void(const std::string &)> callback) {
  notificationCallback = callback;
}

void nixlLibfabricRail::setConnectionAckCallback(
    std::function<void(uint16_t, nixlLibfabricConnection *, ConnectionState)>
        callback) {
  connectionAckCallback = callback;
}

void nixlLibfabricRail::setConnectionReqCallback(
    std::function<nixl_status_t(uint16_t, const std::string &,
                                nixlLibfabricRail *)>
        callback) {
  connectionReqCallback = callback;
}

void nixlLibfabricRail::setXferIdCallback(
    std::function<void(uint32_t)> callback) {
  xferIdCallback = callback;
}

// Per-Rail Completion Processing

// Per-rail completion processing - handles one rail's CQ with configurable
// blocking behavior
nixl_status_t nixlLibfabricRail::progressCompletionQueue(bool use_blocking) {
  // Completion processing
  struct fi_cq_data_entry completion;
  memset(&completion, 0, sizeof(completion));

  int ret;

  // Only protect libfabric CQ hardware operations
  {
    std::lock_guard<std::mutex> cq_lock(cq_progress_mutex_);

    if (use_blocking && blocking_cq_sread_supported) {
      // Blocking read using fi_cq_sread (used by CM thread)
      ret = fi_cq_sread(cq, &completion, 1, nullptr,
                        NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_SEC);
    } else {
      // Non-blocking read (used by progress thread or fallback)
      ret = fi_cq_read(cq, &completion, 1);
    }

    // -FI_EINTR is raised when fi_cq_sread is interrupted by a signal
    // (commonly seen with OpenMPI's signal-based bookkeeping on AWS EFA).
    // It is not a CQ error — fi_cq_readerr returns -FI_EAGAIN with nothing
    // in the error queue — and the correct POSIX response is to retry on
    // the next iteration, exactly like -FI_EAGAIN. Treating it as an error
    // caused the CM thread to bail out and the surrounding poll loops to
    // spin without progress, wedging every chat completion (1600% CPU,
    // 0/300 requests, observed on production p6-b200.48xlarge during
    // GEX-3606 investigation).
    if (ret < 0 && ret != -FI_EAGAIN && ret != -FI_EINTR) {
      NIXL_ERROR << "fi_cq_read returned error " << ret << " on rail "
                 << rail_id << ": " << fi_strerror(-ret);

      // Handle error - but be careful about fi_cq_readerr
      struct fi_cq_err_entry err_entry;
      memset(&err_entry, 0, sizeof(err_entry));

      int err_ret = fi_cq_readerr(cq, &err_entry, 0);
      if (err_ret > 0) {
        NIXL_ERROR << "CQ read failed on rail " << rail_id
                   << " with error: " << fi_strerror(err_entry.err)
                   << " prov_errno: " << err_entry.prov_errno
                   << " len: " << err_entry.len;

        // GEX-3606: a per-request CQ error (e.g. EFA prov_errno=7
        // REMOTE_ERROR_BAD_ADDRESS from a briefly mis-registered receiver
        // MR, observed on production p6-b200.48xlarge) was previously
        // logged and dropped here without finding the failed request.
        // Each event leaked one slot from the data or control request
        // pool. After ~1024 leaks per data rail or ~256 leaks on the
        // control rail, postXfer could no longer allocate new requests
        // and ALL subsequent transfers failed with "Failed to allocate
        // {control,data} request" — turning a transient receiver hiccup
        // into a fatal pool-exhaustion that wedged the workload (a
        // production prefill log showed 2,133 prov_errno=7 events
        // exhausting rail 0's data pool, then leaking the control pool
        // via postXfer's error-return paths, leaving only 17/300
        // requests successful).
        //
        // Look up the failed request via err_entry.op_context, invoke
        // its completion callback so the upper-layer transfer handle
        // advances instead of hanging until the app-level timeout, then
        // release it back to its pool. The data never reached the
        // destination, so the receiver will see whatever was at
        // remote_addr — that's a pre-existing failure mode of the failed
        // write itself; what we eliminate here is the cascade that took
        // every other in-flight request down with it.
        nixlLibfabricReq *failed_req =
            findRequestFromContext(err_entry.op_context);
        if (failed_req) {
          if (failed_req->completion_callback) {
            failed_req->completion_callback();
          }
          releaseRequest(failed_req);
        } else {
          NIXL_WARN << "GEX-3606: no request found for failed completion "
                       "context "
                    << err_entry.op_context << " on rail " << rail_id
                    << " (cannot release; pool slot leaked)";
        }
        // Return SUCCESS so callers (postWrite's EAGAIN retry loop via
        // progressAndLogForEagainRetry, postSend's EAGAIN retry loop,
        // progressActiveDataRails) do not interpret a single bad write
        // as a rail-level failure. The rail itself is healthy; only this
        // one transfer (now released and accounted for) failed. Without
        // this, every prov_errno=7 would also poison whatever postWrite
        // happened to be progressing the CQ at the time, multiplying
        // one transient EFA error into many independent transfer
        // failures (b200-aws workspace run: 7,147 prov_errno=7 events
        // → only 15/300 failures with this path, vs 266/300 without).
        return NIXL_SUCCESS;
      }
      NIXL_ERROR << "fi_cq_readerr failed with " << err_ret;
      // err_ret <= 0 here means we could not even drain the error entry
      // from the CQ; that's a genuine CQ-machinery failure, propagate it.
      return NIXL_ERR_BACKEND;
    }
  }
  // CQ lock released here - completion is now local data

  if (ret == -FI_EAGAIN || ret == -FI_EINTR) {
    return NIXL_IN_PROG; // No completions available
  }

  if (ret == 1) {
    NIXL_TRACE << "Completion received on rail " << rail_id
               << " flags: " << std::hex << completion.flags
               << " data: " << completion.data
               << " context: " << completion.op_context << std::dec;

    // Process completion using local data. Callbacks have their own thread
    // safety
    nixl_status_t status = processCompletionQueueEntry(&completion);
    if (status != NIXL_SUCCESS) {
      NIXL_ERROR << "Failed to process completion on rail " << rail_id;
      return status;
    }

    NIXL_DEBUG << "Completion processed on rail " << rail_id;
    return NIXL_SUCCESS;
  }

  return NIXL_ERR_BACKEND; // Unexpected case
}

// Route completion to appropriate handler (rail-specific)
nixl_status_t
nixlLibfabricRail::processCompletionQueueEntry(struct fi_cq_data_entry *comp) {
  uint64_t flags = comp->flags;

  NIXL_TRACE << "Routing completion from rail " << rail_id
             << " with flags: " << std::hex << flags
             << " FI_SEND: " << (flags & FI_SEND)
             << " FI_RECV: " << (flags & FI_RECV)
             << " FI_WRITE: " << (flags & FI_WRITE)
             << " FI_REMOTE_WRITE: " << (flags & FI_REMOTE_WRITE) << std::dec;

  if (flags & FI_SEND) {
    // Local send completions (fi_senddata) - use context
    return processLocalSendCompletion(comp);

  } else if (flags & FI_RECV) {
    // Receive completions - use immediate data
    return processRecvCompletion(comp);

  } else if (flags & FI_WRITE) {
    // Local write completions (fi_writedata) - use context
    return processLocalTransferCompletion(comp, "write");

  } else if (flags & FI_READ) {
    // Local read completions (fi_readdata) - use context
    return processLocalTransferCompletion(comp, "read");

  } else if (flags & FI_REMOTE_WRITE) {
    // Remote write completions (from fi_writedata) - use immediate data
    return processRemoteWriteCompletion(comp);

  } else {
    // Add more detailed warning for unknown completion flags
    NIXL_WARN << "Unknown completion flags detected on rail " << rail_id
              << " - flags: 0x" << std::hex << flags << std::dec
              << " (FI_SEND=" << !!(flags & FI_SEND)
              << " FI_RECV=" << !!(flags & FI_RECV)
              << " FI_WRITE=" << !!(flags & FI_WRITE)
              << " FI_READ=" << !!(flags & FI_READ)
              << " FI_REMOTE_WRITE=" << !!(flags & FI_REMOTE_WRITE)
              << " FI_REMOTE_READ=" << !!(flags & FI_REMOTE_READ) << ")"
              << " data: 0x" << std::hex << comp->data << std::dec
              << " context: " << comp->op_context << " len: " << comp->len;

    // Try to find the request associated with this context for debugging
    nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
    if (req) {
      NIXL_WARN << "Found request for zero-flags completion: XFER_ID="
                << req->xfer_id << " context=" << &req->ctx
                << " req_ptr=" << req << " rail=" << rail_id
                << " in_use=" << req->in_use << " op_type="
                << (req->operation_type == nixlLibfabricReq::WRITE  ? "WRITE"
                    : req->operation_type == nixlLibfabricReq::READ ? "READ"
                    : req->operation_type == nixlLibfabricReq::SEND ? "SEND"
                                                                    : "RECV");
    } else {
      NIXL_WARN << "No request found for zero-flags completion context "
                << comp->op_context;
    }

    // Check if this might be a spurious completion with flags=0
    if (flags == 0) {
      NIXL_WARN << "Completion with zero flags detected - this may be a "
                   "spurious completion "
                   "or cleanup event";
      // Don't treat zero flags as a fatal error, just skip processing
      return NIXL_SUCCESS;
    }

    NIXL_ERROR << "Unknown completion flags: " << std::hex << flags
               << " data: " << comp->data << " context: " << comp->op_context;
    return NIXL_ERR_BACKEND;
  }
}

// Handle local send completions (establishConnection, genNotif)
nixl_status_t
nixlLibfabricRail::processLocalSendCompletion(struct fi_cq_data_entry *comp) {
  // Release request back to pool
  nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
  if (req) {
    if (req->completion_callback) {
      NIXL_TRACE << "Calling completion callback for send request "
                 << req->xfer_id;
      req->completion_callback();
      NIXL_TRACE << "Completion callback completed for send";
    }
    // Always release request back to pool
    releaseRequest(req);
  } else {
    NIXL_ERROR << "No request found for context " << comp->op_context
               << " on rail " << rail_id;
  }

  return NIXL_SUCCESS;
}

// Handle local transfer completions (both read and write operations from
// postXfer)
nixl_status_t
nixlLibfabricRail::processLocalTransferCompletion(struct fi_cq_data_entry *comp,
                                                  const char *operation_type) {
  // Find the request from context to access the completion callback
  nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
  if (req) {
    // Call completion callback if it exists
    if (req->completion_callback) {
      NIXL_TRACE << "Calling completion callback for " << operation_type
                 << " request " << req->xfer_id;
      req->completion_callback();
      NIXL_TRACE << "Completion callback completed for " << operation_type;
    }

    // Always release request back to pool
    releaseRequest(req);
  } else {
    NIXL_ERROR << "No request found for " << operation_type
               << " completion context " << comp->op_context << " on rail "
               << rail_id;
  }

  return NIXL_SUCCESS;
}

// Handle remote receive completions (conn_req, conn_ack, notification messages)
nixl_status_t
nixlLibfabricRail::processRecvCompletion(struct fi_cq_data_entry *comp) {
  // Get the request from context to access the received buffer
  nixlLibfabricReq *req = findRequestFromContext(comp->op_context);
  if (!req) {
    NIXL_ERROR << "No request found for receive completion context on rail "
               << rail_id;
    return NIXL_ERR_BACKEND;
  }
  // Decode the immediate data format
  uint64_t msg_type = NIXL_GET_MSG_TYPE_FROM_IMM(comp->data);
  uint16_t agent_idx = NIXL_GET_AGENT_INDEX_FROM_IMM(comp->data);
  uint32_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(comp->data);
  NIXL_TRACE << "Received control message type " << msg_type
             << " agent_idx=" << agent_idx << " XFER_ID=" << xfer_id
             << " imm_data=0x" << std::hex << comp->data << std::dec;

  if (msg_type == NIXL_LIBFABRIC_MSG_CONNECT) {
    NIXL_TRACE << "Processing connection request on rail " << rail_id
               << " Xfer_id :" << xfer_id;
    // Use callback to handle connection request processing
    if (connectionReqCallback) {
      std::string serialized_data(static_cast<char *>(req->buffer),
                                  req->buffer_size);
      nixl_status_t callback_status = connectionReqCallback(
          agent_idx, serialized_data, const_cast<nixlLibfabricRail *>(this));
      if (callback_status != NIXL_SUCCESS) {
        NIXL_ERROR << "Connection request callback failed";
        return callback_status;
      }
      NIXL_TRACE << "Connection request processed via callback for rail "
                 << rail_id;
    } else {
      NIXL_ERROR << "No connection request callback set for rail " << rail_id;
      return NIXL_ERR_BACKEND;
    }
  } else if (msg_type == NIXL_LIBFABRIC_MSG_ACK) {
    NIXL_TRACE << "Processing connect request acknowledgement on rail "
               << rail_id;
    // Notify engine that connection is established via callback
    // TODO: validate the current state before calling callback
    if (connectionAckCallback) {
      connectionAckCallback(agent_idx, nullptr, ConnectionState::CONNECTED);
      NIXL_TRACE
          << "Connection state updated to CONNECTED via callback for rail "
          << rail_id;
    } else {
      NIXL_ERROR << "No connection state callback set for rail " << rail_id;
      return NIXL_ERR_BACKEND;
    }
  } else if (msg_type == NIXL_LIBFABRIC_MSG_NOTIFICATION) {
    NIXL_TRACE << "Processing notification request on rail " << rail_id
               << " Xfer_id :" << xfer_id;

    // Create string from received buffer using the actual received length from
    // completion entry
    std::string message(static_cast<char *>(req->buffer), comp->len);

    NIXL_TRACE << "Adding message: " << message
               << " to the notification list on rail " << rail_id;

    // Call engine's callback to store notification in central storage (like
    // reference)
    if (notificationCallback) {
      notificationCallback(message);
      NIXL_TRACE << "Notification stored via callback";
    } else {
      NIXL_ERROR << "No notification callback set!";
      return NIXL_ERR_BACKEND;
    }
  } else if (msg_type == NIXL_LIBFABRIC_MSG_DISCONNECT) {
    NIXL_TRACE << "Processing disconnect request from agent " << agent_idx
               << " on rail " << rail_id
               << "Currently not tracking the fi_addrs, so no callback for "
                  "disconnect to clean "
                  "up libfabric AV list";
  } else {
    NIXL_ERROR << "Unknown message type: " << std::hex << msg_type << std::dec;
    return NIXL_ERR_BACKEND;
  }

  // Clear the receive buffer after processing
  memset(req->buffer, 0, req->buffer_size);

  // Release the current request
  releaseRequest(req);

  // Post a new receive using new resource management system
  nixlLibfabricReq *new_req =
      allocateControlRequest(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE);
  if (!new_req) {
    NIXL_ERROR << "Failed to allocate request for subsequent receive on rail "
               << rail_id;
    return NIXL_ERR_BACKEND;
  }
  nixl_status_t status = postRecv(new_req);
  if (status != NIXL_SUCCESS) {
    NIXL_ERROR << "Failed to post subsequent receive on rail " << rail_id;
    releaseRequest(new_req);
    return status;
  }
  return NIXL_SUCCESS;
}

// Handle remote write completions (data arrival notification)
nixl_status_t nixlLibfabricRail::processRemoteWriteCompletion(
    struct fi_cq_data_entry *comp) const {
  // Decode the immediate data format
  uint64_t msg_type = NIXL_GET_MSG_TYPE_FROM_IMM(comp->data);
  uint16_t agent_idx = NIXL_GET_AGENT_INDEX_FROM_IMM(comp->data);
  uint32_t xfer_id = NIXL_GET_XFER_ID_FROM_IMM(comp->data);

  // For remote write completions, we don't need to post a new receive
  // The write operation doesn't consume a receive buffer
  if (msg_type == NIXL_LIBFABRIC_MSG_TRANSFER) {
    NIXL_TRACE << "Remote write completion on rail " << rail_id
               << " - received " << comp->len << " bytes"
               << " agent_idx=" << agent_idx << " XFER_ID=" << xfer_id
               << " imm_data=0x" << std::hex << comp->data << std::dec;

    // Call XFER_ID tracking callback to add received XFER_ID to global set
    if (xferIdCallback) {
      xferIdCallback(xfer_id);
      NIXL_TRACE << "Called XFER_ID callback for XFER_ID " << xfer_id;
    } else {
      NIXL_ERROR << "No XFER_ID callback set for rail " << rail_id;
      return NIXL_ERR_BACKEND;
    }
  }
  return NIXL_SUCCESS;
}

// Per-Rail Libfabric Operation Wrappers

nixl_status_t nixlLibfabricRail::postRecv(nixlLibfabricReq *req) const {
  if (!req || !req->buffer) {
    NIXL_ERROR << "Invalid request or buffer for receive on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }

  struct fi_msg msg = {};
  struct iovec msg_iov;
  void *desc = fi_mr_desc(req->mr); // Use request's MR

  // Setup message structure using request's buffer
  msg_iov.iov_base = req->buffer;
  msg_iov.iov_len = req->buffer_size;

  msg.msg_iov = &msg_iov;
  msg.desc = &desc;
  msg.iov_count = 1;
  msg.addr = FI_ADDR_UNSPEC;
  msg.context = &req->ctx; // Use request's context directly
  msg.data = 0;

  NIXL_TRACE << "Posting receive on endpoint: " << endpoint
             << " buffer: " << req->buffer << " size: " << req->buffer_size
             << " context: " << &req->ctx;

  int ret = fi_recvmsg(endpoint, &msg, 0);
  if (ret) {
    NIXL_ERROR << "fi_recvmsg failed on rail " << rail_id << ": "
               << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
  }

  NIXL_TRACE << "Receive posted successfully";
  return NIXL_SUCCESS;
}

// On -FI_EAGAIN from a post, the EFA provider's own send queue is full; no
// retry will make progress until the rail's completion queue is drained so
// new TX slots free up. Drive progressCompletionQueue inline from the
// caller thread before looping. Bumps the attempt counter and periodically
// logs at NIXL_DEBUG so a stuck rail still surfaces in operator logs without
// spamming on every retry.
//
// Returns true if the caller should retry the libfabric op, false if some
// unrecoverable CQ error occurred or the wall-clock retry deadline expired.
//
// GEX-3606: bound the wall-clock time spent in the EAGAIN retry loop. The
// EFA RDM provider can silently stop producing TX completions on a peer
// (described in the original ticket as the rail-0 wedge); without a
// deadline, postWrite/postSend/postRead spin in this helper indefinitely
// while the upper-layer scheduler hangs in initiate_send_transfer, taking
// down the whole DI pipeline. 30 s is well above any healthy fi_writedata
// turnaround and lets the operation surface as NIXL_ERR_BACKEND so
// upper-layer retry/cancellation can run. The deadline is per-call (reset
// once attempt hits 1) so a long-lived loop that drains many successful
// retries doesn't accidentally trip it.
static constexpr int kEagainRetryDeadlineSeconds = 30;
static bool progressAndLogForEagainRetry(nixlLibfabricRail *rail, int &attempt,
                                         const char *op_name) {
  static thread_local std::chrono::steady_clock::time_point retry_deadline{};
  if (attempt == 0) {
    retry_deadline = std::chrono::steady_clock::now() +
                     std::chrono::seconds(kEagainRetryDeadlineSeconds);
  }
  ++attempt;
  if (attempt % NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS == 0) {
    NIXL_DEBUG << op_name << " still retrying EAGAIN on rail " << rail->rail_id
               << " after " << attempt << " attempts";
  } else {
    NIXL_TRACE << op_name << " returned EAGAIN on rail " << rail->rail_id
               << ", retrying (attempt " << attempt << ")";
  }
  nixl_status_t progress_status = rail->progressCompletionQueue(false);
  if (progress_status != NIXL_SUCCESS && progress_status != NIXL_IN_PROG) {
    NIXL_ERROR << "progressCompletionQueue failed on rail " << rail->rail_id
               << " during " << op_name << " retry";
    return false;
  }
  if (std::chrono::steady_clock::now() > retry_deadline) {
    NIXL_ERROR << op_name << " exceeded " << kEagainRetryDeadlineSeconds
               << "s EAGAIN retry deadline on rail " << rail->rail_id
               << " after " << attempt
               << " attempts; surfacing as backend error (GEX-3606)";
    return false;
  }
  return true;
}

nixl_status_t nixlLibfabricRail::postSend(uint64_t immediate_data,
                                          fi_addr_t dest_addr,
                                          nixlLibfabricReq *req) const {
  if (req->buffer_size == 0 ||
      req->buffer_size > ControlRequestPool::BUFFER_SIZE) {
    NIXL_ERROR << "Invalid message size: " << req->buffer_size
               << " (max: " << ControlRequestPool::BUFFER_SIZE << ")";
    return NIXL_ERR_INVALID_PARAM;
  }

  // Prepare descriptor
  void *desc = fi_mr_desc(req->mr);

  NIXL_TRACE << "Sending data on endpoint: " << endpoint
             << " buffer: " << req->buffer << " size: " << req->buffer_size
             << " immediate_data: " << std::hex << immediate_data
             << " msg_type: " << NIXL_GET_MSG_TYPE_FROM_IMM(immediate_data)
             << " agent_idx: " << NIXL_GET_AGENT_INDEX_FROM_IMM(immediate_data)
             << " XFER_ID: " << NIXL_GET_XFER_ID_FROM_IMM(immediate_data)
             << " dest_addr: " << dest_addr << std::dec
             << " context: " << &req->ctx;

  // Retry indefinitely on EAGAIN with inline CQ progress. See upstream
  // ai-dynamo/nixl#856, #859, and #1335 for the matching upstream pattern.
  int ret = -FI_EAGAIN;
  int attempt = 0;
  while (true) {
    ret = fi_senddata(endpoint, req->buffer, req->buffer_size, desc,
                      immediate_data, dest_addr, &req->ctx);
    if (ret == 0) {
      NIXL_TRACE << "Send posted successfully"
                 << (attempt > 0
                         ? " after " + std::to_string(attempt + 1) + " attempts"
                         : "");
      return NIXL_SUCCESS;
    }
    if (ret == -FI_EAGAIN) {
      if (!progressAndLogForEagainRetry(const_cast<nixlLibfabricRail *>(this),
                                        attempt, "fi_senddata")) {
        return NIXL_ERR_BACKEND;
      }
      continue;
    }
    break;
  }
  NIXL_ERROR << "fi_senddata failed on rail " << rail_id << ": "
             << fi_strerror(-ret);
  return NIXL_ERR_BACKEND;
}

nixl_status_t
nixlLibfabricRail::postWrite(const void *local_buffer, size_t length,
                             void *local_desc, uint64_t immediate_data,
                             fi_addr_t dest_addr, uint64_t remote_addr,
                             uint64_t remote_key, nixlLibfabricReq *req) const {
  // Validation
  if (!req) {
    NIXL_ERROR << "Invalid request for write on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }

  // Setup and logging
  NIXL_TRACE << "Posting RDMA write on endpoint: " << std::hex << endpoint
             << " local_buffer: " << local_buffer << " length: " << length
             << " immediate_data: " << immediate_data
             << " dest_addr: " << dest_addr
             << " remote_addr: " << (void *)remote_addr
             << " remote_key: " << remote_key << " context: " << &req->ctx;

  // Retry indefinitely on EAGAIN with inline CQ progress. See upstream
  // ai-dynamo/nixl#856, #859, and #1335 for the matching upstream pattern.
  int ret = -FI_EAGAIN;
  int attempt = 0;
  while (true) {
    ret =
        fi_writedata(endpoint, local_buffer, length, local_desc, immediate_data,
                     dest_addr, remote_addr, remote_key, &req->ctx);
    if (ret == 0) {
      NIXL_TRACE << "RDMA write posted successfully"
                 << (attempt > 0
                         ? " after " + std::to_string(attempt + 1) + " attempts"
                         : "");
      return NIXL_SUCCESS;
    }
    if (ret == -FI_EAGAIN) {
      if (!progressAndLogForEagainRetry(const_cast<nixlLibfabricRail *>(this),
                                        attempt, "fi_writedata")) {
        return NIXL_ERR_BACKEND;
      }
      continue;
    }
    break;
  }
  NIXL_ERROR << "fi_writedata failed on rail " << rail_id << ": "
             << fi_strerror(-ret);
  return NIXL_ERR_BACKEND;
}

nixl_status_t nixlLibfabricRail::postRead(void *local_buffer, size_t length,
                                          void *local_desc, fi_addr_t dest_addr,
                                          uint64_t remote_addr,
                                          uint64_t remote_key,
                                          nixlLibfabricReq *req) const {
  if (!req) {
    NIXL_ERROR << "Invalid request for read on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }

  NIXL_TRACE << "Posting RDMA read on endpoint: " << std::hex << endpoint
             << " local_buffer: " << local_buffer << " length: " << length
             << " dest_addr: " << dest_addr
             << " remote_addr: " << (void *)remote_addr
             << " remote_key: " << remote_key << " context: " << &req->ctx;

  // Retry indefinitely on EAGAIN with inline CQ progress. See upstream
  // ai-dynamo/nixl#856, #859, and #1335 for the matching upstream pattern.
  int ret = -FI_EAGAIN;
  int attempt = 0;
  while (true) {
    ret = fi_read(endpoint, local_buffer, length, local_desc, dest_addr,
                  remote_addr, remote_key, &req->ctx);
    if (ret == 0) {
      NIXL_TRACE << "RDMA read posted successfully"
                 << (attempt > 0
                         ? " after " + std::to_string(attempt + 1) + " attempts"
                         : "");
      return NIXL_SUCCESS;
    }
    if (ret == -FI_EAGAIN) {
      if (!progressAndLogForEagainRetry(const_cast<nixlLibfabricRail *>(this),
                                        attempt, "fi_read")) {
        return NIXL_ERR_BACKEND;
      }
      continue;
    }
    break;
  }
  NIXL_ERROR << "fi_read failed on rail " << rail_id << ": "
             << fi_strerror(-ret);
  return NIXL_ERR_BACKEND;
}

// Memory Registration Methods

nixl_status_t nixlLibfabricRail::registerMemory(
    void *buffer, size_t length, nixl_mem_t mem_type, int device_id,
    uint64_t access_flags, struct fid_mr **mr_out, uint64_t *key_out) const {
  if (!buffer || !mr_out || !key_out) {
    NIXL_ERROR << "Invalid parameters on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }
  if (!domain) {
    NIXL_ERROR << "Domain not initialized on rail " << rail_id;
    return NIXL_ERR_BACKEND;
  }

  struct iovec iov = {};
  iov.iov_base = buffer;
  iov.iov_len = length;

  struct fi_mr_attr attr = {};
  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = access_flags;

  if (mem_type == VRAM_SEG) {
    attr.iface = FI_HMEM_CUDA;
    attr.device.cuda = device_id;
  }

  struct fid_mr *mr = nullptr;
  int ret = fi_mr_regattr(domain, &attr, 0, &mr);
  if (ret) {
    NIXL_ERROR << "fi_mr_regattr failed on rail " << rail_id << ": "
               << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
  }

  *mr_out = mr;
  *key_out = fi_mr_key(mr);
  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRail::deregisterMemory(struct fid_mr *mr) const {
  if (!mr) {
    NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }

  int ret = fi_close(&mr->fid);
  if (ret) {
    NIXL_ERROR << "fi_close failed on rail " << rail_id << ": "
               << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
  }

  return NIXL_SUCCESS;
}

// Address Vector Management Methods

nixl_status_t nixlLibfabricRail::insertAddress(const void *addr,
                                               fi_addr_t *fi_addr_out) const {
  if (!addr || !fi_addr_out) {
    NIXL_ERROR << "Invalid parameters on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }
  if (!av) {
    NIXL_ERROR << "Address vector not initialized on rail " << rail_id;
    return NIXL_ERR_BACKEND;
  }

  int ret = fi_av_insert(av, addr, 1, fi_addr_out, 0, NULL);
  if (ret != 1) {
    NIXL_ERROR << "fi_av_insert failed on rail " << rail_id << ": "
               << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
  }

  return NIXL_SUCCESS;
}

nixl_status_t nixlLibfabricRail::removeAddress(fi_addr_t fi_addr) const {
  if (fi_addr == FI_ADDR_UNSPEC) {
    NIXL_ERROR << "Invalid fi_addr parameter on rail " << rail_id;
    return NIXL_ERR_INVALID_PARAM;
  }
  if (!av) {
    NIXL_ERROR << "Address vector not initialized on rail " << rail_id;
    return NIXL_ERR_BACKEND;
  }

  int ret = fi_av_remove(av, &fi_addr, 1, 0);
  if (ret != 0) {
    NIXL_ERROR << "fi_av_remove failed on rail " << rail_id << ": "
               << fi_strerror(-ret);
    return NIXL_ERR_BACKEND;
  }

  return NIXL_SUCCESS;
}

// Memory Descriptor Helper Methods

void *nixlLibfabricRail::getMemoryDescriptor(struct fid_mr *mr) const {
  if (!mr) {
    NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
    return nullptr;
  }
  return fi_mr_desc(mr);
}

uint64_t nixlLibfabricRail::getMemoryKey(struct fid_mr *mr) const {
  if (!mr) {
    NIXL_ERROR << "Invalid MR parameter on rail " << rail_id;
    return 0;
  }
  return fi_mr_key(mr);
}

// Optimized Resource Management Methods

nixlLibfabricReq *
nixlLibfabricRail::allocateControlRequest(size_t needed_size) {
  while (true) {
    nixlLibfabricReq *req = control_request_pool_.allocate(needed_size);
    if (req)
      return req;
    // Pool is full — block until the CM thread releases a slot.
    std::unique_lock<std::mutex> lock(control_slots_mutex_);
    control_slots_cv_.wait_for(lock, std::chrono::milliseconds(1));
  }
}

nixlLibfabricReq *
nixlLibfabricRail::allocateDataRequest(nixlLibfabricReq::OpType op_type) {
  return data_request_pool_.allocate(op_type);
}

void nixlLibfabricRail::releaseRequest(nixlLibfabricReq *req) {
  if (!req) {
    NIXL_ERROR << "Null request provided to releaseRequest on rail " << rail_id;
    return;
  }
  // Determine which pool to release to based on operation type
  if (req->operation_type == nixlLibfabricReq::SEND ||
      req->operation_type == nixlLibfabricReq::RECV) {
    control_request_pool_.release(req);
    control_slots_cv_.notify_one();
  } else {
    data_request_pool_.release(req);
  }
  NIXL_TRACE << "Released request with XFER_ID " << req->xfer_id;
}

nixlLibfabricReq *
nixlLibfabricRail::findRequestFromContext(void *context) const {
  if (!context) {
    NIXL_ERROR << "Null context provided to findRequestFromContext on rail "
               << rail_id;
    return nullptr;
  }
  // Try control pool first
  nixlLibfabricReq *req = control_request_pool_.findByContext(context);
  if (req) {
    return req;
  }
  // Try data pool
  req = data_request_pool_.findByContext(context);
  if (req) {
    return req;
  }
  NIXL_ERROR << "No request found for context " << context << " on rail "
             << rail_id;
  return nullptr;
}

} // namespace M::MLRT
