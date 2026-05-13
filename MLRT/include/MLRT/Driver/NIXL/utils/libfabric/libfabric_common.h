//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
// SPDX-License-Identifier: Apache-2.0

#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H

#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "MLRT/Driver/NIXL/nixl.h"

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_rma.h>

// Libfabric configuration constants
#define NIXL_LIBFABRIC_HAVE_CUDA 0
#define NIXL_LIBFABRIC_DEFAULT_CONTROL_RAILS 1
// Per-control-request scratch buffer size. Sized to fit a
// ``BinaryNotification`` (currently 66820 bytes with
// ``NIXL_LIBFABRIC_MAX_XFER_IDS = 16384``); rounded to 128 KiB for
// alignment / future headroom. Cost per rail is
// ``NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE * CONTROL_REQUESTS_PER_RAIL``
// (256 * 128 KiB = 32 MiB per rail).
#define NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE 131072
#define NIXL_LIBFABRIC_CQ_SREAD_TIMEOUT_SEC 1
#define NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD (128 * 1024) // 128KB
// Maximum XFER_IDs per notification. With 8 EFA rails on AWS B200 each
// 4 MiB block stripes to 8 RDMAs, so a cap of 16384 supports batches up
// to 16384/8 = 2048 blocks (= 8 GiB) before the receiver-visible
// notification starts losing xfer_ids. Submitted requests beyond the
// cap still complete on the sender; only the receiver's notification
// list is bounded.
// Raised from 4096 (2 GiB) after probe runs confirmed batches up to
// ~13815 xfer_ids on Kimi-K2.5 NVfp4 multiturn at mc=16 (B200 AWS).
#define NIXL_LIBFABRIC_MAX_XFER_IDS 16384
// Log every N EAGAIN retry attempts in postSend/postWrite/postRead. Upstream
// NIXL chose 100 to prevent log spam on transiently-saturated EFA send
// queues while still surfacing a warning when the retry loop is not making
// progress. See upstream ai-dynamo/nixl#859 for context.
#define NIXL_LIBFABRIC_LOG_INTERVAL_ATTEMPTS 100
#define LF_EP_NAME_MAX_LEN 56

// The immediate data associated with an RDMA operation is 32 bits and is
// divided as follows: | 4-bit MSG TYPE flag | 8-bit agent index | 20-bit
// XFER_ID |

// Optimized bit field constants (compile-time computed)
#define NIXL_MSG_TYPE_BITS 4
#define NIXL_AGENT_INDEX_BITS 8
#define NIXL_XFER_ID_BITS 20

// Pre-computed shift amounts for better performance
#define NIXL_MSG_TYPE_SHIFT 0
#define NIXL_AGENT_INDEX_SHIFT 4
#define NIXL_XFER_ID_SHIFT 12

// Pre-computed masks (compile-time constants)
#define NIXL_MSG_TYPE_MASK 0xFU     // 0x0000000F (4 bits)
#define NIXL_AGENT_INDEX_MASK 0xFFU // 0x000000FF (8 bits)
#define NIXL_XFER_ID_MASK 0xFFFFFU  // 0x000FFFFF (20 bits)

// Message type constants
#define NIXL_LIBFABRIC_MSG_CONNECT 0
#define NIXL_LIBFABRIC_MSG_ACK 1
#define NIXL_LIBFABRIC_MSG_NOTIFICATION 2
#define NIXL_LIBFABRIC_MSG_DISCONNECT 3
#define NIXL_LIBFABRIC_MSG_TRANSFER 4

// Single-operation immediate data extraction (no intermediate shifts)
#define NIXL_GET_MSG_TYPE_FROM_IMM(data) ((data) & NIXL_MSG_TYPE_MASK)
#define NIXL_GET_AGENT_INDEX_FROM_IMM(data)                                    \
  (((data) >> NIXL_AGENT_INDEX_SHIFT) & NIXL_AGENT_INDEX_MASK)
#define NIXL_GET_XFER_ID_FROM_IMM(data)                                        \
  (((data) >> NIXL_XFER_ID_SHIFT) & NIXL_XFER_ID_MASK)

// Single-operation immediate data creation (minimal bit operations)
#define NIXL_MAKE_IMM_DATA(msg_type, agent_idx, xfer_id)                       \
  (((uint64_t)(msg_type) & NIXL_MSG_TYPE_MASK) |                               \
   (((uint64_t)(agent_idx) & NIXL_AGENT_INDEX_MASK)                            \
    << NIXL_AGENT_INDEX_SHIFT) |                                               \
   (((uint64_t)(xfer_id) & NIXL_XFER_ID_MASK) << NIXL_XFER_ID_SHIFT))

namespace M::MLRT {
using enum nixl_mem_t;
using enum nixl_status_t;
using enum nixl_xfer_op_t;

/**
 * @brief Binary notification format to eliminate SerDes string operations
 *
 * This structure provides a fixed-size, binary format for notifications
 * to avoid expensive string serialization/deserialization operations.
 * Used for high-performance notification passing between agents.
 */
struct BinaryNotification {
  char agent_name[256];   // Fixed-size agent name (null-terminated)
  char message[1024];     // Fixed-size message (null-terminated)
  uint32_t xfer_id_count; // Number of XFER_IDs
  uint32_t xfer_ids[NIXL_LIBFABRIC_MAX_XFER_IDS]; // Fixed array of XFER_IDs

  /** @brief Clear all fields to zero */
  void clear() { memset(this, 0, sizeof(BinaryNotification)); }

  /** @brief Set agent name with bounds checking */
  void setAgentName(const std::string &name) {
    strncpy(agent_name, name.c_str(), sizeof(agent_name) - 1);
    agent_name[sizeof(agent_name) - 1] = '\0';
  }

  /** @brief Set message with bounds checking */
  void setMessage(const std::string &msg) {
    strncpy(message, msg.c_str(), sizeof(message) - 1);
    message[sizeof(message) - 1] = '\0';
  }

  /** @brief Add XFER_ID if space available. Returns true on success, false
   *  if the fixed-size xfer_ids array is full (caller can log/diagnose).
   *  Callers must not equate the post-call ``xfer_id_count`` with the
   *  number of RDMAs they submitted: the two diverge whenever this
   *  returns false, and any submission-count tracking that consumes the
   *  notification field will leave the sender's completion accounting
   *  wedged on the unaccounted tail. */
  bool addXferId(uint32_t xfer_id) {
    if (xfer_id_count < NIXL_LIBFABRIC_MAX_XFER_IDS) {
      xfer_ids[xfer_id_count++] = xfer_id;
      return true;
    }
    return false;
  }

  /** @brief Get agent name as string */
  std::string getAgentName() const { return std::string(agent_name); }

  /** @brief Get message as string */
  std::string getMessage() const { return std::string(message); }

  /** @brief Get all XFER_IDs as unordered set */
  std::unordered_set<uint32_t> getXferIds() const {
    std::unordered_set<uint32_t> result;
    for (uint32_t i = 0; i < xfer_id_count; ++i) {
      result.insert(xfer_ids[i]);
    }
    return result;
  }
};

// Global XFER_ID management
namespace LibfabricUtils {
// Pre-allocate XFER_IDs during initialization (NOT fast path)
std::vector<uint32_t> preallocateXferIds(size_t count);
} // namespace LibfabricUtils

// Utility functions
namespace LibfabricUtils {
// Device discovery
std::vector<std::string> getAvailableEfaDevices();
// String utilities
std::string hexdump(const void *data);
} // namespace LibfabricUtils

} // namespace M::MLRT

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_COMMON_H
