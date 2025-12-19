//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_NIXL_TYPES_H
#define MLRT_DRIVER_NIXL_NIXL_TYPES_H

#include "Support/SymbolExport.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace M::AsyncRT::NIXL {

/*** Forward declarations ***/
class nixlSerDes;
class nixlDlistH;
class nixlBackendH;
class nixlXferReqH;
class nixlAgentData;

/*** NIXL memory type, operation and status enums ***/

/**
 * @enum   nixl_mem_t
 * @brief  An enumeration of segment types for NIXL
 *         FILE_SEG must be last
 */
enum class nixl_mem_t : std::uint8_t {
  DRAM_SEG,
  VRAM_SEG,
  BLK_SEG,
  OBJ_SEG,
  FILE_SEG,
  // Sentinel for number of memory types.
  COUNT,
};

/**
 * @enum   nixl_xfer_op_t
 * @brief  An enumeration of different transfer types for NIXL
 */
enum class nixl_xfer_op_t : std::uint8_t { NIXL_READ, NIXL_WRITE };

/**
 * @enum   nixl_status_t
 * @brief  An enumeration of status values and error codes for NIXL
 */
enum class nixl_status_t {
  NIXL_IN_PROG = 1,
  NIXL_SUCCESS = 0,
  NIXL_ERR_NOT_POSTED = -1,
  NIXL_ERR_INVALID_PARAM = -2,
  NIXL_ERR_BACKEND = -3,
  NIXL_ERR_NOT_FOUND = -4,
  NIXL_ERR_MISMATCH = -5,
  NIXL_ERR_NOT_ALLOWED = -6,
  NIXL_ERR_REPOST_ACTIVE = -7,
  NIXL_ERR_UNKNOWN = -8,
  NIXL_ERR_NOT_SUPPORTED = -9,
  NIXL_ERR_REMOTE_DISCONNECT = -10,
};

constexpr bool isErrorStatus(nixl_status_t status) {
  return !llvm::is_contained(
      {nixl_status_t::NIXL_IN_PROG, nixl_status_t::NIXL_SUCCESS}, status);
}

/**
 * @enum nixl_thread_sync_t
 * @brief An enumeration of supported synchronization modes for NIXL
 */
enum class nixl_thread_sync_t : std::uint8_t {
  NIXL_THREAD_SYNC_NONE,
  NIXL_THREAD_SYNC_STRICT,
  NIXL_THREAD_SYNC_RW,
  NIXL_THREAD_SYNC_DEFAULT = NIXL_THREAD_SYNC_NONE,
};

/**
 * @namespace nixlEnumStrings
 * @brief     This namespace to get string representation
 *            of different enums
 */
namespace nixlEnumStrings {
MODULAR_CXX_EXPORT std::string_view memTypeStr(nixl_mem_t mem);
std::string_view xferOpStr(nixl_xfer_op_t op);
MODULAR_CXX_EXPORT const char *statusStr(nixl_status_t status);
} // namespace nixlEnumStrings

/*** NIXL typedefs and defines used in the API ***/

/**
 * @brief A typedef for a std::string to identify nixl backends
 */
using nixl_backend_t = std::string;

/**
 * @brief A typedef for a std::string as nixl blob
 *        std::string supports \0 natively, so it can be looked as a void* of
 * data, with specified length. Giving it a new name to be clear in the API and
 *        preventing users to think it's a string and call c_str().
 */
using nixl_blob_t = std::string;

/**
 * @brief Type-safe non-owning reference to a block of memory.
 */
struct blob_ref {
  const void *data;
  size_t size;

  blob_ref(const void *data, size_t size) : data{data}, size{size} {}

  explicit blob_ref(std::string_view str)
      : data{reinterpret_cast<const void *>(str.data())}, size{str.size()} {}
};

/**
 * @brief A typedef for a std::vector<nixl_mem_t> to create nixl_mem_list_t
 * objects.
 */
using nixl_mem_list_t = std::vector<nixl_mem_t>;

/**
 * @brief A typedef for a  std::unordered_map<std::string, std::string>
 *        to hold nixl_b_params_t .
 */
using nixl_b_params_t = std::unordered_map<std::string, std::string>;

/**
 * @brief A typedef for a  std::unordered_map<std::string,
 * std::vector<nixl_blob_t>> to hold nixl_notifs_t (nixl notifications)
 */
using nixl_notifs_t = std::unordered_map<std::string, std::vector<nixl_blob_t>>;

/**
 * @brief A typedef for std::optional<nixl_b_params_t> for querying memory
 * results Validity of a nixl_query_resp_t can be checked by has_value() method,
 *        and if true, the dictionary can be accessed by value() method.
 */
using nixl_query_resp_t = std::optional<nixl_b_params_t>;

/**
 * @brief A constant to define the default communication port.
 */
constexpr int default_comm_port = 8888;

/**
 * @struct nixlAgentOptionalArgs
 * @brief A structure for optional argument that can be provided to relevant
 * agent methods.
 */
struct nixlAgentOptionalArgs {
  /**
   * @var backends vector to specify a list of backend handles, to limit the
   * list of backends to be considered. Used in registerMem / deregisterMem
   *      makeConnection / prepXferDlist / makeXferReq / createXferReq /
   * GetNotifs / GenNotif
   */
  std::vector<nixlBackendH *> backends;

  /**
   * @var notifMsg A message to be used in createXferReq / makeXferReq /
   * postXferReq, if a notification message is desired
   */
  nixl_blob_t notifMsg;
  /**
   * @var hasNotif boolean value to indicate that a notification is provided, or
   * to remove notification during a repost. If set to false, notifMsg is not
   * checked.
   */
  bool hasNotif = false;

  /**
   * @var makeXferReq boolean to skip merging consecutive descriptors, used in
   * makeXferReq.
   */
  bool skipDescMerge = false;

  /**
   * @var includeConnInfo boolean to include connection information in the
   * metadata, used in getLocalPartialMD.
   */
  bool includeConnInfo = false;

  /**
   * @var ipAddr Used to specify the IP address of a remote peer for metadata
   * transfer. used in sendLocalMD, fetchRemoteMD, invalidateLocalMD,
   * sendLocalPartialMD.
   */
  std::string ipAddr;
  /**
   * @var port Used to specify the port of a remote peer, ipAddr must also be
   * set used in sendLocalMD, fetchRemoteMD, invalidateLocalMD,
   * sendLocalPartialMD.
   */
  int port = default_comm_port;
};
/**
 * @brief A typedef for a nixlAgentOptionalArgs
 *        for providing extra optional arguments
 */
using nixl_opt_args_t = nixlAgentOptionalArgs;

/**
 * @brief An empty string that indicates the descriptor list is being prepared
 * for the local agent as an initiator in prepXferDlist method.
 */
constexpr std::string_view NIXL_INIT_AGENT = "";

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_NIXL_TYPES_H
