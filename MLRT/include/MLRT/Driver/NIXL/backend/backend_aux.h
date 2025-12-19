//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_BACKEND_BACKEND_AUX_H
#define ASYNCRT_NIXL_BACKEND_BACKEND_AUX_H

#include "MLRT/Driver/NIXL/nixl_descriptors.h"
#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/utils/common/nixl_time.h"
#include "llvm/ADT/Twine.h"
#include <string>

namespace M::AsyncRT::NIXL {

// Might be removed to be decided by backend, or changed to high
// level direction or so.
using notif_list_t = std::vector<std::pair<std::string, std::string>>;

struct nixlBackendOptionalArgs {
  // During postXfer, user might ask for a notification if supported
  nixl_blob_t notifMsg;
  bool hasNotif = false;
};

using nixl_opt_b_args_t = nixlBackendOptionalArgs;

// A base class to point to backend initialization data
// User doesn't know about fields such as local_agent but can access it
// after the backend is initialized by agent. If we needed to make it private
// from the user, we should make nixlBackendEngine/nixlAgent friend classes.
struct nixlBackendInitParams {
  std::string localAgent;

  nixl_backend_t type;
  nixl_b_params_t *customParams;

  bool enableProgTh;
  nixlTime::us_t pthrDelay;
  nixl_thread_sync_t syncMode;
};

// Base class to have a common pointer type.
class nixlBackendReqH {
public:
  nixlBackendReqH() = default;

protected:
  virtual ~nixlBackendReqH() = default;
};

// Base class to have a common pointer type for different backendMD.
class nixlBackendMD {
protected:
  const bool isPrivateMD;

public:
  nixlBackendMD(bool isPrivate) : isPrivateMD{isPrivate} {}

  virtual ~nixlBackendMD() = default;
};

// Each backend can have different connection requirement
// This class would include the required information to make
// a connection to a remote node. Note that local information
// is passed during the constructor and through BackendInitParams
struct nixlBackendConnMD {
  // And some other details
  std::string dstIpAddress;
  uint16_t dstPort;
};

// A pointer required to a metadata object for backends next to each BasicDesc
class nixlMetaDesc : public nixlBasicDesc {
public:
  // To be able to point to any object.
  nixlBackendMD *metadataP{nullptr};

  nixlMetaDesc() = default;

  // No serializer or deserializer, using parent not to expose the metadata

  inline friend bool operator==(const nixlMetaDesc &lhs,
                                const nixlMetaDesc &rhs) {
    return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) &&
            (lhs.metadataP == rhs.metadataP));
  }

  inline void print(Twine suffix) const {
    nixlBasicDesc::print(", Backend ptr val: " + Twine((uintptr_t)metadataP) +
                         suffix);
  }
};

using nixl_meta_dlist_t = nixlDescList<nixlMetaDesc>;

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_BACKEND_BACKEND_AUX_H
