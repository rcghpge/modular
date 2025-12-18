//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_MEM_SECTION_H
#define ASYNCRT_NIXL_MEM_SECTION_H

#include "Support/STLExtras.h"
#include "backend/backend_engine.h"
#include "nixl.h"
#include "nixl_descriptors.h"
#include <array>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace M::AsyncRT::NIXL {

using section_key_t = std::pair<nixl_mem_t, nixlBackendEngine *>;
using backend_set_t = std::set<nixlBackendEngine *>;
using backend_map_t = std::unordered_map<nixl_backend_t, nixlBackendEngine *>;

/**
 * @brief Section descriptor for nixl
 *
 * This class is used to store a section descriptor for nixl.
 * It is derived from nixlMetaDesc and contains the meta blob for the section.
 */
class nixlSectionDesc : public nixlMetaDesc {
public:
  nixl_blob_t metaBlob;

  using nixlMetaDesc::nixlMetaDesc;

  nixl_blob_t serialize() const {
    // Serialize only the meta blob. metadataP is private so we don't include
    // it. The other side will deserialize it as nixlBlobDesc.
    return nixlBasicDesc::serialize() + metaBlob;
  }

  inline friend bool operator==(const nixlSectionDesc &lhs,
                                const nixlSectionDesc &rhs) {
    return (static_cast<nixlMetaDesc>(lhs) == static_cast<nixlMetaDesc>(rhs));
  }

  inline void print(const std::string &suffix) const {
    nixlMetaDesc::print(", meta blob: " + metaBlob + suffix);
  }
};

class nixlSecDescList : public nixlDescList<nixlSectionDesc> {
public:
  // Disable parent's convenience constructors that allow pre-sizing.
  // That particular constructor that allows specifying init_size is
  // intentionally omitted here.
  explicit nixlSecDescList(nixl_mem_t type)
      : nixlDescList<nixlSectionDesc>(type, 0) {}

  using nixlDescList<nixlSectionDesc>::operator[]; // bring in const overload

  void addDesc(const nixlSectionDesc &desc) override;

  bool verifySorted() const;

  nixlSectionDesc &operator[](size_t index) override;

  int getIndex(const nixlBasicDesc &query) const override;

  int getCoveringIndex(const nixlBasicDesc &query) const;

  void resize(const size_t &count) override;

  nixlSecDescList(const nixlSecDescList &) = default;
  nixlSecDescList &operator=(const nixlSecDescList &) = default;
};

using nixl_sec_dlist_t = nixlSecDescList;
using section_map_t = std::map<section_key_t, nixl_sec_dlist_t *>;

class nixlMemSection {
protected:
  std::array<backend_set_t, to_underlying(nixl_mem_t::COUNT)> memToBackend;
  section_map_t sectionMap;

public:
  nixlMemSection() = default;

  backend_set_t *queryBackends(const nixl_mem_t &mem);

  nixl_status_t populate(const nixl_xfer_dlist_t &query,
                         nixlBackendEngine *backend,
                         nixl_meta_dlist_t &resp) const;

  virtual ~nixlMemSection() = 0; // Making the class abstract
};

class nixlLocalSection : public nixlMemSection {
public:
  nixl_status_t addDescList(const nixl_reg_dlist_t &mem_elms,
                            nixlBackendEngine &backend,
                            nixl_sec_dlist_t &remote_self);

  // Each nixlBasicDesc should be same as original registration region
  nixl_status_t remDescList(const nixl_reg_dlist_t &mem_elms,
                            nixlBackendEngine &backend);

  void serialize(nixlSerDes &serializer) const;

  nixl_status_t serializePartial(nixlSerDes &serializer,
                                 const backend_set_t &backends,
                                 const nixl_reg_dlist_t &mem_elms) const;

  // TODO(GENAI-170): Clean up ownership and divergent destructors with
  // `nixlRemoteSection`.
  ~nixlLocalSection();
};

class nixlRemoteSection : public nixlMemSection {
private:
  std::string agentName;

  nixl_status_t addDescList(const nixl_reg_dlist_t &mem_elms,
                            nixlBackendEngine *backend);

public:
  nixlRemoteSection(std::string name) : agentName(std::move(name)) {}

  nixl_status_t loadRemoteData(nixlSerDes *deserializer,
                               backend_map_t &backendToEngineMap);

  // When adding self as a remote agent for local operations
  nixl_status_t loadLocalData(const nixl_sec_dlist_t &mem_elms,
                              nixlBackendEngine *backend);
  ~nixlRemoteSection();
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_MEM_SECTION_H
