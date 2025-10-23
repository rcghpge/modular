//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/nixl_descriptors.h"
#include "AsyncRT/NIXL/backend/backend_aux.h"
#include "AsyncRT/NIXL/mem_section.h"
#include "AsyncRT/NIXL/nixl_types.h"
#include "AsyncRT/NIXL/utils/common/nixl_log.h"
#include "AsyncRT/NIXL/utils/serdes/serdes.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace M;
using namespace M::AsyncRT::NIXL;
using namespace std::string_view_literals;

/*** Class nixlBasicDesc implementation ***/

// No Virtual function in nixlBasicDesc class or its children, as we want
// each object to just have the members during serialization.

nixlBasicDesc::nixlBasicDesc(const nixl_blob_t &blob) {
  if (blob.size() == sizeof(nixlBasicDesc)) {
    blob.copy(reinterpret_cast<char *>(this), sizeof(nixlBasicDesc));
  } else { // Error indicator, not possible by descList deserializer call
    addr = 0;
    len = 0;
    devId = 0;
  }
}

bool nixlBasicDesc::operator<(const nixlBasicDesc &desc) const {
  if (devId != desc.devId)
    return (devId < desc.devId);
  else if (addr != desc.addr)
    return (addr < desc.addr);
  else
    return (len < desc.len);
}

bool M::AsyncRT::NIXL::operator==(const nixlBasicDesc &lhs,
                                  const nixlBasicDesc &rhs) {
  return ((lhs.addr == rhs.addr) && (lhs.len == rhs.len) &&
          (lhs.devId == rhs.devId));
}

bool M::AsyncRT::NIXL::operator!=(const nixlBasicDesc &lhs,
                                  const nixlBasicDesc &rhs) {
  return !(lhs == rhs);
}

bool M::AsyncRT::NIXL::operator==(const nixlBlobDesc &lhs,
                                  const nixlBlobDesc &rhs) {
  return (((nixlBasicDesc)lhs == (nixlBasicDesc)rhs) &&
          (lhs.metaInfo == rhs.metaInfo));
}

bool M::AsyncRT::NIXL::operator!=(const nixlBlobDesc &lhs,
                                  const nixlBlobDesc &rhs) {
  return !(lhs == rhs);
}

bool nixlBasicDesc::covers(const nixlBasicDesc &query) const {
  return (devId == query.devId) && (addr <= query.addr) &&
         (addr + len >= query.addr + query.len);
}

bool nixlBasicDesc::overlaps(const nixlBasicDesc &query) const {
  if (devId != query.devId)
    return false;
  if ((addr + len <= query.addr) || (query.addr + query.len <= addr))
    return false;
  return true;
}

nixl_blob_t nixlBasicDesc::serialize() const {
  return std::string(reinterpret_cast<const char *>(this),
                     sizeof(nixlBasicDesc));
}

void nixlBasicDesc::print(Twine suffix) const {
  llvm::errs() << "LOG: Desc (" << addr << ", " << len << ") from devID "
               << devId << suffix << "\n";
}

/*** Class nixlBlobDesc implementation ***/

nixlBlobDesc::nixlBlobDesc(const nixl_blob_t &blob) {
  size_t meta_size = blob.size() - sizeof(nixlBasicDesc);
  if (meta_size > 0) {
    metaInfo.resize(meta_size);
    blob.copy(reinterpret_cast<char *>(this), sizeof(nixlBasicDesc));
    blob.copy(reinterpret_cast<char *>(&metaInfo[0]), meta_size,
              sizeof(nixlBasicDesc));
  } else if (meta_size == 0) {
    blob.copy(reinterpret_cast<char *>(this), sizeof(nixlBasicDesc));
  } else { // Error
    // TODO(GENAI-170): convert to fallible constructor idiom.
    addr = 0;
    len = 0;
    devId = 0;
    metaInfo.resize(0);
  }
}

nixl_blob_t nixlBlobDesc::serialize() const {
  return nixlBasicDesc::serialize() + metaInfo;
}

void nixlBlobDesc::print(std::string_view suffix) const {
  nixlBasicDesc::print((Twine(", Metadata: ") + metaInfo + suffix).str());
}

/*** Class nixlDescList implementation ***/

// The template is used to select from nixlBasicDesc/nixlMetaDesc/nixlBlobDesc
// There are no virtual functions, so the object is all data, no pointers.

template <class T>
nixlDescList<T>::nixlDescList(nixlSerDes *deserializer) {
  // nixlMetaDesc and nixlSectionDesc should be internal and not be serialized
  if constexpr (std::is_same<nixlMetaDesc, T>::value ||
                std::is_same<nixlSectionDesc, T>::value)
    return;

  size_t n_desc;
  std::string str;

  descs.clear();

  str = deserializer->getStr("nixlDList"); // Object type
  if (str.size() == 0)
    return;

  if (deserializer->getBuf("t"sv, &type, sizeof(type)) !=
      nixl_status_t::NIXL_SUCCESS)
    return;

  if (deserializer->getBuf("n"sv, &n_desc, sizeof(n_desc)) !=
      nixl_status_t::NIXL_SUCCESS)
    return;

  if constexpr (std::is_same<nixlBasicDesc, T>::value) {
    // Contiguous in memory, so no need for per elm deserialization
    if (str != "nixlBDList")
      return;
    str = deserializer->getStr("");
    if (str.size() != n_desc * sizeof(nixlBasicDesc))
      return;
    // If size is proper, deserializer cannot fail
    descs.resize(n_desc);
    str.copy(reinterpret_cast<char *>(descs.data()), str.size());

  } else if constexpr (std::is_same<nixlBlobDesc, T>::value) {
    if (str != "nixlSDList")
      return;
    for (size_t i = 0; i < n_desc; ++i) {
      str = deserializer->getStr("");
      // If size is proper, deserializer cannot fail
      // Allowing empty strings, might change later
      if (str.size() < sizeof(nixlBasicDesc)) {
        descs.clear();
        return;
      }
      T elm(str);
      descs.push_back(elm);
    }
  } else {
    // TODO(GENAI-170): handle nixlDescList<nixlMetaDesc>.
    // static_assert(false && "unknown type: error");
    return;
  }
}

template <class T>
void nixlDescList<T>::addDesc(const T &desc) {
  descs.push_back(desc);
}

template <class T>
bool nixlDescList<T>::overlaps(const T &desc, int &index) const {
  for (size_t i = 0; i < descs.size(); ++i) {
    if (descs[i].overlaps(desc)) {
      index = i;
      return true;
    }
  }
  index = descs.size();
  return false;
}

template <class T>
bool nixlDescList<T>::hasOverlaps() const {
  if ((descs.size() == 0) || (descs.size() == 1))
    return false;

  for (size_t i = 0; i < descs.size() - 1; ++i)
    for (size_t j = i + 1; j < descs.size(); ++j)
      if (descs[i].overlaps(descs[j]))
        return true;
  return false;
}

template <class T>
void nixlDescList<T>::remDesc(const int index) {
  if (((size_t)index >= descs.size()) || (index < 0)) {
    // TODO(GENAI-170): raise ErrorOr.
    NIXL_FATAL << "Index is out of range";
    std::abort();
  }
  descs.erase(descs.begin() + index);
}

template <class T>
void nixlDescList<T>::resize(const size_t &count) {
  descs.resize(count);
}

template <class T>
nixlDescList<nixlBasicDesc> nixlDescList<T>::trim() const {

  if constexpr (std::is_same<nixlBasicDesc, T>::value) {
    return *this;
  } else {
    nixlDescList<nixlBasicDesc> trimmed(type);
    nixlBasicDesc *p;

    for (auto &elm : descs) {
      p = (nixlBasicDesc *)(&elm);
      trimmed.addDesc(*p);
    }

    // No failure scenario
    return trimmed;
  }
}

template <class T>
int nixlDescList<T>::getIndex(const nixlBasicDesc &query) const {
  auto itr = std::find(descs.begin(), descs.end(), query);
  if (itr == descs.end())
    return to_underlying(nixl_status_t::NIXL_ERR_NOT_FOUND); // not found
  return itr - descs.begin();
}

template <class T>
template <class>
void nixlDescList<T>::serialize(nixlSerDes *serializer) const {
  static_assert(std::is_same<nixlBasicDesc, T>::value ||
                std::is_same<nixlBlobDesc, T>::value ||
                std::is_same<nixlSectionDesc, T>::value);

  const size_t nDesc = descs.size();

  // For now very few descriptor types, if needed can add a name method to each
  // descriptor. std::string_view(typeid(T).name()) is compiler dependent
  if constexpr (std::is_same<nixlBasicDesc, T>::value)
    serializer->addStr("nixlDList", "nixlBDList");
  // We serialize SectionDesc the same as BlobDesc so it will be deserialized as
  // BlobDesc on the other side
  else
    serializer->addStr("nixlDList", "nixlSDList");

  serializer->addBuf("t", &type, sizeof(type));
  serializer->addBuf("n", &nDesc, sizeof(nDesc));

  if (nDesc == 0)
    return; // Unusual, but supporting it

  // Optimization for nixlBasicDesc,
  // contiguous in memory, so no need for per elm serialization
  if constexpr (std::is_same<nixlBasicDesc, T>::value) {
    serializer->addBuf("", reinterpret_cast<const void *>(descs.data()),
                       nDesc * sizeof(nixlBasicDesc));
  } else { // already checked it can be only nixlBlobDesc or nixlSectionDesc
    for (auto &elm : descs) {
      serializer->addStr("", elm.serialize());
    }
  }
}

template <class T>
void nixlDescList<T>::print() const {
  std::cout << "LOG: DescList of mem type " << nixlEnumStrings::memTypeStr(type)
            << "\n";
  for (auto &elm : descs) {
    std::cout << "    ";
    elm.print("");
  }
}

template <class T>
bool M::AsyncRT::NIXL::operator==(const nixlDescList<T> &lhs,
                                  const nixlDescList<T> &rhs) {
  if ((lhs.getType() != rhs.getType()) || (lhs.descCount() != rhs.descCount()))
    return false;

  for (size_t i = 0; i < lhs.descs.size(); ++i)
    if (lhs.descs[i] != rhs.descs[i])
      return false;
  return true;
}

namespace M::AsyncRT::NIXL {

// Since we implement a template class declared in a header files, this is
// necessary
template class nixlDescList<nixlBasicDesc>;
template class nixlDescList<nixlMetaDesc>;
template class MODULAR_CXX_EXPORT nixlDescList<nixlBlobDesc>;
template class nixlDescList<nixlSectionDesc>;
template void nixlDescList<nixlBasicDesc>::serialize<>(nixlSerDes *) const;
template void nixlDescList<nixlBlobDesc>::serialize<>(nixlSerDes *) const;
template void nixlDescList<nixlSectionDesc>::serialize<>(nixlSerDes *) const;

template bool MODULAR_CXX_EXPORT operator==
    <nixlBasicDesc>(const nixlDescList<nixlBasicDesc> &lhs,
                    const nixlDescList<nixlBasicDesc> &rhs);
template bool operator== <nixlMetaDesc>(const nixlDescList<nixlMetaDesc> &lhs,
                                        const nixlDescList<nixlMetaDesc> &rhs);
template bool MODULAR_CXX_EXPORT operator==
    <nixlBlobDesc>(const nixlDescList<nixlBlobDesc> &lhs,
                   const nixlDescList<nixlBlobDesc> &rhs);
template bool operator==
    <nixlSectionDesc>(const nixlDescList<nixlSectionDesc> &lhs,
                      const nixlDescList<nixlSectionDesc> &rhs);

// nixlSecDescList keeps the elements sorted
void nixlSecDescList::addDesc(const nixlSectionDesc &desc) {
  auto &vec = this->descs;
  auto itr = std::upper_bound(vec.begin(), vec.end(), desc);
  if (itr == vec.end())
    vec.push_back(desc);
  else
    vec.insert(itr, desc);
}

bool nixlSecDescList::verifySorted() const {
  const auto &vec = this->descs;
  int size = (int)vec.size();
  if (size <= 1)
    return (size == 1);
  for (int i = 0; i < size - 1; ++i) {
    if (vec[i + 1] < vec[i])
      return false;
  }
  return true;
}

nixlSectionDesc &nixlSecDescList::operator[](size_t index) {
  nixlSectionDesc &ref = this->descs[index];
  assert(verifySorted());
  return ref;
}

int nixlSecDescList::getIndex(const nixlBasicDesc &query) const {
  auto itr = std::lower_bound(this->descs.begin(), this->descs.end(), query);
  if (itr == this->descs.end())
    return to_underlying(nixl_status_t::NIXL_ERR_NOT_FOUND);
  if (static_cast<const nixlBasicDesc &>(*itr) == query)
    return static_cast<int>(itr - this->descs.begin());
  return to_underlying(nixl_status_t::NIXL_ERR_NOT_FOUND);
}

int nixlSecDescList::getCoveringIndex(const nixlBasicDesc &query) const {
  auto itr = std::lower_bound(this->descs.begin(), this->descs.end(), query);
  if (itr != this->descs.end() && itr->covers(query))
    return static_cast<int>(itr - this->descs.begin());
  // If query and element don't have the same start address, try previous entry
  if (itr != this->descs.begin()) {
    auto prev_itr = std::prev(itr, 1);
    if (prev_itr->covers(query))
      return static_cast<int>(prev_itr - this->descs.begin());
  }
  return -1;
}

void nixlSecDescList::resize(const size_t &count) {
  if (count > this->descs.size()) {
    // TODO(GENAI-170): raise ErrorOr.
    NIXL_FATAL << "nixlSecDescList: to keep list sorted, resize growth is not "
                  "allowed.";
    std::abort();
  }
  this->descs.resize(count);
}

} // namespace M::AsyncRT::NIXL
