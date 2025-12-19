//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_NIXL_DESCRIPTORS_H
#define ASYNCRT_NIXL_NIXL_DESCRIPTORS_H

#include "Support/LLVMForwardDecls.h"
#include "Support/SymbolExport.h"
#include "nixl_types.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <sys/types.h>
#include <vector>

namespace M::AsyncRT::NIXL {

/**
 * @class nixlBasicDesc
 * @brief A basic descriptor class, single contiguous memory/storage
 *        element, alongside supporting methods
 */
class MODULAR_CXX_EXPORT nixlBasicDesc {
public:
  /** @var Start of Buffer */
  uintptr_t addr;
  /** @var Buffer Length */
  size_t len;
  /** @var deviceID/blockID/fileID */
  uint64_t devId;

  /**
   * @brief Default constructor for nixlBasicDesc
   *      Initializes to invalid values.
   */
  nixlBasicDesc()
      : addr{0}, len{0}, devId{std::numeric_limits<uint64_t>::max()} {}
  /**
   * @brief Parametrized constructor for nixlBasicDesc
   *
   * @param addr  Start of buffer/block/offset-in-file
   * @param len   Length of buffer
   * @param devID deviceID/BlockID/fileID
   */
  nixlBasicDesc(uintptr_t addr, size_t len, uint64_t dev_id)
      : addr{addr}, len{len}, devId{dev_id} {}

  /**
   * @brief Deserializer constructor for nixlBasicDesc with
   *        serialized blob of another nixlBasicDesc
   *
   * @param str   Serialized Descriptor
   */
  nixlBasicDesc(const nixl_blob_t &str); // deserializer

  /**
   * @brief Operator overloading (<) to compare BasicDesc objects
   *        Comparison criteria is devID, then addr, then len
   */
  bool operator<(const nixlBasicDesc &desc) const;
  /**
   * @brief Operator overloading (==) to compare BasicDesc objects
   *
   * @param lhs   nixlBasicDesc object
   * @param rhs   nixlBasicDesc object
   *
   */
  friend bool operator==(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
  /**
   * @brief Operator overloading (!=) to compare BasicDesc objects
   *
   * @param lhs   nixlBasicDesc object
   * @param rhs   nixlBasicDesc object
   *
   */
  friend bool operator!=(const nixlBasicDesc &lhs, const nixlBasicDesc &rhs);
  /**
   * @brief Check if current object address range covers the input object's
   *
   * @param query   nixlBasicDesc object
   */
  bool covers(const nixlBasicDesc &query) const;
  /**
   * @brief Check for overlap between BasicDesc objects
   *
   * @param query   nixlBasicDesc Object
   */
  bool overlaps(const nixlBasicDesc &query) const;
  /**
   * @brief Serialize descriptor into a blob
   */
  nixl_blob_t serialize() const;
  /**
   * @brief Print descriptor for debugging
   *
   * @param suffix gets prepended to the descriptor print
   */
  void print(Twine suffix) const;
};

/**
 * @class nixlBlobDesc
 * @brief A descriptor class, with additional metadata in form of a blob
 *        bundled with a nixlBasicDesc.
 */
class nixlBlobDesc : public nixlBasicDesc {
public:
  /** @var blob for metadata information */
  nixl_blob_t metaInfo;

  /** @var Reuse parent constructor without the metadata */
  using nixlBasicDesc::nixlBasicDesc;

  /**
   * @brief Parametrized constructor for nixlBlobDesc
   *
   * @param addr      Start of buffer/block/offset-in-file
   * @param len       Length of buffer
   * @param devID     deviceID/BlockID/fileID
   * @param meta_info Metadata blob
   */
  nixlBlobDesc(uintptr_t addr, size_t len, uint64_t dev_id,
               nixl_blob_t meta_info = "")
      : nixlBasicDesc(addr, len, dev_id), metaInfo(std::move(meta_info)) {}

  /**
   * @brief Constructor for nixlBlobDesc from nixlBasicDesc and metadata blob
   *
   * @param desc      nixlBasicDesc object
   * @param meta_info Metadata blob
   */
  nixlBlobDesc(nixlBasicDesc desc, nixl_blob_t meta_info)
      : nixlBasicDesc(std::move(desc)), metaInfo(std::move(meta_info)) {}

  /**
   * @brief Deserializer constructor for nixlBlobDesc with serialized blob
   *
   * @param str   Serialized blob from another nixlBlobDesc
   */
  nixlBlobDesc(const nixl_blob_t &str);
  /**
   * @brief Operator overloading (==) to compare nixlBlobDesc objects
   *
   * @param lhs   nixlBlobDesc object
   * @param rhs   nixlBlobDesc object
   */
  friend bool operator==(const nixlBlobDesc &lhs, const nixlBlobDesc &rhs);
  /**
   * @brief Operator overloading (!=) to compare BlobDesc objects
   *
   * @param lhs   nixlBlobDesc object
   * @param rhs   nixlBlobDesc object
   *
   */
  friend bool operator!=(const nixlBlobDesc &lhs, const nixlBlobDesc &rhs);
  /**
   * @brief Serialize nixlBlobDesc to a blob
   */
  nixl_blob_t serialize() const;
  /**
   * @brief Print nixlBlobDesc for debugging purpose
   *
   * @param suffix gets prepended to the descriptor print
   */
  void print(std::string_view suffix) const;
};

/**
 * @class nixlDescList
 * @brief A class for describing a list of descriptors, as a template based on
 *        the nixlDesc type that is used.
 */
template <class T>
class MODULAR_CXX_EXPORT nixlDescList {
protected:
  /** @var NIXL memory type */
  nixl_mem_t type;
  /** @var Vector for storing nixlDescs */
  std::vector<T> descs;

public:
  /**
   * @brief Parametrized Constructor for nixlDescList
   *
   * @param type         NIXL memory type of descriptor list
   * @param init_size    initial size for descriptor list (default = 0)
   */
  nixlDescList(nixl_mem_t type, int init_size = 0) : type{type} {
    static_assert(std::is_base_of<nixlBasicDesc, T>::value);
    descs.resize(init_size);
  }

  /**
   * @brief Deserializer constructor for nixlDescList from nixlSerDes object
   *        which serializes/deserializes our classes into/from blobs
   *
   * @param deserialize nixlSerDes object to construct nixlDescList
   */
  nixlDescList(nixlSerDes *deserializer);

  /**
   * @brief nixlDescList Destructor
   */
  virtual ~nixlDescList() = default;

  /**
   * @brief      Get NIXL memory type for this DescList
   */
  nixl_mem_t getType() const { return type; }

  /**
   * @brief       Get count of descriptors
   */
  int descCount() const { return descs.size(); }
  /**
   * @brief Check if nixlDescList is empty or not
   */
  bool isEmpty() const { return (descs.size() == 0); }
  /**
   * @brief Check if any two nixlDescs in the internal list of descriptors
   *        overlap with each other
   */
  bool hasOverlaps() const;
  /**
   * @brief Operator [] overloading, get/set descriptor at [index].
   *        Can throw std::out_of_range exception.
   */
  const T &operator[](size_t index) const {
    assert(index < descs.size());
    return descs[index];
  }
  virtual T &operator[](size_t index) {
    assert(index < descs.size());
    return descs[index];
  }

  /**
   * @brief Appends a descriptor to the end of the list
   */
  void push_back(T &&obj) { descs.push_back(std::move(obj)); }

  /**
   * @brief Vector like iterators for const and non-const elements
   */
  typename std::vector<T>::const_iterator begin() const {
    return descs.begin();
  }
  typename std::vector<T>::const_iterator end() const { return descs.end(); }
  typename std::vector<T>::iterator begin() { return descs.begin(); }
  typename std::vector<T>::iterator end() { return descs.end(); }
  /**
   * @brief Operator overloading (==) to compare nixlDescList objects
   *
   * @param lhs   nixlDescList object
   * @param rhs   nixlDescList object
   *
   */
  template <class Y>
  friend bool operator==(const nixlDescList<Y> &lhs,
                         const nixlDescList<Y> &rhs);
  /**
   * @brief Resize nixlDescList object.
   *
   * @param count Number of elements after resizing DescList object
   */
  virtual void resize(const size_t &count);

  /**
   * @brief Empty the descriptors list
   */
  void clear() { descs.clear(); }

  /**
   * @brief     Add Descriptors to descriptor list
   */
  virtual void addDesc(const T &desc);
  /**
   * @brief Remove descriptor from list at index
   *        Can throw std::out_of_range exception.
   */
  void remDesc(const int index);
  /**
   * @brief Convert a nixlDescList with metadata by trimming it to a
   *        nixlDescList of nixlBasicDesc elements
   */
  nixlDescList<nixlBasicDesc> trim() const;
  /**
   * @brief  Check if input descriptor `desc` overlaps with any descriptor
   *         within the current object, and returns its index if found.
   *
   * @param  index [out] index of overlapping descriptor
   */
  bool overlaps(const T &desc, int &index) const;
  /**
   * @brief  Get the index of a descriptor that matches the `query`
   *
   * @param  query nixlBasicDesc object to find among the object's descriptors
   * @return int   index of the queried nixlBasicDesc if found, or negative
   * error value
   */
  virtual int getIndex(const nixlBasicDesc &query) const;
  /**
   * @brief Serialize a descriptor list with nixlSerDes class
   * @param serializer nixlSerDes object to serialize nixlDescList
   * @return nixl_status_t Error code if serialize was not successful
   * @note This class is templated to allow it to be compiled independently of
   * the class template
   */
  template <class = void>
  void serialize(nixlSerDes *serializer) const;
  /**
   * @brief Print the descriptor list for debugging
   */
  void print() const;
};
/**
 * @brief A typedef for a nixlDescList<nixlBasicDesc>
 *        used for creating transfer descriptor lists
 */
using nixl_xfer_dlist_t = nixlDescList<nixlBasicDesc>;
/**
 * @brief A typedef for a nixlDescList<nixlBlobDesc>
 *        used for creating registration descriptor lists
 */
using nixl_reg_dlist_t = nixlDescList<nixlBlobDesc>;

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_NIXL_DESCRIPTORS_H
