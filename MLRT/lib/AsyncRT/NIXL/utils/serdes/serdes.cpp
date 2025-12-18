//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/AsyncRT/NIXL/utils/serdes/serdes.h"
#include "MLRT/AsyncRT/NIXL/nixl_types.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

using namespace M::AsyncRT::NIXL;

std::string nixlSerDes::_bytesToString(const void *buf, size_t size) {
  return std::string(reinterpret_cast<const char *>(buf), size);
}

void nixlSerDes::_stringToBytes(void *fill_buf, const std::string &s,
                                size_t size) {
  s.copy(reinterpret_cast<char *>(fill_buf), size);
}

/* Ser/Des for Strings */
void nixlSerDes::addStr(const std::string_view tag,
                        const std::string_view str) {
  const size_t len = str.size();

  workingStr.append(tag);
  workingStr.append(reinterpret_cast<const char *>(&len), sizeof(size_t));
  workingStr.append(str);
  workingStr.append("|");
}

std::string nixlSerDes::getStr(const std::string_view tag) {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return "";
  }
  size_t len;

  // skip tag
  des_offset += tag.size();

  // get len
  //_stringToBytes(&len, workingStr.data() + des_offset, sizeof(size_t));
  _stringToBytes(&len, workingStr.substr(des_offset, sizeof(size_t)),
                 sizeof(size_t));
  des_offset += sizeof(size_t);

  // get string
  std::string ret = workingStr.substr(des_offset, len);

  // move past string plus | delimiter
  des_offset += len + 1;

  return ret;
}

/* Ser/Des for Byte buffers */
void nixlSerDes::addBuf(std::string_view tag, const void *buf, size_t len) {
  workingStr.append(tag);
  workingStr.append(reinterpret_cast<const char *>(&len), sizeof(size_t));
  workingStr.append(reinterpret_cast<const char *>(buf), len);
  workingStr.append("|");
}

size_t nixlSerDes::getBufLen(std::string_view tag) const {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return -1;
  }

  size_t len;

  // get len
  //_stringToBytes(&len, workingStr.data() + des_offset + tag.size(),
  // sizeof(size_t));
  _stringToBytes(&len,
                 workingStr.substr(des_offset + tag.size(), sizeof(size_t)),
                 sizeof(size_t));

  return len;
}

nixl_status_t nixlSerDes::getBuf(std::string_view tag, void *buf, size_t len) {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return nixl_status_t::NIXL_ERR_MISMATCH;
  }

  // skip over tag and size, which we assume has been read previously
  des_offset += tag.size() + sizeof(size_t);

  //_stringToBytes(buf, workingStr.data() + des_offset, len);
  _stringToBytes(buf, workingStr.substr(des_offset, len), len);

  // bytes in string form are twice as long, skip those plus | delimiter
  des_offset += len + 1;

  return nixl_status_t::NIXL_SUCCESS;
}

/* Ser/Des buffer management */
std::string nixlSerDes::exportStr() const { return workingStr; }

nixl_status_t nixlSerDes::importBlob(const blob_ref sdbuf) {
  // Ensure the blob has the expected starting tag
  if (!StringRef(reinterpret_cast<const char *>(sdbuf.data), sdbuf.size)
           .starts_with(kWireMagic))
    return nixl_status_t::NIXL_ERR_MISMATCH;

  workingStr =
      std::string(reinterpret_cast<const char *>(sdbuf.data), sdbuf.size);
  mode = ser_mode_t::DESERIALIZE;
  des_offset = kWireMagic.size();

  return nixl_status_t::NIXL_SUCCESS;
}
