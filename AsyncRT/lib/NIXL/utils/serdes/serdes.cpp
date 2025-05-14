//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "AsyncRT/NIXL/utils/serdes/serdes.h"
#include "AsyncRT/NIXL/nixl_types.h"

using namespace M::AsyncRT::NIXL;

std::string nixlSerDes::_bytesToString(const void *buf, ssize_t size) {
  return std::string(reinterpret_cast<const char *>(buf), size);
}

void nixlSerDes::_stringToBytes(void *fill_buf, const std::string &s,
                                ssize_t size) {
  s.copy(reinterpret_cast<char *>(fill_buf), size);
}

/* Ser/Des for Strings */
nixl_status_t nixlSerDes::addStr(const std::string &tag,
                                 const std::string &str) {

  size_t len = str.size();

  workingStr.append(tag);
  workingStr.append(_bytesToString(&len, sizeof(size_t)));
  workingStr.append(str);
  workingStr.append("|");

  return nixl_status_t::NIXL_SUCCESS;
}

std::string nixlSerDes::getStr(const std::string &tag) {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return "";
  }
  ssize_t len;

  // skip tag
  des_offset += tag.size();

  // get len
  //_stringToBytes(&len, workingStr.data() + des_offset, sizeof(ssize_t));
  _stringToBytes(&len, workingStr.substr(des_offset, sizeof(ssize_t)),
                 sizeof(ssize_t));
  des_offset += sizeof(ssize_t);

  // get string
  std::string ret = workingStr.substr(des_offset, len);

  // move past string plus | delimiter
  des_offset += len + 1;

  return ret;
}

/* Ser/Des for Byte buffers */
nixl_status_t nixlSerDes::addBuf(std::string_view tag, const void *buf,
                                 ssize_t len) {
  workingStr.append(tag);
  workingStr.append(_bytesToString(&len, sizeof(ssize_t)));
  workingStr.append(_bytesToString(buf, len));
  workingStr.append("|");

  return nixl_status_t::NIXL_SUCCESS;
}

ssize_t nixlSerDes::getBufLen(std::string_view tag) const {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return -1;
  }

  ssize_t len;

  // get len
  //_stringToBytes(&len, workingStr.data() + des_offset + tag.size(),
  // sizeof(ssize_t));
  _stringToBytes(&len,
                 workingStr.substr(des_offset + tag.size(), sizeof(ssize_t)),
                 sizeof(ssize_t));

  return len;
}

nixl_status_t nixlSerDes::getBuf(std::string_view tag, void *buf, ssize_t len) {
  if (workingStr.compare(des_offset, tag.size(), tag) != 0) {
    // incorrect tag
    return nixl_status_t::NIXL_ERR_MISMATCH;
  }

  // skip over tag and size, which we assume has been read previously
  des_offset += tag.size() + sizeof(ssize_t);

  //_stringToBytes(buf, workingStr.data() + des_offset, len);
  _stringToBytes(buf, workingStr.substr(des_offset, len), len);

  // bytes in string form are twice as long, skip those plus | delimiter
  des_offset += len + 1;

  return nixl_status_t::NIXL_SUCCESS;
}

/* Ser/Des buffer management */
std::string nixlSerDes::exportStr() const { return workingStr; }

nixl_status_t nixlSerDes::importStr(const std::string &sdbuf) {
  if (sdbuf.compare(0, kWireMagic.size(), kWireMagic) != 0) {
    // incorrect tag
    return nixl_status_t::NIXL_ERR_MISMATCH;
  }

  workingStr = sdbuf;
  mode = ser_mode_t::DESERIALIZE;
  des_offset = kWireMagic.size();

  return nixl_status_t::NIXL_SUCCESS;
}
