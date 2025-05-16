//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNCRT_NIXL_UTILS_SERDES_SERDES_H
#define ASYNCRT_NIXL_UTILS_SERDES_SERDES_H

#include "Support/SymbolExport.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "AsyncRT/NIXL/nixl_types.h"

using namespace std::string_view_literals;

namespace M::AsyncRT::NIXL {

class MODULAR_CXX_EXPORT nixlSerDes {
private:
  enum class ser_mode_t { SERIALIZE, DESERIALIZE };

  static constexpr std::string_view kWireMagic = "NIXL_SERDES|"sv;

  std::string workingStr;
  ssize_t des_offset;
  ser_mode_t mode;

public:
  nixlSerDes()
      : workingStr(kWireMagic), des_offset{kWireMagic.size()},
        mode{ser_mode_t::SERIALIZE} {};

  /* Ser/Des for Strings */
  nixl_status_t addStr(const std::string &tag, const std::string &str);
  std::string getStr(const std::string &tag);

  /* Ser/Des for Byte buffers */
  nixl_status_t addBuf(std::string_view tag, const void *buf, ssize_t len);
  ssize_t getBufLen(std::string_view tag) const;
  nixl_status_t getBuf(std::string_view tag, void *buf, ssize_t len);

  /* Ser/Des buffer management */
  std::string exportStr() const;
  nixl_status_t importStr(const std::string &sdbuf);

  /// Imports a blob of data. Prefer this method if dealing
  /// with non-human readable strings.
  nixl_status_t importBlob(const blob_ref sdbuf);

  static std::string _bytesToString(const void *buf, ssize_t size);
  static void _stringToBytes(void *fill_buf, const std::string &s,
                             ssize_t size);
};

} // namespace M::AsyncRT::NIXL

#endif // ASYNCRT_NIXL_UTILS_SERDES_SERDES_H
