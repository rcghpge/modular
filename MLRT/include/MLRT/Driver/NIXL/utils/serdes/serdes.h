//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_UTILS_SERDES_SERDES_H
#define MLRT_DRIVER_NIXL_UTILS_SERDES_SERDES_H

#include "Support/SymbolExport.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "MLRT/Driver/NIXL/nixl_types.h"

using namespace std::string_view_literals;

namespace M::AsyncRT::NIXL {

class MODULAR_CXX_EXPORT nixlSerDes {
private:
  enum class ser_mode_t { SERIALIZE, DESERIALIZE };

  static constexpr std::string_view kWireMagic = "NIXL_SERDES|"sv;

  std::string workingStr;
  size_t des_offset;
  ser_mode_t mode;

public:
  nixlSerDes()
      : workingStr(kWireMagic), des_offset{kWireMagic.size()},
        mode{ser_mode_t::SERIALIZE} {}

  /* Ser/Des for Strings */
  void addStr(std::string_view tag, std::string_view str);
  std::string getStr(std::string_view tag);
  /* Ser/Des for Byte buffers */
  void addBuf(std::string_view tag, const void *buf, size_t len);
  size_t getBufLen(std::string_view tag) const;
  nixl_status_t getBuf(std::string_view tag, void *buf, size_t len);

  /* Ser/Des buffer management */
  std::string exportStr() const;

  /// Imports a blob of data. Prefer this method if dealing
  /// with non-human readable strings.
  nixl_status_t importBlob(const blob_ref sdbuf);

  static std::string _bytesToString(const void *buf, size_t size);
  static void _stringToBytes(void *fill_buf, const std::string &s, size_t size);
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_UTILS_SERDES_SERDES_H
