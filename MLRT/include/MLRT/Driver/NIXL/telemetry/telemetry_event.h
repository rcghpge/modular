//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EVENT_H
#define MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EVENT_H

#include <cstdint>
#include <cstring>
#include <string>

namespace M::AsyncRT::NIXL {

inline constexpr char TELEMETRY_BUFFER_SIZE_VAR[] =
    "MODULAR_NIXL_TELEMETRY_BUFFER_SIZE";
inline constexpr char TELEMETRY_RUN_INTERVAL_VAR[] =
    "MODULAR_NIXL_TELEMETRY_RUN_INTERVAL";

inline constexpr int TELEMETRY_VERSION = 1;
inline constexpr size_t MAX_EVENT_NAME_LEN = 32;

enum class nixl_telemetry_category_t {
  NIXL_TELEMETRY_MEMORY = 0,
  NIXL_TELEMETRY_TRANSFER = 1,
  NIXL_TELEMETRY_CONNECTION = 2,
  NIXL_TELEMETRY_BACKEND = 3,
  NIXL_TELEMETRY_ERROR = 4,
  NIXL_TELEMETRY_PERFORMANCE = 5,
  NIXL_TELEMETRY_SYSTEM = 6,
  NIXL_TELEMETRY_CUSTOM = 7,
};

namespace NixlEnumStrings {
std::string telemetryCategoryStr(nixl_telemetry_category_t category);
} // namespace NixlEnumStrings

struct nixlTelemetryEvent {
  uint64_t timestampUs_ = 0;
  nixl_telemetry_category_t category_ =
      nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM;
  char eventName_[MAX_EVENT_NAME_LEN] = {};
  uint64_t value_ = 0;

  nixlTelemetryEvent() = default;

  nixlTelemetryEvent(uint64_t timestampUs, nixl_telemetry_category_t category,
                     const char *eventName, uint64_t value) noexcept
      : timestampUs_(timestampUs), category_(category), value_(value) {
    std::strncpy(eventName_, eventName, MAX_EVENT_NAME_LEN - 1);
    eventName_[MAX_EVENT_NAME_LEN - 1] = '\0';
  }

  nixlTelemetryEvent(uint64_t timestampUs, nixl_telemetry_category_t category,
                     const std::string &eventName, uint64_t value) noexcept
      : nixlTelemetryEvent(timestampUs, category, eventName.c_str(), value) {}
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EVENT_H
