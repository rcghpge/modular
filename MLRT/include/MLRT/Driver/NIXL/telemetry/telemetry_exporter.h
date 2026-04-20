//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EXPORTER_H
#define MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EXPORTER_H

#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"

#include <cstddef>
#include <string>

namespace M::MLRT {

inline constexpr char telemetryExporterVar[] =
    "MODULAR_NIXL_TELEMETRY_EXPORTER";

struct nixlTelemetryExporterInitParams {
  std::string agentName;
  size_t maxEventsBuffered;
};

class nixlTelemetryExporter {
public:
  explicit nixlTelemetryExporter(
      const nixlTelemetryExporterInitParams &initParams) noexcept
      : maxEventsBuffered_(initParams.maxEventsBuffered) {}
  nixlTelemetryExporter(nixlTelemetryExporter &&) = delete;
  nixlTelemetryExporter(const nixlTelemetryExporter &) = delete;

  void operator=(nixlTelemetryExporter &&) = delete;
  void operator=(const nixlTelemetryExporter &) = delete;

  virtual ~nixlTelemetryExporter() = default;

  [[nodiscard]] size_t getMaxEventsBuffered() const noexcept {
    return maxEventsBuffered_;
  }

  virtual nixl_status_t exportEvent(const nixlTelemetryEvent &event) = 0;

private:
  const size_t maxEventsBuffered_;
};

} // namespace M::MLRT

#endif // MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_EXPORTER_H
