//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TELEMETRY_BUFFER_EXPORTER_H
#define MLRT_DRIVER_NIXL_TELEMETRY_BUFFER_EXPORTER_H

#include "MLRT/Driver/NIXL/telemetry/telemetry.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"

#include <filesystem>

namespace M::AsyncRT::NIXL {

class nixlTelemetryBufferExporter : public nixlTelemetryExporter {
public:
  explicit nixlTelemetryBufferExporter(
      const nixlTelemetryExporterInitParams &initParams);

  nixl_status_t exportEvent(const nixlTelemetryEvent &event) override;

private:
  std::filesystem::path filePath_;
  sharedRingBuffer<nixlTelemetryEvent> buffer_;
};

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_TELEMETRY_BUFFER_EXPORTER_H
