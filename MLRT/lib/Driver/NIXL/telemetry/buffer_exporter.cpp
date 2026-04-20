//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/telemetry/buffer_exporter.h"

#include "MLRT/Driver/NIXL/utils/common/nixl_log.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>

namespace M::MLRT {
namespace {

[[noreturn]] void bufferExporterFatal(const char *message) {
  llvm::report_fatal_error(message);
}

std::filesystem::path
getFilePath(const nixlTelemetryExporterInitParams &initParams) {
  const char *telemetryDir = std::getenv(TELEMETRY_DIR_VAR);
  if (!telemetryDir || *telemetryDir == '\0') {
    bufferExporterFatal(
        "MODULAR_NIXL_TELEMETRY_DIR must be set for buffer exporter");
  }
  return std::filesystem::path(telemetryDir) / initParams.agentName;
}

} // namespace

nixlTelemetryBufferExporter::nixlTelemetryBufferExporter(
    const nixlTelemetryExporterInitParams &initParams)
    : nixlTelemetryExporter(initParams), filePath_(getFilePath(initParams)),
      buffer_(filePath_.string(), /*create=*/true, TELEMETRY_VERSION,
              getMaxEventsBuffered()) {
  NIXL_INFO << "Telemetry enabled, using buffer path: " << filePath_.string()
            << " with size: " << getMaxEventsBuffered();
}

nixl_status_t
nixlTelemetryBufferExporter::exportEvent(const nixlTelemetryEvent &event) {
  if (!buffer_.push(event)) {
    return nixl_status_t::NIXL_ERR_UNKNOWN;
  }

  return nixl_status_t::NIXL_SUCCESS;
}

} // namespace M::MLRT
