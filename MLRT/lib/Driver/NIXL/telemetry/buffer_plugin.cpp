//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "MLRT/Driver/NIXL/telemetry/buffer_exporter.h"

#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_plugin.h"

namespace M::MLRT {

using buffer_exporter_plugin_t =
    nixlTelemetryPluginCreator<nixlTelemetryBufferExporter>;

nixlTelemetryPlugin *createStaticBUFFERPlugin() {
  return buffer_exporter_plugin_t::create(nixlTelemetryPluginApiVersionV1,
                                          "buffer", "1.0.0");
}

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *
nixl_telemetry_plugin_init() {
  return createStaticBUFFERPlugin();
}

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT void nixl_telemetry_plugin_fini() {}

} // namespace M::MLRT
