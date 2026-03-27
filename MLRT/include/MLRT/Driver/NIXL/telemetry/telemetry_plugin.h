//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Adapted by Modular.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_PLUGIN_H
#define MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_PLUGIN_H

#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"

#include <memory>
#include <string>
#include <string_view>

namespace M::AsyncRT::NIXL {

enum class nixl_telemetry_plugin_api_version : unsigned int { V1 = 1 };

inline constexpr nixl_telemetry_plugin_api_version
    nixlTelemetryPluginApiVersionV1 = nixl_telemetry_plugin_api_version::V1;

using exporter_creator_fn_t = std::unique_ptr<nixlTelemetryExporter> (*)(
    const nixlTelemetryExporterInitParams &initParams);

class nixlTelemetryPlugin {
public:
  nixl_telemetry_plugin_api_version api_version;
  exporter_creator_fn_t create_exporter;

  nixlTelemetryPlugin(nixl_telemetry_plugin_api_version version,
                      std::string_view name, std::string_view ver,
                      exporter_creator_fn_t create) noexcept
      : api_version(version), create_exporter(create), name_(name),
        version_(ver) {}

  const std::string &getName() const noexcept { return name_; }
  const std::string &getVersion() const noexcept { return version_; }

private:
  std::string name_;
  std::string version_;
};

#define NIXL_TELEMETRY_PLUGIN_EXPORT __attribute__((visibility("default")))

template <typename exporterType>
class nixlTelemetryPluginCreator {
public:
  static nixlTelemetryPlugin *
  create(nixl_telemetry_plugin_api_version apiVersion, std::string_view name,
         std::string_view version) {
    static nixlTelemetryPlugin pluginInstance(apiVersion, name, version,
                                              createExporter);
    return &pluginInstance;
  }

private:
  static std::unique_ptr<nixlTelemetryExporter>
  createExporter(const nixlTelemetryExporterInitParams &initParams) {
    return std::make_unique<exporterType>(initParams);
  }
};

using nixlTelemetryStaticPluginCreatorFunc = nixlTelemetryPlugin *(*)();

extern "C" {
NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *nixl_telemetry_plugin_init();
NIXL_TELEMETRY_PLUGIN_EXPORT void nixl_telemetry_plugin_fini();
}

} // namespace M::AsyncRT::NIXL

#endif // MLRT_DRIVER_NIXL_TELEMETRY_TELEMETRY_PLUGIN_H
