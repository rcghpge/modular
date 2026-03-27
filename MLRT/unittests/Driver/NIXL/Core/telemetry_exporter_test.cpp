//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_plugin.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace M::AsyncRT::NIXL {

class TestTelemetryExporter : public nixlTelemetryExporter {
public:
  explicit TestTelemetryExporter(
      const nixlTelemetryExporterInitParams &initParams)
      : nixlTelemetryExporter(initParams) {
    const char *telemetryDir = std::getenv("MODULAR_NIXL_TELEMETRY_DIR");
    if (telemetryDir && *telemetryDir != '\0') {
      filePath_ = std::filesystem::path(telemetryDir) /
                  (initParams.agentName + ".test_exporter");
    }
  }

  nixl_status_t exportEvent(const nixlTelemetryEvent &event) override {
    if (!filePath_.empty()) {
      std::ofstream out(filePath_, std::ios::app);
      out << event.eventName_ << "," << event.value_ << "\n";
    }
    return nixl_status_t::NIXL_SUCCESS;
  }

private:
  std::filesystem::path filePath_;
};

using test_exporter_plugin_t =
    nixlTelemetryPluginCreator<TestTelemetryExporter>;

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT nixlTelemetryPlugin *
nixl_telemetry_plugin_init() {
  return test_exporter_plugin_t::create(nixlTelemetryPluginApiVersionV1, "test",
                                        "1.0.0");
}

extern "C" NIXL_TELEMETRY_PLUGIN_EXPORT void nixl_telemetry_plugin_fini() {}

} // namespace M::AsyncRT::NIXL
