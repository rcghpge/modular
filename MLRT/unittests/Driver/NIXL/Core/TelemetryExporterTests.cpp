//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/plugin_manager.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_exporter.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_plugin.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

using namespace M::MLRT;

namespace {

std::filesystem::path getTestTelemetryPluginDir() {
  const char *pluginPath = std::getenv("TEST_TELEMETRY_PLUGIN_PATH");
  EXPECT_NE(pluginPath, nullptr);
  return std::filesystem::path(pluginPath).parent_path();
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const std::string &value) : name_(name) {
    if (const char *existing = std::getenv(name_)) {
      oldValue_ = std::string(existing);
    }
    setenv(name_, value.c_str(), 1);
  }

  ~ScopedEnvVar() {
    if (oldValue_) {
      setenv(name_, oldValue_->c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

private:
  const char *name_;
  std::optional<std::string> oldValue_;
};

bool waitUntil(const std::function<bool()> &predicate,
               std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return predicate();
}

} // namespace

TEST(TelemetryExporterPluginTests, LoadsBuiltinBufferTelemetryPlugin) {
  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadTelemetryPlugin("BUFFER");

  auto pluginHandle = pluginManager.loadTelemetryPlugin("BUFFER");
  ASSERT_NE(pluginHandle, nullptr);
  EXPECT_STREQ(pluginHandle->getName(), "buffer");
}

TEST(TelemetryExporterPluginTests,
     LoadsTelemetryPluginFromRegisteredDirectory) {
  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadTelemetryPlugin("test");
  pluginManager.addPluginDirectory(getTestTelemetryPluginDir().string());

  auto pluginHandle = pluginManager.loadTelemetryPlugin("test");
  ASSERT_NE(pluginHandle, nullptr);
  EXPECT_STREQ(pluginHandle->getName(), "test");
  EXPECT_STREQ(pluginHandle->getVersion(), "1.0.0");

  nixlTelemetryExporterInitParams initParams{
      .agentName = "TelemetryExporterTestAgent",
      .maxEventsBuffered = 32,
  };
  auto exporter = pluginHandle->createExporter(initParams);
  ASSERT_NE(exporter, nullptr);

  nixlTelemetryEvent event(0, nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM,
                           "test_event", 1);
  EXPECT_EQ(exporter->exportEvent(event), nixl_status_t::NIXL_SUCCESS);
}

TEST(TelemetryExporterPluginTests,
     UsesExplicitTelemetryExporterSelectionAtRuntime) {
  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadTelemetryPlugin("test");
  pluginManager.addPluginDirectory(getTestTelemetryPluginDir().string());

  const auto telemetryDir =
      std::filesystem::temp_directory_path() / "nixl_telem_exporter_runtime";
  std::filesystem::remove_all(telemetryDir);
  std::filesystem::create_directories(telemetryDir);

  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDirVar("MODULAR_NIXL_TELEMETRY_DIR",
                               telemetryDir.string());
  ScopedEnvVar telemetryExporter("MODULAR_NIXL_TELEMETRY_EXPORTER", "test");
  ScopedEnvVar telemetryRunInterval("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1");

  nixlTelemetry telemetry("runtime_agent");
  telemetry.updateTxBytes(9);

  const auto exportedPath = telemetryDir / "runtime_agent.test_exporter";
  ASSERT_TRUE(waitUntil([&]() {
    std::ifstream in(exportedPath);
    std::string line;
    return in.is_open() && static_cast<bool>(std::getline(in, line));
  }));

  std::ifstream in(exportedPath);
  std::string line;
  ASSERT_TRUE(std::getline(in, line));
  EXPECT_EQ(line, "agent_tx_bytes,9");
}

TEST(TelemetryExporterPluginTests,
     FallsBackToBuiltinBufferExporterWhenOnlyDirectoryIsSet) {
  const auto telemetryDir =
      std::filesystem::temp_directory_path() / "nixl_telem_exporter_buffer";
  std::filesystem::remove_all(telemetryDir);
  std::filesystem::create_directories(telemetryDir);

  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDirVar("MODULAR_NIXL_TELEMETRY_DIR",
                               telemetryDir.string());
  ScopedEnvVar telemetryRunInterval("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1");

  nixlTelemetry telemetry("buffer_agent");
  telemetry.updateTxBytes(5);

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (telemetryDir / "buffer_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() { return !buffer->empty(); }));
  nixlTelemetryEvent event;
  ASSERT_TRUE(buffer->pop(event));
  EXPECT_STREQ(event.eventName_, "agent_tx_bytes");
  EXPECT_EQ(event.value_, 5u);
}

TEST(TelemetryExporterPluginTests, ExportsOnlyAfterRunIntervalExpires) {
  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadTelemetryPlugin("test");
  pluginManager.addPluginDirectory(getTestTelemetryPluginDir().string());

  const auto telemetryDir =
      std::filesystem::temp_directory_path() / "nixl_telem_exporter_interval";
  std::filesystem::remove_all(telemetryDir);
  std::filesystem::create_directories(telemetryDir);

  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDirVar("MODULAR_NIXL_TELEMETRY_DIR",
                               telemetryDir.string());
  ScopedEnvVar telemetryExporter("MODULAR_NIXL_TELEMETRY_EXPORTER", "test");
  ScopedEnvVar telemetryRunInterval("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL",
                                    "200");

  nixlTelemetry telemetry("interval_agent");
  telemetry.updateTxBytes(13);

  const auto exportedPath = telemetryDir / "interval_agent.test_exporter";
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  std::ifstream early(exportedPath);
  std::string earlyLine;
  EXPECT_FALSE(early.is_open() &&
               static_cast<bool>(std::getline(early, earlyLine)));

  ASSERT_TRUE(waitUntil(
      [&]() {
        std::ifstream in(exportedPath);
        std::string line;
        return in.is_open() && static_cast<bool>(std::getline(in, line));
      },
      std::chrono::milliseconds(600)));
}
