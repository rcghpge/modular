//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/backend/backend_engine.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace M::MLRT {
namespace {

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
      return;
    }
    unsetenv(name_);
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

class TelemetryCoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    testDir_ = std::filesystem::temp_directory_path() /
               ("nixl_telemetry_core_" + std::to_string(getpid()));
    std::filesystem::create_directories(testDir_);
  }

  void TearDown() override { std::filesystem::remove_all(testDir_); }

  std::filesystem::path testDir_;
};

class TestBackendEngine : public nixlBackendEngine {
public:
  explicit TestBackendEngine(const nixlBackendInitParams *params)
      : nixlBackendEngine(params) {}

  void publishEvent(const std::string &name, uint64_t value) {
    addTelemetryEvent(name, value);
  }

  bool supportsRemote() const override { return false; }
  bool supportsLocal() const override { return false; }
  bool supportsNotif() const override { return false; }
  bool supportsProgTh() const override { return false; }
  nixl_mem_list_t getSupportedMems() const override { return {}; }
  nixl_status_t registerMem(const nixlBlobDesc &, const nixl_mem_t &,
                            nixlBackendMD *&) override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t deregisterMem(nixlBackendMD *) override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t connect(const std::string &) override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t disconnect(const std::string &) override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t unloadMD(nixlBackendMD *) override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t prepXfer(const nixl_xfer_op_t &, const nixl_meta_dlist_t &,
                         const nixl_meta_dlist_t &, const std::string &,
                         nixlBackendReqH *&,
                         const nixl_opt_b_args_t *) const override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t postXfer(const nixl_xfer_op_t &, const nixl_meta_dlist_t &,
                         const nixl_meta_dlist_t &, const std::string &,
                         nixlBackendReqH *&,
                         const nixl_opt_b_args_t *) const override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t checkXfer(nixlBackendReqH *) const override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
  nixl_status_t releaseReqH(nixlBackendReqH *) const override {
    return nixl_status_t::NIXL_ERR_NOT_SUPPORTED;
  }
};

TEST_F(TelemetryCoreTest, CreatesSharedBufferWhenTelemetryEnabledByEnv) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  const std::string telemetryFile = "telemetry_agent";
  nixlTelemetry telemetry(telemetryFile);

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / telemetryFile).string(), /*create=*/false, TELEMETRY_VERSION);

  EXPECT_EQ(buffer->version(), TELEMETRY_VERSION);
  EXPECT_TRUE(buffer->empty());
}

TEST_F(TelemetryCoreTest, RecordsTransferAndErrorEvents) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixlTelemetry telemetry("telemetry_agent");
  telemetry.updateTxBytes(1024);
  telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND);

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / "telemetry_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() { return buffer->size() >= 2; }));

  nixlTelemetryEvent firstEvent;
  nixlTelemetryEvent secondEvent;
  ASSERT_TRUE(buffer->pop(firstEvent));
  ASSERT_TRUE(buffer->pop(secondEvent));

  EXPECT_STREQ(firstEvent.eventName_, "agent_tx_bytes");
  EXPECT_EQ(firstEvent.value_, 1024u);
  EXPECT_EQ(firstEvent.category_,
            nixl_telemetry_category_t::NIXL_TELEMETRY_TRANSFER);

  EXPECT_STREQ(secondEvent.eventName_,
               NixlEnumStrings::statusStr(nixl_status_t::NIXL_ERR_BACKEND));
  EXPECT_EQ(secondEvent.category_,
            nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR);
}

TEST_F(TelemetryCoreTest, SupportsCustomBufferSize) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryBufferSize(TELEMETRY_BUFFER_SIZE_VAR, "32");
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixlTelemetry telemetry("telemetry_agent");

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / "telemetry_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  EXPECT_EQ(buffer->capacity(), 32u);
}

TEST_F(TelemetryCoreTest, TracksTransferBytesAndCounts) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixlTelemetry telemetry("telemetry_agent");
  telemetry.updateTxBytes(1024);
  telemetry.updateRxBytes(1024);
  telemetry.updateTxRequestsNum(1);
  telemetry.updateRxRequestsNum(1);
  telemetry.updateErrorCount(nixl_status_t::NIXL_ERR_BACKEND);
  telemetry.updateMemoryRegistered(1024);
  telemetry.updateMemoryDeregistered(1024);
  telemetry.addXferTime(std::chrono::microseconds(100), true, 2000);

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / "telemetry_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() { return buffer->size() >= 10; }));
  EXPECT_EQ(buffer->size(), 10u);
}

TEST_F(TelemetryCoreTest, ConvertsTelemetryCategoriesToStrings) {
  for (int i = 0;
       i <= static_cast<int>(nixl_telemetry_category_t::NIXL_TELEMETRY_CUSTOM);
       ++i) {
    auto category = static_cast<nixl_telemetry_category_t>(i);
    EXPECT_NE(NixlEnumStrings::telemetryCategoryStr(category), "BAD_CATEGORY");
  }

  auto invalidCategory = static_cast<nixl_telemetry_category_t>(999);
  EXPECT_EQ(NixlEnumStrings::telemetryCategoryStr(invalidCategory),
            "BAD_CATEGORY");
}

TEST_F(TelemetryCoreTest, HandlesConcurrentAccess) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixlTelemetry telemetry("telemetry_agent");
  std::vector<std::thread> threads;
  constexpr int kThreads = 4;
  constexpr int kOperationsPerThread = 100;

  for (int threadIndex = 0; threadIndex < kThreads; ++threadIndex) {
    threads.emplace_back([&telemetry, threadIndex]() {
      for (int opIndex = 0; opIndex < kOperationsPerThread; ++opIndex) {
        switch (threadIndex % 4) {
        case 0:
          telemetry.updateTxBytes(opIndex * 100);
          break;
        case 1:
          telemetry.updateRxBytes(opIndex * 50);
          break;
        case 2:
          telemetry.updateTxRequestsNum(opIndex);
          break;
        case 3:
          telemetry.updateRxRequestsNum(opIndex);
          break;
        }
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / "telemetry_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() {
    return buffer->size() >=
           static_cast<size_t>(kThreads * kOperationsPerThread);
  }));
  EXPECT_EQ(buffer->size(),
            static_cast<size_t>(kThreads * kOperationsPerThread));
}

TEST_F(TelemetryCoreTest, QueuesBackendTelemetryEvents) {
  nixl_b_params_t params;
  nixlBackendInitParams initParams;
  initParams.localAgent = "telemetry_agent";
  initParams.type = "TEST";
  initParams.customParams = &params;
  initParams.enableTelemetry_ = true;

  TestBackendEngine backend(&initParams);
  backend.publishEvent("backend_event", 7);

  auto events = backend.getTelemetryEvents();
  ASSERT_EQ(events.size(), 1u);
  EXPECT_STREQ(events[0].eventName_, "backend_event");
  EXPECT_EQ(events[0].value_, 7u);
  EXPECT_EQ(events[0].category_,
            nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);
  EXPECT_TRUE(backend.getTelemetryEvents().empty());
}

TEST_F(TelemetryCoreTest, WritesQueuedBackendTelemetryEventsToSharedBuffer) {
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDir("MODULAR_NIXL_TELEMETRY_DIR", testDir_.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixl_b_params_t params;
  nixlBackendInitParams initParams;
  initParams.localAgent = "telemetry_agent";
  initParams.type = "TEST";
  initParams.customParams = &params;
  initParams.enableTelemetry_ = true;

  TestBackendEngine backend(&initParams);
  backend.publishEvent("backend_event", 11);

  nixlTelemetry telemetry("telemetry_agent");
  telemetry.recordBackendEvents(backend.getTelemetryEvents());

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (testDir_ / "telemetry_agent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() { return !buffer->empty(); }));
  nixlTelemetryEvent event;
  ASSERT_TRUE(buffer->pop(event));
  EXPECT_STREQ(event.eventName_, "backend_event");
  EXPECT_EQ(event.value_, 11u);
  EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_BACKEND);
}

} // namespace
} // namespace M::MLRT
