//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/nixl.h"
#include "MLRT/Driver/NIXL/nixl_types.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace M::MLRT;

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
    } else {
      unsetenv(name_);
    }
  }

private:
  const char *name_;
  std::optional<std::string> oldValue_;
};

} // namespace

namespace {

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

class NixlIntegrationTest : public ::testing::Test {
protected:
  std::unique_ptr<nixlAgent> agent = nullptr;

  // Backend handle lifetime is managed by the agent.
  nixlBackendH *posixBackend = nullptr;
  nixl_opt_args_t registrationArgs;

  std::vector<char> dramBuffer;
  size_t bufferSize = 1024;

  std::unique_ptr<nixl_reg_dlist_t> regDlist = nullptr;
  std::string agentName = "TestAgent";

  void SetUp() override {
    // 1. Configure agent: disable threads for simplicity.
    // Use NIXL_THREAD_SYNC_NONE as we are not testing thread safety here.
    nixlAgentConfig agentConfig(/*use_prog_thread=*/false,
                                /*use_listen_thread=*/false, 0,
                                nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);

    // 2. Create agent.
    agent = std::make_unique<nixlAgent>(agentName, agentConfig);
    ASSERT_NE(agent, nullptr) << "Agent creation returned null";

    // 4. Memory Allocation.
    dramBuffer.resize(bufferSize);
    ASSERT_FALSE(dramBuffer.empty()) << "DRAM buffer allocation failed";

    // 5. Descriptor Creation (Using make_unique for the list)
    regDlist = std::make_unique<nixl_reg_dlist_t>(nixl_mem_t::DRAM_SEG);
    ASSERT_NE(regDlist, nullptr)
        << "Failed to create registration descriptor list";
  }

  void TearDown() override {
    nixl_status_t deregStatus = nixl_status_t::NIXL_SUCCESS;
    // Cleanup: deregister memory (best effort in TearDown)
    // Check if agent and reg_dlist were successfully created in SetUp.
    if (agent && regDlist && !dramBuffer.empty() && !regDlist->isEmpty()) {
      deregStatus = agent->deregisterMem(*regDlist, &registrationArgs);
      // Don't ASSERT in TearDown, just log potential issues.
      if (deregStatus != nixl_status_t::NIXL_SUCCESS)
        std::cerr << "[ WARNING ] Failed to deregister memory in TearDown: "
                  << NixlEnumStrings::statusStr(deregStatus) << std::endl;
    }
  }
};

// NIXL expects at least one backend to handle DRAM_SEG, so every operation
// below must return an error code – but it must be the *right* error code
// (no segfault, no UB, no silent success).
TEST_F(NixlIntegrationTest, NoBackendOperationsFailGracefully) {
  // Create descriptor for DRAM segment and add it to the descriptor list.
  nixlBlobDesc desc;
  desc.addr = reinterpret_cast<uintptr_t>(dramBuffer.data());
  desc.len = dramBuffer.size();
  // Use device ID 0 for DRAM.
  desc.devId = 0;
  // No backend-specific metadata needed for this basic test.
  desc.metaInfo = "";
  regDlist->addDesc(desc);

  ASSERT_EQ(regDlist->descCount(), 1)
      << "Descriptor list should contain 1 descriptor";

  // 1. With an empty agent we should see no plugins/backends.
  std::vector<nixl_backend_t> names;
  EXPECT_EQ(agent->getAvailPlugins(names), nixl_status_t::NIXL_SUCCESS);

  // 2. A direct attempt to register memory must fail: the helper inside
  //    `nixlAgent::registerMem()` will find `memToBackend[DRAM_SEG]` empty and
  //    return NIXL_ERR_NOT_FOUND.
  nixl_status_t st = agent->registerMem(*regDlist, &registrationArgs);
  EXPECT_EQ(st, nixl_status_t::NIXL_ERR_NOT_FOUND);

  // 3. Even if we try to *create* a backend with a bogus name, the plugin
  //    manager won’t find a .so and must surface NIXL_ERR_NOT_FOUND.
  nixl_b_params_t params;
  nixlBackendH *bogusHandle = nullptr;
  st = agent->createBackend("I_DO_NOT_EXIST", params, bogusHandle);
  EXPECT_EQ(st, nixl_status_t::NIXL_ERR_NOT_FOUND);
  EXPECT_EQ(bogusHandle, nullptr);

  // 4. Finally, posting or checking an empty handle should be a hard error.
  st = agent->postXferReq(nullptr);
  EXPECT_EQ(st, nixl_status_t::NIXL_ERR_INVALID_PARAM);

  // No resources were ever successfully created, so TearDown() will simply
  // attempt to deregister the DRAM descriptor and get the same NOT_FOUND
  // (logged, not fatal).
}

TEST_F(NixlIntegrationTest, NoBackendFailureRecordsTelemetryError) {
  namespace fs = std::filesystem;

  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_agent_errors_" + std::to_string(getpid()));
  fs::create_directories(telemetryDir);
  ScopedEnvVar telemetryEnabled("MODULAR_NIXL_TELEMETRY_ENABLE", "y");
  ScopedEnvVar telemetryDirVar("MODULAR_NIXL_TELEMETRY_DIR",
                               telemetryDir.string());
  ScopedEnvVar telemetryRunInterval(TELEMETRY_RUN_INTERVAL_VAR, "1");

  nixlAgentConfig telemetryConfig(/*use_prog_thread=*/false,
                                  /*use_listen_thread=*/false, 0,
                                  nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE);
  auto telemetryAgent =
      std::make_unique<nixlAgent>("TelemetryAgent", telemetryConfig);

  nixl_reg_dlist_t descs(nixl_mem_t::DRAM_SEG);
  descs.addDesc({reinterpret_cast<uintptr_t>(dramBuffer.data()),
                 dramBuffer.size(), 0, ""});

  EXPECT_EQ(telemetryAgent->registerMem(descs, &registrationArgs),
            nixl_status_t::NIXL_ERR_NOT_FOUND);

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (telemetryDir / "TelemetryAgent").string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(waitUntil([&]() { return !buffer->empty(); }));

  nixlTelemetryEvent event;
  ASSERT_TRUE(buffer->pop(event));
  EXPECT_STREQ(event.eventName_,
               NixlEnumStrings::statusStr(nixl_status_t::NIXL_ERR_NOT_FOUND));
  EXPECT_EQ(event.category_, nixl_telemetry_category_t::NIXL_TELEMETRY_ERROR);

  fs::remove_all(telemetryDir);
}
