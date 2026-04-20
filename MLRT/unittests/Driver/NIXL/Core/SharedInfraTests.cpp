//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/nixl.h"
#include "MLRT/Driver/NIXL/plugin_manager.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <string>

using namespace M::MLRT;

TEST(NixlSharedInfraTest,
     PluginManagerSupportsExplicitSearchPathsWithoutEnvironmentVariable) {
  const char *pluginDir = std::getenv("MODULAR_NIXL_PLUGIN_DIR");
  ASSERT_NE(pluginDir, nullptr);

  std::string pluginDirPath = pluginDir;
  unsetenv("MODULAR_NIXL_PLUGIN_DIR");

  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadBackendPlugin("ucx_cpu");
  pluginManager.addPluginDirectory(pluginDirPath);

  auto pluginHandle = pluginManager.loadBackendPlugin("ucx_cpu");
  ASSERT_NE(pluginHandle, nullptr);
  EXPECT_STREQ(pluginHandle->getName(), "UCX");
}

TEST(NixlSharedInfraTest, AgentExposesEstimateXferCostApi) {
  nixlAgent agent("SharedInfraRedAgent",
                  nixlAgentConfig(false, false, 0,
                                  nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE));

  std::chrono::microseconds duration(0);
  std::chrono::microseconds errorMargin(0);
  nixl_cost_t method = nixl_cost_t::ANALYTICAL_BACKEND;

  EXPECT_EQ(agent.estimateXferCost(nullptr, duration, errorMargin, method),
            nixl_status_t::NIXL_ERR_INVALID_PARAM);
}
