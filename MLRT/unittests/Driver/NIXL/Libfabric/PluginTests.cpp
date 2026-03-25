//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/plugin_manager.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"
#include "MLRT/unittests/Driver/NIXL/TestCommon.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <string>

using namespace M::AsyncRT::NIXL;

TEST(LibfabricPluginTests, LoadsLibfabricPluginFromRegisteredDirectory) {
  const char *pluginDir = std::getenv("MODULAR_NIXL_PLUGIN_DIR");
  ASSERT_NE(pluginDir, nullptr);

  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadPlugin("libfabric");
  pluginManager.addPluginDirectory(std::string(pluginDir));

  auto pluginHandle = pluginManager.loadPlugin("libfabric");
  ASSERT_NE(pluginHandle, nullptr);
  EXPECT_STREQ(pluginHandle->getName(), "LIBFABRIC");
}

TEST(LibfabricPluginTests, AgentCanCreateLibfabricBackend) {
  if (LibfabricUtils::getAvailableEfaDevices().empty()) {
    GTEST_SKIP() << "libfabric backend creation requires available EFA devices";
  }

  nixlAgent agent("LibfabricPluginTestAgent",
                  nixlAgentConfig(/*use_prog_thread=*/true,
                                  /*use_listen_thread=*/false,
                                  /*port=*/0));

  nixl_mem_list_t mems;
  nixl_b_params_t params;
  ASSERT_SUCCESS(agent.getPluginParams("libfabric", mems, params));
  EXPECT_FALSE(mems.empty());

  nixlBackendH *backend = nullptr;
  ASSERT_SUCCESS(agent.createBackend("libfabric", params, backend));
  ASSERT_NE(backend, nullptr);
}
