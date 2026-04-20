//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/NIXL/plugin_manager.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"
#include "MLRT/unittests/Driver/NIXL/TestCommon.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace M::MLRT;

namespace {

bool containsMemType(const nixl_mem_list_t &mems, nixl_mem_t memType) {
  return std::find(mems.begin(), mems.end(), memType) != mems.end();
}

std::shared_ptr<const nixlBackendPluginHandle> loadLibfabricPluginForTest() {
  const char *pluginDir = std::getenv("MODULAR_NIXL_PLUGIN_DIR");
  EXPECT_NE(pluginDir, nullptr);

  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadBackendPlugin("libfabric");
  pluginManager.addPluginDirectory(std::string(pluginDir));
  return pluginManager.loadBackendPlugin("libfabric");
}

} // namespace

TEST(LibfabricPluginTests, LoadsLibfabricPluginFromRegisteredDirectory) {
  const char *pluginDir = std::getenv("MODULAR_NIXL_PLUGIN_DIR");
  ASSERT_NE(pluginDir, nullptr);

  auto &pluginManager = nixlPluginManager::getInstance();
  pluginManager.unloadBackendPlugin("libfabric");
  pluginManager.addPluginDirectory(std::string(pluginDir));

  auto pluginHandle = pluginManager.loadBackendPlugin("libfabric");
  ASSERT_NE(pluginHandle, nullptr);
  EXPECT_STREQ(pluginHandle->getName(), "LIBFABRIC");
}

TEST(LibfabricPluginTests, PluginMetadataIsConservative) {
  nixlAgent agent("LibfabricPluginMetadataTestAgent",
                  nixlAgentConfig(/*use_prog_thread=*/true,
                                  /*use_listen_thread=*/false,
                                  /*port=*/0));

  nixl_mem_list_t mems;
  nixl_b_params_t params;
  ASSERT_SUCCESS(agent.getPluginParams("libfabric", mems, params));

  EXPECT_EQ(mems.size(), 1u);
  EXPECT_TRUE(containsMemType(mems, DRAM_SEG));
  EXPECT_FALSE(containsMemType(mems, VRAM_SEG));
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

  nixl_mem_list_t runtimeMems;
  nixl_b_params_t runtimeParams;
  ASSERT_SUCCESS(agent.getBackendParams(backend, runtimeMems, runtimeParams));

  EXPECT_TRUE(containsMemType(runtimeMems, DRAM_SEG));
  bool expectVram =
      M::Driver::Device::numberOfDevices(M::Driver::Device::gpuAPI) > 0;
  EXPECT_EQ(containsMemType(runtimeMems, VRAM_SEG), expectVram);
}

TEST(LibfabricPluginTests, EngineRejectsMissingVramDeviceId) {
  if (LibfabricUtils::getAvailableEfaDevices().empty()) {
    GTEST_SKIP() << "libfabric backend creation requires available EFA devices";
  }

  auto pluginHandle = loadLibfabricPluginForTest();
  ASSERT_NE(pluginHandle, nullptr);

  nixl_b_params_t params;
  nixlBackendInitParams initParams{
      .localAgent = "LibfabricInvalidVramDeviceIdAgent",
      .type = "libfabric",
      .customParams = &params,
      .enableProgTh = true,
      .pthrDelay = nixlTime::us_t(0),
      .syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE,
  };

  nixlBackendEngine *engine = pluginHandle->createEngine(&initParams);
  ASSERT_NE(engine, nullptr);
  ASSERT_FALSE(engine->getInitErr());

  std::vector<uint8_t> buffer(256, 0);
  nixlBlobDesc desc(reinterpret_cast<uintptr_t>(buffer.data()), buffer.size(),
                    std::numeric_limits<uint64_t>::max());
  nixlBackendMD *metadata = nullptr;

  EXPECT_EQ(engine->registerMem(desc, VRAM_SEG, metadata),
            nixl_status_t::NIXL_ERR_INVALID_PARAM);
  EXPECT_EQ(metadata, nullptr);

  pluginHandle->destroyEngine(engine);
}

TEST(LibfabricPluginTests, AgentRegistersVramWithImplicitHmem) {
  if (LibfabricUtils::getAvailableEfaDevices().empty()) {
    GTEST_SKIP() << "libfabric backend creation requires available EFA devices";
  }

  auto gpuContextOrError =
      M::Driver::DeviceContext::create(M::Driver::Device::gpuAPI, /*id=*/0);
  if (gpuContextOrError.isError()) {
    GTEST_SKIP() << "Need a GPU device to test libfabric VRAM registration";
  }
  M::Driver::DeviceContextRef gpuContext = gpuContextOrError.takeValue();

  const size_t bufferSize = 64 * 1024;
  ASSIGN_NO_ERR(M::Driver::DeviceBufferRef, deviceBuffer,
                gpuContext->enqueueCreateBuffer(bufferSize));
  EXPECT_NO_ERR(
      gpuContext->enqueueSetMemory(*deviceBuffer, 0x5A, sizeof(char)));
  EXPECT_NO_ERR(gpuContext->synchronize());

  nixlAgent agent("LibfabricRegisterVramTestAgent",
                  nixlAgentConfig(/*use_prog_thread=*/true,
                                  /*use_listen_thread=*/false,
                                  /*port=*/0));

  nixl_mem_list_t mems;
  nixl_b_params_t params;
  ASSERT_SUCCESS(agent.getPluginParams("libfabric", mems, params));

  nixlBackendH *backend = nullptr;
  ASSERT_SUCCESS(agent.createBackend("libfabric", params, backend));
  ASSERT_NE(backend, nullptr);

  nixl_reg_dlist_t descs(nixl_mem_t::VRAM_SEG);
  descs.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(deviceBuffer->ptr()),
                             bufferSize, /*devId=*/0, ""));

  nixl_opt_args_t regArgs;
  regArgs.backends = {backend};
  ASSERT_SUCCESS(agent.registerMem(descs, &regArgs));
  ASSERT_SUCCESS(agent.deregisterMem(descs, &regArgs));
}
