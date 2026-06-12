//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/unittests/Driver/NIXL/TestCommon.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace M;
using namespace M::Driver;
using namespace M::MLRT;
using namespace M::MLRT::Test;

TEST_F(NixlUcxIntegrationTest, QueryMemNotSupportedForUcx) {
  NixlAgentWithUCXBackend nixl;
  NixlAgentWithUCXBackend::create(nixl, "TestAgent_QueryMem", available_port(),
                                  "ucx");

  std::vector<char> dramBuffer(1024);
  nixl_reg_dlist_t descs(nixl_mem_t::DRAM_SEG);
  descs.addDesc(blobDescfromVector(dramBuffer, 0));

  nixl_opt_args_t queryArgs;
  queryArgs.backends = {nixl.backend};

  std::vector<nixl_query_resp_t> resp;
  ASSERT_EQ(nixl.agent->queryMem(descs, resp, &queryArgs),
            nixl_status_t::NIXL_ERR_NOT_SUPPORTED);
  EXPECT_TRUE(resp.empty());
}

TEST_F(NixlUcxIntegrationTest, PrepGpuSignal) {
  auto gpuContextOrError = DeviceContext::create(Device::gpuAPI, /*id=*/0);
  if (gpuContextOrError.isError()) {
    GTEST_SKIP() << "Need a GPU device to test GPU signal setup";
  }
  DeviceContextRef gpuContext = gpuContextOrError.takeValue();

  NixlAgentWithUCXBackend nixl;
  NixlAgentWithUCXBackend::create(nixl, "TestAgent_GpuSignal", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/0);

  size_t gpuSignalSize = 0;
  nixl_status_t sizeStatus =
      nixl.agent->getGpuSignalSize(*nixl.backend, gpuSignalSize);
  if (sizeStatus != nixl_status_t::NIXL_SUCCESS) {
    EXPECT_EQ(sizeStatus, nixl_status_t::NIXL_ERR_BACKEND);
    GTEST_SKIP() << "UCX GPU device API support is unavailable";
  }
  ASSERT_GT(gpuSignalSize, 0u);

  ASSIGN_NO_ERR(DeviceBufferRef, signalBuffer,
                gpuContext->enqueueCreateBuffer(gpuSignalSize));
  EXPECT_NO_ERR(signalBuffer->context().synchronize());

  nixl_reg_dlist_t signalDescs(nixl_mem_t::VRAM_SEG);
  signalDescs.addDesc(nixlBlobDesc(
      reinterpret_cast<uintptr_t>(signalBuffer->ptr()), gpuSignalSize,
      /*devId=*/0, ""));

  nixl_opt_args_t regArgs;
  regArgs.backends = {nixl.backend};
  ASSERT_SUCCESS(nixl.agent->registerMem(signalDescs, &regArgs));

  EXPECT_EQ(nixl.agent->prepGpuSignal(signalDescs),
            nixl_status_t::NIXL_SUCCESS);

  ASSERT_SUCCESS(nixl.agent->deregisterMem(signalDescs, &regArgs));
}
