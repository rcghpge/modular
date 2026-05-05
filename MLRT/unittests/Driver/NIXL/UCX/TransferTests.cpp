//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/unittests/Driver/NIXL/TestCommon.h"

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/NIXL/telemetry/telemetry_event.h"
#include "MLRT/Driver/NIXL/utils/common/cyclic_buffer.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace M;
using namespace M::Driver;
using namespace M::MLRT;
using namespace M::MLRT::Test;

TEST_F(NixlUcxIntegrationTest, DramTransfer) {
  // Explicitly enable verbose logging for UCX.
  // This is incredibly helpful to leave enabled by default.
  // If a test passes locally but fails in CI, it lets us see logs in CI.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents on separate ports with 1KB host buffers
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Ucx2", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  size_t srcOffset = 0;
  size_t dstOffset = bufferSize / 4; // Arbitrary offset

  ASSERT_LE(srcOffset + transferSize, dramBuffer1.size());
  ASSERT_LE(dstOffset + transferSize, dramBuffer2.size());

  const char srcValue = 'S';
  const char initialDstValue = 'D';

  std::fill(dramBuffer1.begin() + srcOffset,
            dramBuffer1.begin() + srcOffset + transferSize, srcValue);
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), initialDstValue);

  // === Prepare Transfer Descriptors ===
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data() + srcOffset, transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data() + dstOffset, transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  transferData(nixl1, nixl2, regSrcDescs, regDstDescs);

  // Verify transferred data
  ASSERT_EQ(std::memcmp(dramBuffer1.data() + srcOffset,
                        dramBuffer2.data() + dstOffset, transferSize),
            0)
      << "Transferred data in dramBuffer2 does not match source in dramBuffer1";

  // Check if data outside the destination range was untouched
  if (dstOffset > 0) {
    ASSERT_EQ(dramBuffer2[dstOffset - 1], initialDstValue)
        << "Data before destination offset was modified";
  }
  if (dstOffset + transferSize < dramBuffer2.size()) {
    ASSERT_EQ(dramBuffer2[dstOffset + transferSize], initialDstValue)
        << "Data after destination range was modified";
  }
}

TEST_F(NixlUcxIntegrationTest, VramTransfer) {
  // Explicitly enable verbose logging for UCX.
  // This is incredibly helpful to leave enabled by default.
  // If a test passes locally but fails in CI, it lets us see logs in CI.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents on separate ports and separate GPU devices with 1KB
  // device buffers
  const size_t bufferSize = 1024;
  const char srcValue = 'S';
  const char initialDstValue = 'D';

  // NOTE: needs CUDA API for now, will make generic later

  // Initialize the NIXL agents
  // Note that we MUST instantiate the CUDA device context before loading UCX
  // NIXL plugin or else there will be a dlopen error.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::gpuAPI, 0));
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer1,
                ctx1->enqueueCreateBuffer(bufferSize));
  EXPECT_NO_ERR(ctx1->enqueueSetMemory(*deviceBuffer1, srcValue, sizeof(char)));
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", available_port(),
                                  "ucx", 0);
  nixl1.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer1->ptr()),
                   bufferSize, 0, ""});

  ErrorOr<DeviceContextRef> ctx2Result =
      DeviceContext::create(Device::gpuAPI, /*id=*/1);
  if (ctx2Result.isError()) {
    GTEST_SKIP() << "Need at least 2 GPU devices for this test";
  }
  DeviceContextRef ctx2 = ctx2Result.takeValue();
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer2,
                ctx2->enqueueCreateBuffer(bufferSize));
  EXPECT_NO_ERR(
      ctx2->enqueueSetMemory(*deviceBuffer2, initialDstValue, sizeof(char)));
  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Ucx2", available_port(),
                                  "ucx", 1);
  nixl2.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer2->ptr()),
                   bufferSize, 1, ""});

  EXPECT_NO_ERR(ctx1->synchronize());
  EXPECT_NO_ERR(ctx2->synchronize());

  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  size_t srcOffset = 0;
  size_t dstOffset = bufferSize / 4; // Arbitrary offset

  // === Prepare Transfer Descriptors ===
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::VRAM_SEG);
  regSrcDescs.addDesc(nixlBasicDesc{
      reinterpret_cast<uintptr_t>(deviceBuffer1->ptr()) + srcOffset,
      transferSize, 0});

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::VRAM_SEG);
  regDstDescs.addDesc(nixlBasicDesc{
      reinterpret_cast<uintptr_t>(deviceBuffer2->ptr()) + dstOffset,
      transferSize, 1});

  transferData(nixl1, nixl2, regSrcDescs, regDstDescs);

  // Verify transferred data
  // Allocate host buffers for verification
  std::vector<char> dramBuffer1(bufferSize);
  std::vector<char> dramBuffer2(bufferSize);

  // Copy data back from device to host for verification
  EXPECT_NO_ERR(
      ctx1->enqueueCopyFromDevice(dramBuffer1.data(), *deviceBuffer1));
  EXPECT_NO_ERR(
      ctx2->enqueueCopyFromDevice(dramBuffer2.data(), *deviceBuffer2));

  // Synchronize to ensure copies are complete
  EXPECT_NO_ERR(ctx1->synchronize());
  EXPECT_NO_ERR(ctx2->synchronize());
  ASSERT_EQ(std::memcmp(dramBuffer1.data() + srcOffset,
                        dramBuffer2.data() + dstOffset, transferSize),
            0)
      << "Transferred data in dramBuffer2 does not match source in dramBuffer1";

  // Check if data outside the destination range was untouched
  if (dstOffset > 0) {
    ASSERT_EQ(dramBuffer2[dstOffset - 1], initialDstValue)
        << "Data before destination offset was modified";
  }
  if (dstOffset + transferSize < dramBuffer2.size()) {
    ASSERT_EQ(dramBuffer2[dstOffset + transferSize], initialDstValue)
        << "Data after destination range was modified";
  }
}

TEST_F(NixlUcxIntegrationTest, BasicXfer_Read) {
  // Test basic successful READ operation.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  size_t srcOffset = 0;
  size_t dstOffset = bufferSize / 4;

  const char srcValue = 'S';
  const char initialDstValue = 'D';

  std::fill(dramBuffer2.begin() + srcOffset,
            dramBuffer2.begin() + srcOffset + transferSize, srcValue);
  std::fill(dramBuffer1.begin(), dramBuffer1.end(), initialDstValue);

  // Prepare transfer descriptors for READ operation
  nixl_xfer_dlist_t localDescs(nixl_mem_t::DRAM_SEG);
  auto localBuf = llvm::ArrayRef{dramBuffer1.data() + dstOffset, transferSize};
  localDescs.addDesc(basicDescfromArrayRef(localBuf, 0));

  nixl_xfer_dlist_t remoteDescs(nixl_mem_t::DRAM_SEG);
  auto remoteBuf = llvm::ArrayRef{dramBuffer2.data() + srcOffset, transferSize};
  remoteDescs.addDesc(basicDescfromArrayRef(remoteBuf, 0));

  // Perform READ transfer
  transferData(nixl1, nixl2, localDescs, remoteDescs,
               nixl_xfer_op_t::NIXL_READ);

  // Verify transferred data
  ASSERT_EQ(std::memcmp(dramBuffer1.data() + dstOffset,
                        dramBuffer2.data() + srcOffset, transferSize),
            0)
      << "READ: Transferred data in dramBuffer1 does not match source in "
         "dramBuffer2";

  // Check if data outside the destination range was untouched
  if (dstOffset > 0) {
    ASSERT_EQ(dramBuffer1[dstOffset - 1], initialDstValue)
        << "READ: Data before destination offset was modified";
  }
  if (dstOffset + transferSize < dramBuffer1.size()) {
    ASSERT_EQ(dramBuffer1[dstOffset + transferSize], initialDstValue)
        << "READ: Data after destination range was modified";
  }
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryDisabled) {
  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "n", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Telemetry1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Telemetry2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  EXPECT_EQ(nixl1.agent->getXferTelemetry(reqHandle, telemetry),
            nixl_status_t::NIXL_ERR_NO_TELEMETRY);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryEnabledByEnv) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_env_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("MODULAR_NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_TelemetryEnabled1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_TelemetryEnabled2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  EXPECT_EQ(telemetry.totalBytes, bufferSize);
  EXPECT_EQ(telemetry.descCount, 1u);
  EXPECT_GE(telemetry.postDuration, chrono_period_us_t(0));
  EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("MODULAR_NIXL_TELEMETRY_DIR");
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}

// Pins the contract that ``xferDuration`` reflects when the backend's
// progress thread observed the transfer complete, NOT the wall-clock at
// which the agent's poll loop happened to call ``getXferStatus``.
//
// The test forces a large gap between actual completion and the first
// poll: it waits for the receiver's notification to land (proxy for
// "the transfer is done"), sleeps 50ms, *then* makes its first
// ``getXferStatus`` call. With the old poll-time stamping, xferDuration
// would inflate to 50ms+. With ``getCompletionTime`` plumbed through
// it stays bounded by the actual transfer.
TEST_F(NixlUcxIntegrationTest, XferDurationDoesNotInflateUnderPollDelay) {
  namespace fs = std::filesystem;

  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_progress_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("MODULAR_NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_XferDurationBounded1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_XferDurationBounded2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "progress_thread_completion";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);

  // Wait for the receiver to see the notification: by that time the
  // sender's progress thread has also flipped the request to SUCCESS,
  // but we deliberately have NOT called getXferStatus yet.
  ASSERT_TRUE(wait_until_true([&]() {
    nixl_notifs_t notifMap;
    if (nixl2.agent->getNotifs(notifMap) != nixl_status_t::NIXL_SUCCESS) {
      return false;
    }
    auto it = notifMap.find(nixl1.agentName);
    return it != notifMap.end() && !it->second.empty();
  }));

  // Inject a polling gap. xferDuration must not absorb this delay.
  // Busy-wait rather than ``sleep_for`` so the gap has a guaranteed
  // lower bound regardless of OS scheduling.
  constexpr auto kPollGap = std::chrono::milliseconds(50);
  const auto deadline = std::chrono::steady_clock::now() + kPollGap;
  while (std::chrono::steady_clock::now() < deadline) {
  }

  status = nixl1.agent->getXferStatus(reqHandle);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  EXPECT_LT(telemetry.xferDuration, kPollGap)
      << "xferDuration=" << telemetry.xferDuration.count()
      << "us; expected to reflect progress-thread completion, not the "
      << kPollGap.count() << "ms poll delay.";

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("MODULAR_NIXL_TELEMETRY_DIR");
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryEnabledForMakeXferReq) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() /
      ("nixl_ucx_telem_make_req_" + std::to_string(getpid()) + "_" +
       std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("MODULAR_NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);
  setenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);

  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_TelemetryMakeReq1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_TelemetryMakeReq2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlDlistH *localHandle = nullptr;
  nixlDlistH *remoteHandle = nullptr;
  ASSERT_SUCCESS(nixl1.agent->prepXferDlist("", srcDescs, localHandle));
  ASSERT_SUCCESS(
      nixl1.agent->prepXferDlist(nixl2.agentName, dstDescs, remoteHandle));

  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->makeXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                          localHandle, {0}, remoteHandle, {0},
                                          reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  nixl_xfer_telem_t telemetry;
  ASSERT_SUCCESS(nixl1.agent->getXferTelemetry(reqHandle, telemetry));
  EXPECT_EQ(telemetry.totalBytes, bufferSize);
  EXPECT_EQ(telemetry.descCount, 1u);
  EXPECT_GE(telemetry.postDuration, chrono_period_us_t(0));
  EXPECT_GE(telemetry.xferDuration, telemetry.postDuration);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  ASSERT_SUCCESS(nixl1.agent->releasedDlistH(localHandle));
  ASSERT_SUCCESS(nixl1.agent->releasedDlistH(remoteHandle));
  ASSERT_SUCCESS(nixl1.agent->invalidateRemoteMD(nixl2.agentName));
  ASSERT_SUCCESS(nixl2.agent->invalidateRemoteMD(nixl1.agentName));
  nixl_reg_dlist_t regDescs1(nixl_mem_t::DRAM_SEG);
  regDescs1.addDesc(blobDescfromVector(dramBuffer1, 0));
  nixl_opt_args_t regArgs1;
  regArgs1.backends = {nixl1.backend};
  ASSERT_SUCCESS(nixl1.agent->deregisterMem(regDescs1, &regArgs1));
  nixl_reg_dlist_t regDescs2(nixl_mem_t::DRAM_SEG);
  regDescs2.addDesc(blobDescfromVector(dramBuffer2, 0));
  nixl_opt_args_t regArgs2;
  regArgs2.backends = {nixl2.backend};
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(regDescs2, &regArgs2));
  fs::remove_all(telemetryDir);
  unsetenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("MODULAR_NIXL_TELEMETRY_DIR");
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, GetXferTelemetryWritesSharedBufferFile) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 1);
  setenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);
  const fs::path telemetryDir = fs::temp_directory_path() /
                                ("nixl_ucx_telem_" + std::to_string(getpid()) +
                                 "_" + std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("MODULAR_NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);

  size_t bufferSize = 512;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_TelemetryFile1",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer1(bufferSize, 'S');
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_TelemetryFile2",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize, 'D');
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  exchangeMetadata(nixl1, nixl2);

  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  srcDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), dramBuffer1.size()}, 0));
  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  dstDescs.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), dramBuffer2.size()}, 0));

  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "telemetry";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));
  ASSERT_NE(reqHandle, nullptr);

  nixl_status_t status = nixl1.agent->postXferReq(reqHandle);
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG);
  ASSERT_TRUE(wait_until_true([&]() {
    status = nixl1.agent->getXferStatus(reqHandle);
    return status == nixl_status_t::NIXL_SUCCESS;
  }));
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS);

  const fs::path telemetryPath = telemetryDir / nixl1.agentName;
  ASSERT_TRUE(wait_until_true([&]() { return fs::exists(telemetryPath); }));

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      telemetryPath.string(), /*create=*/false, TELEMETRY_VERSION);
  ASSERT_TRUE(wait_until_true([&]() { return buffer->size() >= 4; }));

  nixlTelemetryEvent event;
  bool sawTransferEvent = false;
  while (buffer->pop(event)) {
    if (std::string(event.eventName_) == "agent_xfer_time") {
      sawTransferEvent = true;
      break;
    }
  }
  EXPECT_TRUE(sawTransferEvent);

  ASSERT_SUCCESS(nixl1.agent->releaseXferReq(reqHandle));
  fs::remove_all(telemetryDir);
  unsetenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("MODULAR_NIXL_TELEMETRY_DIR");
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}

TEST_F(NixlUcxIntegrationTest, TelemetryTracksMemoryRegistrationLifecycle) {
  namespace fs = std::filesystem;

  setenv("UCX_LOG_LEVEL", "debug", 1);
  setenv("MODULAR_NIXL_TELEMETRY_ENABLE", "y", 1);
  setenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL", "1", 1);
  const fs::path telemetryDir =
      fs::temp_directory_path() / ("nixl_ucx_mem_" + std::to_string(getpid()) +
                                   "_" + std::to_string(available_port()));
  fs::create_directories(telemetryDir);
  setenv("MODULAR_NIXL_TELEMETRY_DIR", telemetryDir.c_str(), 1);

  const size_t bufferSize = 256;
  NixlAgentWithUCXBackend nixl;
  NixlAgentWithUCXBackend::create(nixl, "TestAgent_MemoryTelemetry",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer(bufferSize, 'R');

  nixl_reg_dlist_t regDescs(nixl_mem_t::DRAM_SEG);
  regDescs.addDesc(blobDescfromVector(dramBuffer, 0));

  nixl_opt_args_t regArgs;
  regArgs.backends = {nixl.backend};
  ASSERT_SUCCESS(nixl.agent->registerMem(regDescs, &regArgs));
  ASSERT_SUCCESS(nixl.agent->deregisterMem(regDescs, &regArgs));

  auto buffer = std::make_unique<sharedRingBuffer<nixlTelemetryEvent>>(
      (telemetryDir / nixl.agentName).string(), /*create=*/false,
      TELEMETRY_VERSION);
  ASSERT_TRUE(wait_until_true([&]() { return !buffer->empty(); }));

  bool sawRegistered = false;
  bool sawDeregistered = false;
  nixlTelemetryEvent event;
  while (buffer->pop(event)) {
    const std::string_view eventName(event.eventName_);
    if (eventName == "agent_memory_registered") {
      sawRegistered = true;
      EXPECT_EQ(event.value_, bufferSize);
    }
    if (eventName == "agent_memory_deregistered") {
      sawDeregistered = true;
      EXPECT_EQ(event.value_, bufferSize);
    }
  }

  EXPECT_TRUE(sawRegistered);
  EXPECT_TRUE(sawDeregistered);

  fs::remove_all(telemetryDir);
  unsetenv("MODULAR_NIXL_TELEMETRY_RUN_INTERVAL");
  unsetenv("MODULAR_NIXL_TELEMETRY_DIR");
  unsetenv("MODULAR_NIXL_TELEMETRY_ENABLE");
}
