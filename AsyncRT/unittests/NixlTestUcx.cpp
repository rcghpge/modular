//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/NIXL/nixl.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADeviceContext.h"
#include "AsyncRT/lib/DeviceContext/CUDA/CUDADriver.h"
#include "AsyncRT/unittests/TestUtils.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// Pretty-printer for NIXL status values.
namespace M::AsyncRT::NIXL {
std::ostream &operator<<(std::ostream &os, nixl_status_t status) {
  return os << nixlEnumStrings::statusStr(status);
}
} // namespace M::AsyncRT::NIXL

using namespace M;
using namespace M::AsyncRT;
using namespace M::AsyncRT::NIXL;

// Make a NIXL call and assert that the status is NIXL_SUCCESS.
#define ASSERT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    nixl_status_t status = (EXPR);                                             \
    ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS) << #EXPR;                   \
  } while (false)

// Create a blob descriptor from a vector.
template <class T>
nixlBlobDesc blobDescfromVector(const std::vector<T> &vec, uint64_t devId,
                                llvm::StringRef metaInfo = "") {
  return {reinterpret_cast<uintptr_t>(vec.data()), vec.size() * sizeof(T),
          devId, metaInfo.str()};
}

// Create a basic descriptor from an ArrayRef.
template <class T>
nixlBasicDesc basicDescfromArrayRef(llvm::ArrayRef<T> ref, uint64_t devId) {
  return {reinterpret_cast<uintptr_t>(ref.data()), ref.size() * sizeof(T),
          devId};
}

// Helper class to hold state for a NIXL agent with a UCX backend.
struct NixlAgentWithUCXBackend {
  std::string agentName;
  std::unique_ptr<nixlAgent> agent;
  nixlBackendH *backend;

  void registerMemory(nixl_mem_t memType, nixlBlobDesc desc) {
    nixl_reg_dlist_t descList(memType);
    descList.addDesc(desc);
    nixl_opt_args_t regArgs;
    regArgs.backends = {backend};
    ASSERT_SUCCESS(agent->registerMem(descList, &regArgs));
  }

  static void create(NixlAgentWithUCXBackend &self, std::string agentName,
                     int port, std::string backendName, int gpuDeviceId = -1,
                     int numWorkers = 1, std::string errorHandlingMode = "") {
    self.agentName = agentName;
    self.agent = std::make_unique<nixlAgent>(
        agentName, nixlAgentConfig(/*use_prog_thread=*/true,
                                   /*use_listen_thread=*/true,
                                   /*port=*/port));
    // Check if UCX plugin is available
    std::vector<nixl_backend_t> plugins;
    ASSERT_SUCCESS(self.agent->getAvailPlugins(plugins));
    auto ucxFound =
        0 < std::count_if(plugins.begin(), plugins.end(),
                          [&](const nixl_backend_t &p) {
                            return StringRef(p).lower() == backendName;
                          });
    ASSERT_TRUE(ucxFound) << "Plugin not found: " << backendName;

    nixl_b_params_t backendParams;
    nixl_mem_list_t mems;
    ASSERT_SUCCESS(
        self.agent->getPluginParams(backendName, mems, backendParams));

    if (gpuDeviceId != -1) {
      backendParams["gpu_device_id"] = std::to_string(gpuDeviceId);
    }

    if (numWorkers > 1) {
      backendParams["num_workers"] = std::to_string(numWorkers);
    }

    if (!errorHandlingMode.empty()) {
      backendParams["ucx_error_handling_mode"] = errorHandlingMode;
    }

    ASSERT_SUCCESS(
        self.agent->createBackend(backendName, backendParams, self.backend));
    ASSERT_NE(self.backend, nullptr) << "Backend handle is null";
  }
};

static void exchangeMetadata(NixlAgentWithUCXBackend &nixl1,
                             NixlAgentWithUCXBackend &nixl2) {
  // Exchange metadata between the agends
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));
  ASSERT_SUCCESS(nixl2.agent->getLocalMD(meta2));

  ASSERT_FALSE(meta1.empty()) << "Agent 1 metadata is empty";
  ASSERT_FALSE(meta2.empty()) << "Agent 2 metadata is empty";

  ASSERT_SUCCESS(nixl1.agent->loadRemoteMD(blob_ref{meta2}, loadedAgentName2));
  ASSERT_EQ(loadedAgentName2, nixl2.agentName);

  ASSERT_SUCCESS(nixl2.agent->loadRemoteMD(blob_ref{meta1}, loadedAgentName1));
  ASSERT_EQ(loadedAgentName1, nixl1.agentName);
}

static void transferData(NixlAgentWithUCXBackend &nixl1,
                         NixlAgentWithUCXBackend &nixl2,
                         nixl_xfer_dlist_t &srcDescs,
                         nixl_xfer_dlist_t &dstDescs,
                         nixl_xfer_op_t op = nixl_xfer_op_t::NIXL_WRITE) {
  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(
      op, srcDescs, dstDescs, nixl2.agentName, reqHandle, &xferArgs));

  // === Create and Execute Transfer (Agent 1 writes to Agent 2) ===
  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  if (xferStatus != nixl_status_t::NIXL_SUCCESS &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq failed: " << nixlEnumStrings::statusStr(xferStatus);
  }

  // Poll for completion
  nixl_notifs_t notifMap;
  nixl_status_t checkStatus = nixl_status_t::NIXL_IN_PROG;
  bool notifReceived = false;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10); // 10 second timeout

  while ((checkStatus != nixl_status_t::NIXL_SUCCESS || !notifReceived) &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    if (checkStatus != nixl_status_t::NIXL_SUCCESS) {
      checkStatus = nixl1.agent->getXferStatus(reqHandle);
      ASSERT_NE(checkStatus, nixl_status_t::NIXL_ERR_UNKNOWN)
          << "getXferStatus returned error";
    }

    if (!notifReceived) {
      ASSERT_SUCCESS(nixl2.agent->getNotifs(notifMap));
      if (notifMap.count(nixl1.agentName) > 0 &&
          !notifMap[nixl1.agentName].empty()) {
        notifReceived = true;
      }
    }
    // Yield/Sleep briefly to avoid busy-waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, nixl_status_t::NIXL_SUCCESS)
      << "Transfer did not complete within timeout";
  ASSERT_TRUE(notifReceived) << "Notification not received within timeout";

  // === Verification ===
  // Verify notification content
  ASSERT_EQ(notifMap.count(nixl1.agentName), 1)
      << "Notification map missing agent1";
  ASSERT_EQ(notifMap[nixl1.agentName].size(), 1)
      << "Incorrect number of notifications received";
}

static void transferDataWithFailure(NixlAgentWithUCXBackend &nixl1,
                                    NixlAgentWithUCXBackend &nixl2,
                                    nixl_xfer_dlist_t &srcDescs,
                                    nixl_xfer_dlist_t &dstDescs,
                                    nixl_status_t expectedStatus) {
  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(nixl_xfer_op_t::NIXL_WRITE,
                                            srcDescs, dstDescs, nixl2.agentName,
                                            reqHandle, &xferArgs));

  // === Create and Execute Transfer (Agent 1 writes to Agent 2) ===
  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  // The error may be returned immediately or later during polling
  if (xferStatus != expectedStatus &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq returned unexpected status: "
           << nixlEnumStrings::statusStr(xferStatus);
  }

  // Poll for completion
  nixl_status_t checkStatus = xferStatus;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10); // 10 second timeout

  while (checkStatus == nixl_status_t::NIXL_IN_PROG &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    checkStatus = nixl1.agent->getXferStatus(reqHandle);
    // Yield/Sleep briefly to avoid busy-waiting
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, expectedStatus)
      << "Transfer did not complete with expected status. Got: "
      << nixlEnumStrings::statusStr(checkStatus)
      << ", Expected: " << nixlEnumStrings::statusStr(expectedStatus);

  // Clean up the request handle
  nixl1.agent->releaseXferReq(reqHandle);
}

// Fixture for UCX Integration Tests
class NixlUcxIntegrationTest : public ::testing::Test {};

TEST_F(NixlUcxIntegrationTest, DramTransfer) {
  // Explicitly enable verbose logging for UCX.
  // This is incredibly helpful to leave enabled by default.
  // If a test passes locally but fails in CI, it lets us see logs in CI.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Instantiating NIXL loads the UCX backend. Currently the only UCX backend
  // we have is the one that has CUDA enabled. This plugin's shared object
  // assumes that symbols from libcuda.so.1 and libnvidia-ml.so.1 (nvml) are
  // available at runtime globally (w/ RTLD_GLOBAL). As such, we must ensure
  // that we create a CUDA device context prior to instantiating a NIXL agent as
  // loading of CUDA and NVML is done in the DeviceContext constructor.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create two NIXL agents on separate ports with 1KB host buffers
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", 8037, "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Ucx2", 8042, "ucx");
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
                DeviceContext::create(Device::cudaAPI, 0));
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer1, ctx1->createBuffer(bufferSize));
  EXPECT_NO_ERR(ctx1->enqueueSetMemory(*deviceBuffer1, srcValue, sizeof(char)));
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", 8037, "ucx", 0);
  nixl1.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer1->ptr()),
                   bufferSize, 0, ""});

  ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                DeviceContext::create(Device::cudaAPI, 1));
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer2, ctx2->createBuffer(bufferSize));
  EXPECT_NO_ERR(
      ctx2->enqueueSetMemory(*deviceBuffer2, initialDstValue, sizeof(char)));
  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Ucx2", 8042, "ucx", 1);
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

TEST_F(NixlUcxIntegrationTest, ReceiverFailure) {
  // Test remote disconnection error handling
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create CUDA device context to ensure UCX plugin loads properly.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create two NIXL agents.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  // Use "peer" error handling mode to propagate remote disconnect errors.
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", 8037, "ucx",
                                  /*gpuDeviceId=*/-1, /*numWorkers=*/1,
                                  /*errorHandlingMode=*/"peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", 8042, "ucx",
                                  /*gpuDeviceId=*/-1, /*numWorkers=*/1,
                                  /*errorHandlingMode=*/"peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for transfer
  size_t transferSize = bufferSize / 2;
  const char srcValue = 'S';
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue);

  // Prepare transfer descriptors
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // Simulate receiver failure by deregistering memory and destroying agent
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));

  // Destroy the receiver agent to simulate disconnection
  nixl2.agent.reset();

  // Attempt transfer which should fail with NIXL_ERR_REMOTE_DISCONNECT
  transferDataWithFailure(nixl1, nixl2, regSrcDescs, regDstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, XferThenFail_Write) {
  // Test scenario where first transfer succeeds, then receiver fails before
  // second transfer
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create CUDA device context to ensure UCX plugin loads properly.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", 8037, "ucx", -1, 1,
                                  "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", 8042, "ucx", -1,
                                  1, "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for first transfer.
  size_t transferSize = bufferSize / 2;
  const char srcValue1 = 'A';
  const char srcValue2 = 'B';
  const char initialDstValue = 'Z';

  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue1);
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), initialDstValue);

  // Prepare transfer descriptors.
  nixl_xfer_dlist_t regSrcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  regSrcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  regDstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // First transfer should succeed.
  transferData(nixl1, nixl2, regSrcDescs, regDstDescs);

  // Verify first transfer succeeded.
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "First transfer failed to complete correctly";

  // Prepare for second transfer with different data.
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue2);

  // Simulate receiver failure by deregistering memory and destroying agent.
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Second transfer should fail with NIXL_ERR_REMOTE_DISCONNECT.
  transferDataWithFailure(nixl1, nixl2, regSrcDescs, regDstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, XferThenFail_Read) {
  // Test scenario where first READ transfer succeeds, then receiver fails
  // before second transfer
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create CUDA device context to ensure UCX plugin loads properly.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", 8037, "ucx", -1, 1,
                                  "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", 8042, "ucx", -1, 1,
                                  "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers for first transfer
  size_t transferSize = bufferSize / 2;
  const char srcValue1 = 'A';
  const char srcValue2 = 'B';
  const char initialDstValue = 'Z';

  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, srcValue1);
  std::fill(dramBuffer1.begin(), dramBuffer1.end(), initialDstValue);

  // Prepare transfer descriptors for READ operation
  // For READ: local buffer is destination, remote buffer is source
  nixl_xfer_dlist_t localDescs(nixl_mem_t::DRAM_SEG);
  auto localBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  localDescs.addDesc(basicDescfromArrayRef(localBuf, 0));

  nixl_xfer_dlist_t remoteDescs(nixl_mem_t::DRAM_SEG);
  auto remoteBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  remoteDescs.addDesc(basicDescfromArrayRef(remoteBuf, 0));

  // First READ transfer should succeed.
  transferData(nixl1, nixl2, localDescs, remoteDescs,
               nixl_xfer_op_t::NIXL_READ);

  // Verify first transfer succeeded.
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "First READ transfer failed to complete correctly";

  // Prepare for second transfer with different data.
  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, srcValue2);

  // Simulate source failure by deregistering memory and destroying agent.
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Second READ transfer should fail with NIXL_ERR_REMOTE_DISCONNECT
  transferDataWithFailure(nixl1, nixl2, localDescs, remoteDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, BasicXfer_Read) {
  // Test basic successful READ operation.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create two NIXL agents
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", 8037, "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", 8042, "ucx");
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

TEST_F(NixlUcxIntegrationTest, ErrorHandling_PeerMode) {
  // Test explicit peer error handling mode configuration
  setenv("UCX_LOG_LEVEL", "debug", 1);

  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Create agents with explicit peer error handling mode
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Peer1", 8037, "ucx", -1, 1,
                                  "peer");

  // Verify that the backend was created with peer mode
  nixl_b_params_t params;
  nixl_mem_list_t mems;
  ASSERT_SUCCESS(nixl1.agent->getBackendParams(nixl1.backend, mems, params));
  ASSERT_EQ(params["ucx_error_handling_mode"], "peer")
      << "Error handling mode was not set to peer";

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Peer2", 8042, "ucx", -1, 1,
                                  "peer");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata
  exchangeMetadata(nixl1, nixl2);

  // Prepare buffers
  size_t transferSize = bufferSize / 2;
  const char srcValue = 'P';
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, srcValue);

  // Prepare descriptors
  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // First transfer should succeed with peer mode
  transferData(nixl1, nixl2, srcDescs, dstDescs);

  // Verify transfer
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0)
      << "Transfer with peer mode failed";

  // Now test error handling with peer mode
  // Destroy receiver to trigger peer error handling
  nixl_opt_args_t deregArgs;
  deregArgs.backends = {nixl2.backend};
  nixl_reg_dlist_t deregList(nixl_mem_t::DRAM_SEG);
  deregList.addDesc(blobDescfromVector(dramBuffer2, 0));
  ASSERT_SUCCESS(nixl2.agent->deregisterMem(deregList, &deregArgs));
  nixl2.agent.reset();

  // Transfer should fail gracefully with peer error handling
  transferDataWithFailure(nixl1, nixl2, srcDescs, dstDescs,
                          nixl_status_t::NIXL_ERR_REMOTE_DISCONNECT);
}

TEST_F(NixlUcxIntegrationTest, MultiWorkerDramTransfer) {
  // Test multi-worker UCX support with concurrent transfers from multiple
  // threads.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create CUDA device context to ensure UCX plugin loads properly.
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::cudaAPI, 0));

  // Number of workers and threads for testing
  const int numWorkers = 4;
  const int numThreads = 4;
  const size_t bufferSize = 1024;
  const size_t transferSize = bufferSize / 2;

  // Create NIXL agents with multiple workers - use different ports to avoid
  // conflicts
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_MultiWorker1", 8047, "ucx",
                                  -1, numWorkers);

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_MultiWorker2", 8052, "ucx",
                                  -1, numWorkers);

  // Register multiple buffers for concurrent transfers
  std::vector<std::vector<char>> dramBuffers1(numThreads);
  std::vector<std::vector<char>> dramBuffers2(numThreads);

  for (int i = 0; i < numThreads; i++) {
    dramBuffers1[i].resize(bufferSize);
    dramBuffers2[i].resize(bufferSize);

    // Fill source buffers with unique values
    std::fill(dramBuffers1[i].begin(), dramBuffers1[i].end(), 'A' + i);
    std::fill(dramBuffers2[i].begin(), dramBuffers2[i].end(), 'Z');

    // Register memory for each buffer
    nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(dramBuffers1[i], 0));
    nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(dramBuffers2[i], 0));
  }

  // Exchange metadata
  exchangeMetadata(nixl1, nixl2);

  // Launch concurrent transfers from multiple threads
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int i = 0; i < numThreads; i++) {
    threads.emplace_back([&, i]() {
      // Prepare transfer descriptors for this thread
      nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
      auto srcBuf = llvm::ArrayRef{dramBuffers1[i].data(), transferSize};
      srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

      nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
      auto dstBuf = llvm::ArrayRef{dramBuffers2[i].data(), transferSize};
      dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

      // Create and execute transfer
      nixlXferReqH *reqHandle = nullptr;
      nixlAgentOptionalArgs xferArgs;
      xferArgs.notifMsg = "notification_thread_" + std::to_string(i);
      xferArgs.hasNotif = true;

      if (nixl1.agent->createXferReq(
              nixl_xfer_op_t::NIXL_WRITE, srcDescs, dstDescs, nixl2.agentName,
              reqHandle, &xferArgs) == nixl_status_t::NIXL_SUCCESS) {

        // Post the transfer
        nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
        if (xferStatus == nixl_status_t::NIXL_SUCCESS ||
            xferStatus == nixl_status_t::NIXL_IN_PROG) {

          // Poll for completion
          nixl_status_t checkStatus = xferStatus;
          const auto startTime = std::chrono::steady_clock::now();
          const auto timeout = std::chrono::seconds(10);

          while (checkStatus == nixl_status_t::NIXL_IN_PROG &&
                 (std::chrono::steady_clock::now() - startTime < timeout)) {
            checkStatus = nixl1.agent->getXferStatus(reqHandle);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
          }

          if (checkStatus == nixl_status_t::NIXL_SUCCESS) {
            // Verify the transfer
            if (std::memcmp(dramBuffers1[i].data(), dramBuffers2[i].data(),
                            transferSize) == 0) {
              successCount++;
            }
          }

          nixl1.agent->releaseXferReq(reqHandle);
        }
      }
    });
  }

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // Verify all transfers completed successfully
  ASSERT_EQ(successCount, numThreads)
      << "Not all concurrent transfers completed successfully. "
      << "Success count: " << successCount << "/" << numThreads;

  // Additionally verify all destination buffers have correct data
  for (int i = 0; i < numThreads; i++) {
    ASSERT_EQ(std::memcmp(dramBuffers1[i].data(), dramBuffers2[i].data(),
                          transferSize),
              0)
        << "Buffer " << i << " transfer verification failed";

    // Check untouched region
    ASSERT_EQ(dramBuffers2[i][transferSize], 'Z')
        << "Buffer " << i << " has data corruption beyond transfer region";
  }
}
