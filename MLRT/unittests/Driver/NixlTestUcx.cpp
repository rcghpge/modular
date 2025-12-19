//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/Driver/NIXL/nixl.h"
#include "MLRT/unittests/Driver/TestUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <netinet/in.h>
#include <random>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
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

// Helper function to find an available port
static int available_port(int start_port = 8000, int end_port = 9000,
                          int max_attempts = 100) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(start_port, end_port);

  for (int i = 0; i < max_attempts; ++i) {
    int port = dis(gen);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      continue;
    }

    // Set SO_REUSEADDR to avoid TIME_WAIT issues
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      close(sock);
      return port;
    }

    close(sock);
  }

  llvm_unreachable("no available port found");
}

// Helper function to wait until a condition becomes true with timeout
static bool wait_until_true(std::function<bool()> func, int retries = 500) {
  bool result;
  while (!(result = func()) && retries-- > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return result;
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
                     int numWorkers = 1, std::string errorHandlingMode = "",
                     int numThreads = 0) {
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

    if (numThreads > 0)
      backendParams["num_threads"] = std::to_string(numThreads);

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

TEST_F(NixlUcxIntegrationTest, UcxThreadpoolSmoke) {
  setenv("UCX_LOG_LEVEL", "warn", 1);

  // Create two agents with UCX backend using a threadpool configuration:
  // 2 workers total, 1 dedicated thread (leaves 1 shared worker).
  const size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TP_Agent1", available_port(), "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/2, /*errMode=*/"",
                                  /*numThreads=*/1);

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TP_Agent2", available_port(), "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/2, /*errMode=*/"",
                                  /*numThreads=*/1);
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange connection/metadata and perform a small transfer
  exchangeMetadata(nixl1, nixl2);

  const size_t transferSize = bufferSize / 4;
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, 'X');
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), 'Z');

  nixl_xfer_dlist_t src(nixl_mem_t::DRAM_SEG);
  nixl_xfer_dlist_t dst(nixl_mem_t::DRAM_SEG);
  src.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), transferSize}, 0));
  dst.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), transferSize}, 0));

  transferData(nixl1, nixl2, src, dst);

  // Verify copy
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0);
}

TEST_F(NixlUcxIntegrationTest, UcxThreadpoolMultiThread) {
  setenv("UCX_LOG_LEVEL", "warn", 1);

  // Create two agents with UCX backend using a threadpool configuration:
  // 3 workers total, 2 dedicated threads (leaves 1 shared worker).
  // This tests the scenario with numThreads > 1.
  const size_t bufferSize = 4096;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TP_MT_Agent1", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/3, /*errMode=*/"",
                                  /*numThreads=*/2);

  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TP_MT_Agent2", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1,
                                  /*numWorkers=*/3, /*errMode=*/"",
                                  /*numThreads=*/2);
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange connection/metadata
  exchangeMetadata(nixl1, nixl2);

  // Perform multiple transfers to exercise multiple workers
  const size_t transferSize = bufferSize / 2;
  std::fill(dramBuffer1.begin(), dramBuffer1.begin() + transferSize, 'M');
  std::fill(dramBuffer2.begin(), dramBuffer2.end(), 'N');

  nixl_xfer_dlist_t src(nixl_mem_t::DRAM_SEG);
  nixl_xfer_dlist_t dst(nixl_mem_t::DRAM_SEG);
  src.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer1.data(), transferSize}, 0));
  dst.addDesc(basicDescfromArrayRef(
      llvm::ArrayRef{dramBuffer2.data(), transferSize}, 0));

  transferData(nixl1, nixl2, src, dst);

  // Verify copy
  ASSERT_EQ(std::memcmp(dramBuffer1.data(), dramBuffer2.data(), transferSize),
            0);
}

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
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer1, ctx1->createBuffer(bufferSize));
  EXPECT_NO_ERR(ctx1->enqueueSetMemory(*deviceBuffer1, srcValue, sizeof(char)));
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Ucx1", available_port(),
                                  "ucx", 0);
  nixl1.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer1->ptr()),
                   bufferSize, 0, ""});

  ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                DeviceContext::create(Device::gpuAPI, 1));
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer2, ctx2->createBuffer(bufferSize));
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

TEST_F(NixlUcxIntegrationTest, ReceiverFailure) {
  // Test remote disconnection error handling
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  // Use "peer" error handling mode to propagate remote disconnect errors.
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", available_port(),
                                  "ucx",
                                  /*gpuDeviceId=*/-1, /*numWorkers=*/1,
                                  /*errorHandlingMode=*/"peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", available_port(),
                                  "ucx",
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

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Sender", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver", available_port(),
                                  "ucx", -1, 1, "peer");
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

  // Create two NIXL agents with peer error handling mode.
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Reader", available_port(),
                                  "ucx", -1, 1, "peer");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Source", available_port(),
                                  "ucx", -1, 1, "peer");
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

TEST_F(NixlUcxIntegrationTest, ErrorHandling_PeerMode) {
  // Test explicit peer error handling mode configuration
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create agents with explicit peer error handling mode
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Peer1", available_port(),
                                  "ucx", -1, 1, "peer");

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
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Peer2", available_port(),
                                  "ucx", -1, 1, "peer");
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

  for (int i = 0; i < numThreads; ++i) {
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

  for (int i = 0; i < numThreads; ++i) {
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
              ++successCount;
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
  for (int i = 0; i < numThreads; ++i) {
    ASSERT_EQ(std::memcmp(dramBuffers1[i].data(), dramBuffers2[i].data(),
                          transferSize),
              0)
        << "Buffer " << i << " transfer verification failed";

    // Check untouched region
    ASSERT_EQ(dramBuffers2[i][transferSize], 'Z')
        << "Buffer " << i << " has data corruption beyond transfer region";
  }
}

// Test loading remote VRAM metadata from a thread with wrong CUDA context.
// This simulates what happens in the listener thread when it has a different
// device context active than the one expected by the VRAM memory.
TEST_F(NixlUcxIntegrationTest, LoadRemoteMDWithWrongContext) {
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Check if we have at least 2 CUDA devices
  // Try to create context on device 1 - if it fails, we don't have 2 devices
  ErrorOr<DeviceContextRef> ctx1Result =
      DeviceContext::create(Device::gpuAPI, /*id=*/1);
  if (ctx1Result.isError())
    GTEST_SKIP() << "Need at least 2 CUDA devices for this test";

  DeviceContextRef ctx1 = ctx1Result.takeValue();

  // Create CUDA device context for device 0.
  ASSIGN_NO_ERR(DeviceContextRef, ctx0,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Create device buffer on device 0.
  const size_t bufferSize = 64 * 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer0, ctx0->createBuffer(bufferSize));

  // Create NIXL agent1 with device 0.
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Device0",
                                  /*port=*/available_port(), "ucx",
                                  /*gpuDeviceId=*/0);
  nixl1.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer0->ptr()),
                   bufferSize, 0, ""});

  // Create NIXL agent2 with device 1.
  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Device1",
                                  /*port=*/available_port(), "ucx",
                                  /*gpuDeviceId=*/1);

  // Get metadata from agent1 (device 0).
  nixl_blob_t meta1;
  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));

  // Create a thread that will have device 1 context active.
  std::thread wrongContextThread([&]() {
    // Activate device 1 context in this thread.
    DeviceContextScope scope(*ctx1);
    ASSERT_FALSE(scope.err().isError());

    // Now try to load metadata for device 0 VRAM while device 1 is active.
    // This simulates the listener thread scenario where the wrong context
    // might be active when processing metadata from a different device.
    nixl_status_t status =
        nixl2.agent->loadRemoteMD(blob_ref{meta1}, nixl1.agentName);

    // This should still succeed because the implementation should handle
    // activating the correct context for the VRAM operations.
    EXPECT_EQ(status, nixl_status_t::NIXL_SUCCESS);
  });

  wrongContextThread.join();
}

// Test loading remote VRAM metadata from a thread with NO CUDA context.
// This directly simulates the listener thread scenario.
TEST_F(NixlUcxIntegrationTest, LoadRemoteMDWithNoContext) {
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create CUDA device context for device 0.
  ASSIGN_NO_ERR(DeviceContextRef, ctx0,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Create device buffer on device 0.
  const size_t bufferSize = 64 * 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer0, ctx0->createBuffer(bufferSize));

  // Create NIXL agent with device 0.
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_NoCtx",
                                  /*port=*/available_port(), "ucx",
                                  /*gpuDeviceId=*/0);
  nixl1.registerMemory(
      nixl_mem_t::VRAM_SEG,
      nixlBlobDesc{reinterpret_cast<uintptr_t>(deviceBuffer0->ptr()),
                   bufferSize, 0, ""});

  // Get metadata with VRAM memory
  nixl_blob_t meta1;
  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));

  // Create another agent to receive the metadata.
  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Receiver",
                                  /*port=*/available_port(), "ucx",
                                  /*gpuDeviceId=*/0);

  // Create a thread with NO device context (simulating listener thread).
  std::thread noContextThread([&]() {
    // This thread has no CUDA context active - exactly like the listener
    // thread. Try to load VRAM metadata without any context.
    std::string loadedAgentName;
    nixl_status_t status =
        nixl2.agent->loadRemoteMD(blob_ref{meta1}, loadedAgentName);

    // Without activateContext, this would fail because UCX operations on VRAM
    // need a CUDA context.
    EXPECT_EQ(status, nixl_status_t::NIXL_SUCCESS);
    EXPECT_EQ(loadedAgentName, nixl1.agentName);
  });

  noContextThread.join();
}

TEST_F(NixlUcxIntegrationTest, SelfNotification) {
  // Test self notification support
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create a single NIXL agent that will send notifications to itself.
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Self", available_port(),
                                  "ucx");

  // Test multiple self notifications.
  const int numNotifications = 10;
  std::vector<std::string> sentMessages;

  for (int i = 0; i < numNotifications; ++i) {
    std::string msg = "self_notification_" + std::to_string(i);
    sentMessages.push_back(msg);
    ASSERT_SUCCESS(nixl1.agent->genNotif(nixl1.agentName, msg));
  }

  // Poll for notifications with timeout.
  nixl_notifs_t notifMap;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);
  size_t receivedCount = 0;

  while (receivedCount < numNotifications &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    ASSERT_SUCCESS(nixl1.agent->getNotifs(notifMap));
    if (notifMap.count(nixl1.agentName) > 0) {
      receivedCount = notifMap[nixl1.agentName].size();
    }
    if (receivedCount < numNotifications) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Verify we received all self notifications.
  ASSERT_EQ(receivedCount, numNotifications)
      << "Did not receive all self notifications within timeout";

  // Verify notification content matches what was sent.
  ASSERT_EQ(notifMap.count(nixl1.agentName), 1)
      << "Self notification map missing agent";

  const std::vector<nixl_blob_t> &receivedMessages = notifMap[nixl1.agentName];
  ASSERT_EQ(receivedMessages.size(), sentMessages.size())
      << "Incorrect number of self notifications received";

  // Check that all sent messages were received (order may vary).
  for (const std::string &sentMsg : sentMessages) {
    auto it =
        std::find(receivedMessages.begin(), receivedMessages.end(), sentMsg);
    ASSERT_NE(it, receivedMessages.end())
        << "Self notification message not found: " << sentMsg;
  }
}

// TODO(GEX-2564): Re-enable after fixing flakiness.
TEST_F(NixlUcxIntegrationTest, DISABLED_MetadataInvalidation) {
  // Test socket-based metadata invalidation functionality
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with listener threads enabled
  size_t bufferSize = 1024;
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Source", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  int port2 = available_port();
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Dest", port2, "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata - nixl2 loads metadata from nixl1
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));
  ASSERT_SUCCESS(nixl2.agent->getLocalMD(meta2));

  ASSERT_SUCCESS(nixl2.agent->loadRemoteMD(blob_ref{meta1}, loadedAgentName1));
  ASSERT_EQ(loadedAgentName1, nixl1.agentName);

  // Verify that remote metadata check succeeds initially
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  auto checkBuf = llvm::ArrayRef{dramBuffer1.data(), bufferSize};
  checkDescs.addDesc(basicDescfromArrayRef(checkBuf, 0));
  ASSERT_SUCCESS(nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs));

  // Invalidate metadata from nixl1 to nixl2
  nixl_opt_args_t invalidateArgs;
  invalidateArgs.ipAddr = "127.0.0.1";
  invalidateArgs.port = port2; // nixl2's port

  ASSERT_SUCCESS(nixl1.agent->invalidateLocalMD(&invalidateArgs));

  // Give time for invalidation message to be processed
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Verify that remote metadata check now fails
  ASSERT_NE(nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs),
            nixl_status_t::NIXL_SUCCESS);
}

TEST_F(NixlUcxIntegrationTest, ListenerCommSize) {
  // Test large metadata exchange to verify segmented message handling
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with listener threads enabled
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_LargeSource",
                                  available_port(), "ucx");

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_LargeDest",
                                  available_port(), "ucx");

  // Register many small buffers to create large metadata
  const size_t bufferSize = 64;
  const size_t numBuffers = 1000;
  std::vector<std::vector<char>> buffers(numBuffers);

  for (size_t i = 0; i < numBuffers; ++i) {
    buffers[i].resize(bufferSize);
    std::fill(buffers[i].begin(), buffers[i].end(),
              static_cast<char>('A' + (i % 26)));
    nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(buffers[i], 0));
  }

  // Exchange metadata - this will create a large message that tests
  // segmentation
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));
  ASSERT_SUCCESS(nixl2.agent->getLocalMD(meta2));

  // The large metadata should be exchanged successfully via segmented messages
  ASSERT_SUCCESS(nixl1.agent->loadRemoteMD(blob_ref{meta2}, loadedAgentName2));
  ASSERT_EQ(loadedAgentName2, nixl2.agentName);

  // Verify metadata exchange completed within timeout using wait_until_true
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  auto checkBuf = llvm::ArrayRef{buffers[0].data(), bufferSize};
  checkDescs.addDesc(basicDescfromArrayRef(checkBuf, 0));

  ASSERT_TRUE(wait_until_true([&]() {
    return nixl1.agent->checkRemoteMD(nixl2.agentName, checkDescs) ==
           nixl_status_t::NIXL_SUCCESS;
  })) << "Large metadata exchange failed or timed out";
}

TEST_F(NixlUcxIntegrationTest, ConcurrentNotifications) {
  // Test concurrent genNotif calls with RW sync mode.
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents configured with RW sync mode.
  NixlAgentWithUCXBackend nixl1;
  nixl1.agentName = "TestAgent_NotifSender";
  nixl1.agent = std::make_unique<nixlAgent>(
      nixl1.agentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port(),
                      /*sync_mode=*/nixl_thread_sync_t::NIXL_THREAD_SYNC_RW));

  // Create backend with UCX.
  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  std::vector<nixl_backend_t> plugins;
  ASSERT_SUCCESS(nixl1.agent->getAvailPlugins(plugins));
  bool ucxFound =
      std::find_if(plugins.begin(), plugins.end(), [](const nixl_backend_t &p) {
        return StringRef(p).lower() == "ucx";
      }) != plugins.end();
  ASSERT_TRUE(ucxFound) << "UCX plugin not found";

  ASSERT_SUCCESS(nixl1.agent->getPluginParams("ucx", mems, backendParams));
  ASSERT_SUCCESS(
      nixl1.agent->createBackend("ucx", backendParams, nixl1.backend));

  // Register memory.
  std::vector<char> dramBuffer1(1024);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  // Create second agent with RW sync mode.
  NixlAgentWithUCXBackend nixl2;
  nixl2.agentName = "TestAgent_NotifReceiver";
  nixl2.agent = std::make_unique<nixlAgent>(
      nixl2.agentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port(),
                      /*sync_mode=*/nixl_thread_sync_t::NIXL_THREAD_SYNC_RW));

  ASSERT_SUCCESS(nixl2.agent->getPluginParams("ucx", mems, backendParams));
  ASSERT_SUCCESS(
      nixl2.agent->createBackend("ucx", backendParams, nixl2.backend));

  std::vector<char> dramBuffer2(1024);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Exchange metadata.
  exchangeMetadata(nixl1, nixl2);

  // Launch concurrent notification threads.
  const int numThreads = 4;
  const int notifsPerThread = 10;
  std::vector<std::thread> threads;
  std::atomic<int> successCount{0};

  for (int threadId = 0; threadId < numThreads; ++threadId) {
    threads.emplace_back([&, threadId]() {
      for (int i = 0; i < notifsPerThread; ++i) {
        std::string msg =
            "thread" + std::to_string(threadId) + "_notif" + std::to_string(i);
        auto status = nixl1.agent->genNotif(nixl2.agentName, msg);
        if (status == nixl_status_t::NIXL_SUCCESS)
          ++successCount;
      }
    });
  }

  // Wait for all threads to complete.
  for (std::thread &thread : threads)
    thread.join();

  // Verify all notifications were sent successfully
  ASSERT_EQ(successCount, numThreads * notifsPerThread)
      << "Not all concurrent notifications were sent successfully";

  // Poll for notifications with timeout.
  nixl_notifs_t notifMap;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);
  size_t receivedCount = 0;

  while (receivedCount < static_cast<size_t>(numThreads * notifsPerThread) &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    ASSERT_SUCCESS(nixl2.agent->getNotifs(notifMap));
    if (notifMap.count(nixl1.agentName) > 0) {
      receivedCount = notifMap[nixl1.agentName].size();
    }
    if (receivedCount < static_cast<size_t>(numThreads * notifsPerThread)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  // Verify we received all notifications.
  ASSERT_EQ(receivedCount, static_cast<size_t>(numThreads * notifsPerThread))
      << "Did not receive all concurrent notifications within timeout";
}

TEST_F(NixlUcxIntegrationTest, LoadRemoteWithErrors) {
  // Test error handling when loading metadata fails
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create source agent with backend
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Source", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer1(1024);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  // Create destination agent WITHOUT backend initially
  std::string destAgentName = "TestAgent_Dest";
  auto nixl2 = std::make_unique<nixlAgent>(
      destAgentName,
      nixlAgentConfig(/*use_prog_thread=*/true, /*use_listen_thread=*/true,
                      /*port=*/available_port()));

  std::string meta1;
  std::string loadedAgentName;

  ASSERT_SUCCESS(nixl1.agent->getLocalMD(meta1));

  // Try to load metadata without backend - should fail
  ASSERT_NE(nixl2->loadRemoteMD(blob_ref{meta1}, loadedAgentName),
            nixl_status_t::NIXL_SUCCESS);

  // Create backend for destination agent
  nixlBackendH *backend2 = nullptr;
  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  ASSERT_SUCCESS(nixl2->getPluginParams("ucx", mems, backendParams));
  ASSERT_SUCCESS(nixl2->createBackend("ucx", backendParams, backend2));

  // Try to load invalid metadata - should fail
  std::string invalidMeta = "invalid_metadata_string";
  ASSERT_NE(nixl2->loadRemoteMD(blob_ref{invalidMeta}, loadedAgentName),
            nixl_status_t::NIXL_SUCCESS);

  // Now load valid metadata - should succeed
  ASSERT_SUCCESS(nixl2->loadRemoteMD(blob_ref{meta1}, loadedAgentName));
  ASSERT_EQ(loadedAgentName, nixl1.agentName);
}

TEST_F(NixlUcxIntegrationTest, GetLocalPartialAndLoadRemote) {
  // Test partial metadata exchange functionality
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents
  NixlAgentWithUCXBackend nixl1;
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_Partial1", available_port(),
                                  "ucx");

  // Register multiple buffers to test partial metadata
  const size_t bufferSize = 512;
  const size_t numBuffers = 5;
  std::vector<std::vector<char>> buffers1(numBuffers);

  for (size_t i = 0; i < numBuffers; ++i) {
    buffers1[i].resize(bufferSize);
    std::fill(buffers1[i].begin(), buffers1[i].end(), 'A' + i);
    nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                         blobDescfromVector(buffers1[i], 0));
  }

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_Partial2", available_port(),
                                  "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  std::string partialMeta;
  std::string loadedAgentName;

  // Step 1: Get and load connection info only
  nixl_reg_dlist_t emptyList(nixl_mem_t::DRAM_SEG);
  ASSERT_SUCCESS(
      nixl1.agent->getLocalPartialMD(emptyList, partialMeta, nullptr));
  ASSERT_SUCCESS(
      nixl2.agent->loadRemoteMD(blob_ref{partialMeta}, loadedAgentName));
  ASSERT_EQ(loadedAgentName, nixl1.agentName);

  // Verify connection info was loaded
  nixl_xfer_dlist_t emptyXferList(nixl_mem_t::DRAM_SEG);
  ASSERT_SUCCESS(nixl2.agent->checkRemoteMD(nixl1.agentName, emptyXferList));

  // Step 2: Get partial metadata for first 3 buffers only
  nixl_reg_dlist_t partialDescs(nixl_mem_t::DRAM_SEG);
  for (size_t i = 0; i < 3; ++i) {
    partialDescs.addDesc(blobDescfromVector(buffers1[i], 0));
  }

  ASSERT_SUCCESS(
      nixl1.agent->getLocalPartialMD(partialDescs, partialMeta, nullptr));
  ASSERT_SUCCESS(
      nixl2.agent->loadRemoteMD(blob_ref{partialMeta}, loadedAgentName));

  // Verify only the first 3 buffers are available
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  for (size_t i = 0; i < 3; ++i) {
    auto buf = llvm::ArrayRef{buffers1[i].data(), bufferSize};
    checkDescs.addDesc(basicDescfromArrayRef(buf, 0));
    ASSERT_SUCCESS(nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs));
    checkDescs.clear();
  }

  // Verify buffers 4 and 5 are NOT available
  for (size_t i = 3; i < numBuffers; ++i) {
    nixl_xfer_dlist_t missingDescs(nixl_mem_t::DRAM_SEG);
    auto buf = llvm::ArrayRef{buffers1[i].data(), bufferSize};
    missingDescs.addDesc(basicDescfromArrayRef(buf, 0));
    ASSERT_NE(nixl2.agent->checkRemoteMD(nixl1.agentName, missingDescs),
              nixl_status_t::NIXL_SUCCESS);
  }
}

TEST_F(NixlUcxIntegrationTest, SocketFetchRemote) {
  // Test socket-based metadata fetching
  setenv("UCX_LOG_LEVEL", "debug", 1);

  // Create two NIXL agents with listener threads
  size_t bufferSize = 1024;

  NixlAgentWithUCXBackend nixl1;
  int port1 = available_port();
  NixlAgentWithUCXBackend::create(nixl1, "TestAgent_FetchSource", port1, "ucx");
  std::vector<char> dramBuffer1(bufferSize);
  nixl1.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer1, 0));

  NixlAgentWithUCXBackend nixl2;
  NixlAgentWithUCXBackend::create(nixl2, "TestAgent_FetchDest",
                                  available_port(), "ucx");
  std::vector<char> dramBuffer2(bufferSize);
  nixl2.registerMemory(nixl_mem_t::DRAM_SEG,
                       blobDescfromVector(dramBuffer2, 0));

  // Use fetchRemoteMD to request metadata from nixl1
  nixl_opt_args_t fetchArgs;
  fetchArgs.ipAddr = "127.0.0.1";
  fetchArgs.port = port1;

  ASSERT_SUCCESS(nixl2.agent->fetchRemoteMD(nixl1.agentName, &fetchArgs));

  // Wait for metadata to be received
  nixl_xfer_dlist_t checkDescs(nixl_mem_t::DRAM_SEG);
  auto checkBuf = llvm::ArrayRef{dramBuffer1.data(), bufferSize};
  checkDescs.addDesc(basicDescfromArrayRef(checkBuf, 0));

  // Use wait_until_true to poll for metadata arrival
  ASSERT_TRUE(wait_until_true([&]() {
    return nixl2.agent->checkRemoteMD(nixl1.agentName, checkDescs) ==
           nixl_status_t::NIXL_SUCCESS;
  })) << "Metadata fetch failed or timed out";

  // Fill source buffer with test data
  const size_t transferSize = bufferSize / 2;
  std::fill(dramBuffer2.begin(), dramBuffer2.begin() + transferSize, 'F');

  // Verify we can create a transfer using the fetched metadata
  nixl_xfer_dlist_t srcDescs(nixl_mem_t::DRAM_SEG);
  auto srcBuf = llvm::ArrayRef{dramBuffer2.data(), transferSize};
  srcDescs.addDesc(basicDescfromArrayRef(srcBuf, 0));

  nixl_xfer_dlist_t dstDescs(nixl_mem_t::DRAM_SEG);
  auto dstBuf = llvm::ArrayRef{dramBuffer1.data(), transferSize};
  dstDescs.addDesc(basicDescfromArrayRef(dstBuf, 0));

  // Perform transfer using fetched metadata
  transferData(nixl2, nixl1, srcDescs, dstDescs);

  // Verify transfer succeeded
  ASSERT_EQ(std::memcmp(dramBuffer2.data(), dramBuffer1.data(), transferSize),
            0)
      << "Transfer using fetched metadata failed";
}
