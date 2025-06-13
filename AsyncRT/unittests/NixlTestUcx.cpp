//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/NIXL/nixl.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
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
                     int port, std::string backendName, int gpuDeviceId = -1) {
    self.agentName = agentName;
    self.agent = std::make_unique<nixlAgent>(
        agentName, nixlAgentConfig(/*use_prog_thread=*/false,
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

    ASSERT_SUCCESS(
        self.agent->createBackend(backendName, backendParams, self.backend));
    ASSERT_NE(self.backend, nullptr) << "Backend handle is null";
  }
};

void exchangeMetadata(NixlAgentWithUCXBackend &nixl1,
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

void transferData(NixlAgentWithUCXBackend &nixl1,
                  NixlAgentWithUCXBackend &nixl2, nixl_xfer_dlist_t &srcDescs,
                  nixl_xfer_dlist_t &dstDescs) {
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

// Fixture for UCX Integration Tests
class NixlUcxIntegrationTest : public ::testing::Test {};

TEST_F(NixlUcxIntegrationTest, DramTransfer) {
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
