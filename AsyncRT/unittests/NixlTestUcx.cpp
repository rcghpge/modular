//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/NIXL/nixl.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace M;
using namespace M::AsyncRT::NIXL;

// Fixture for UCX Integration Tests
class NixlUcxIntegrationTest : public ::testing::Test {
protected:
  std::unique_ptr<nixlAgent> agent1 = nullptr;
  nixlBackendH *ucxBackend1 = nullptr; // Managed by agent1
  nixl_opt_args_t registrationArgs1;
  std::vector<char> dramBuffer1;
  size_t bufferSize = 1024;
  std::unique_ptr<nixl_reg_dlist_t> regDlist1 = nullptr;
  std::string agentName1 = "TestAgent_Ucx1";
  static constexpr int kAgent1Port = 8037;
  static constexpr int kAgent2Port = 8042;

  void SetUp() override {
    // 1. Configure agent: Enable listener and progress threads for UCX.
    nixlAgentConfig agentConfig(/*use_prog_thread=*/false,
                                /*use_listen_thread=*/true,
                                /*port=*/kAgent1Port);

    // 2. Create agent1.
    agent1 = std::make_unique<nixlAgent>(agentName1, agentConfig);
    ASSERT_NE(agent1, nullptr) << "Agent1 creation failed";

    // 3. Create UCX backend for agent1.
    nixl_b_params_t backendParams1;
    nixl_mem_list_t mems1; // Can be used to get/set params if needed

    // Check if UCX plugin is available
    std::vector<nixl_backend_t> plugins;
    ASSERT_EQ(agent1->getAvailPlugins(plugins), nixl_status_t::NIXL_SUCCESS);
    bool ucxFound = false;
    for (const nixl_backend_t &p : plugins) {
      if (StringRef(p).lower() == "ucx") {
        ucxFound = true;
        break;
      }
    }
    ASSERT_TRUE(ucxFound)
        << "UCX Plugin not found or NIXL not built with UCX support";

    ASSERT_EQ(agent1->getPluginParams("UCX", mems1, backendParams1),
              nixl_status_t::NIXL_SUCCESS)
        << "Failed to get UCX plugin parameters";

    // ucxBackend1 is managed by agent1
    nixl_status_t status =
        agent1->createBackend("UCX", backendParams1, ucxBackend1);
    ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
        << "Failed to create UCX backend for Agent1, status: "
        << nixlEnumStrings::statusStr(status);
    ASSERT_NE(ucxBackend1, nullptr) << "Agent1 UCX backend handle is null";

    // Specify the backend for registration
    registrationArgs1.backends.push_back(ucxBackend1);

    // 4. Memory Allocation for agent1.
    dramBuffer1.resize(bufferSize);
    ASSERT_FALSE(dramBuffer1.empty()) << "Agent1 DRAM buffer allocation failed";

    // 5. Descriptor Creation for agent1
    regDlist1 = std::make_unique<nixl_reg_dlist_t>(nixl_mem_t::DRAM_SEG);
    ASSERT_NE(regDlist1, nullptr)
        << "Failed to create Agent1 registration descriptor list";

    nixlBlobDesc desc1;
    desc1.addr = reinterpret_cast<uintptr_t>(dramBuffer1.data());
    desc1.len = dramBuffer1.size();
    desc1.devId = 0; // Device ID 0 for DRAM
    desc1.metaInfo = "";
    regDlist1->addDesc(desc1);
    ASSERT_EQ(regDlist1->descCount(), 1);

    // 6. Register Agent1 Memory (Done in test case or here if needed globally)
    // For this test structure, we'll register within the test case itself
    // to pair it clearly with Agent2's registration.
  }

  void TearDown() override {
    nixl_status_t deregStatus = nixl_status_t::NIXL_SUCCESS;
    // Cleanup: deregister memory (best effort in TearDown)
    if (agent1 && regDlist1 &&
        !regDlist1->isEmpty() /*&& (*regDlist1)[0].metadataP != nullptr*/) {
      // Only attempt dereg if metadata exists (meaning registration likely
      // succeeded)
      deregStatus = agent1->deregisterMem(*regDlist1, &registrationArgs1);
      if (deregStatus != nixl_status_t::NIXL_SUCCESS)
        std::cerr
            << "[ WARNING ] Agent1 failed to deregister memory in TearDown: "
            << nixlEnumStrings::statusStr(deregStatus) << std::endl;
    }
    // agent1 unique_ptr will handle deletion. Backend handles are managed by
    // agent.
  }
};

TEST_F(NixlUcxIntegrationTest, DramTransfer) {
  nixl_status_t status;

  // === Agent 1 Memory Registration (from fixture) ===
  status = agent1->registerMem(*regDlist1, &registrationArgs1);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 1 failed to register DRAM: "
      << nixlEnumStrings::statusStr(status);
  // ASSERT_NE((*regDlist1)[0].metadataP, nullptr)
  //     << "Agent 1 metadata pointer null after registration";

  // === Agent 2 Setup ===
  std::string agentName2 = "TestAgent_Ucx2";
  // Use same config as agent1 for simplicity
  nixlAgentConfig agentConfig2(/*use_prog_thread=*/false,
                               /*use_listen_thread=*/true,
                               /*port=*/kAgent2Port);
  auto agent2 = std::make_unique<nixlAgent>(agentName2, agentConfig2);
  ASSERT_NE(agent2, nullptr) << "Agent 2 creation failed";

  nixlBackendH *ucxBackend2 = nullptr;
  nixl_b_params_t backendParams2;
  nixl_mem_list_t mems2; // Can be used to get/set params if needed
  ASSERT_EQ(agent2->getPluginParams("UCX", mems2, backendParams2),
            nixl_status_t::NIXL_SUCCESS)
      << "Failed to get UCX plugin parameters for Agent2";

  status = agent2->createBackend("UCX", backendParams2, ucxBackend2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 2 failed to create UCX backend: "
      << nixlEnumStrings::statusStr(status);
  ASSERT_NE(ucxBackend2, nullptr) << "Agent 2 UCX backend handle is null";

  nixl_opt_args_t registrationArgs2;
  registrationArgs2.backends.push_back(ucxBackend2);

  // === Agent 2 Memory Setup ===
  std::vector<char> dramBuffer2(bufferSize);
  auto regDlist2 = std::make_unique<nixl_reg_dlist_t>(nixl_mem_t::DRAM_SEG);
  ASSERT_NE(regDlist2, nullptr) << "Failed to create Agent 2 reg dlist";

  nixlBlobDesc desc2;
  desc2.addr = reinterpret_cast<uintptr_t>(dramBuffer2.data());
  desc2.len = dramBuffer2.size();
  desc2.devId = 0; // DRAM device ID
  desc2.metaInfo = "";
  regDlist2->addDesc(desc2);

  status = agent2->registerMem(*regDlist2, &registrationArgs2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 2 failed to register DRAM: "
      << nixlEnumStrings::statusStr(status);
  // ASSERT_NE((*regDlist2)[0].metadataP, nullptr)
  //     << "Agent 2 metadata pointer null after registration";

  // === Metadata Exchange ===
  std::string meta1, meta2;
  std::string loadedAgentName1, loadedAgentName2;

  status = agent1->getLocalMD(meta1);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS) << "Agent 1 getLocalMD failed";
  status = agent2->getLocalMD(meta2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS) << "Agent 2 getLocalMD failed";

  ASSERT_FALSE(meta1.empty()) << "Agent 1 metadata is empty";
  ASSERT_FALSE(meta2.empty()) << "Agent 2 metadata is empty";

  status = agent1->loadRemoteMD(meta2, loadedAgentName2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 1 loadRemoteMD(Agent2) failed: "
      << nixlEnumStrings::statusStr(status);
  ASSERT_EQ(loadedAgentName2, agentName2);

  status = agent2->loadRemoteMD(meta1, loadedAgentName1);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 2 loadRemoteMD(Agent1) failed: "
      << nixlEnumStrings::statusStr(status);
  ASSERT_EQ(loadedAgentName1, agentName1);

  // === Prepare Buffers for Transfer ===
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
  nixlBasicDesc regSrc;
  regSrc.addr = reinterpret_cast<uintptr_t>(dramBuffer1.data()) + srcOffset;
  regSrc.len = transferSize;
  regSrc.devId = 0;
  regSrcDescs.addDesc(regSrc);

  nixl_xfer_dlist_t regDstDescs(nixl_mem_t::DRAM_SEG);
  nixlBasicDesc regDst;
  regDst.addr = reinterpret_cast<uintptr_t>(dramBuffer2.data()) + dstOffset;
  regDst.len = transferSize;
  regDst.devId = 0;
  regDstDescs.addDesc(regDst);

  // === Create and Execute Transfer (Agent 1 writes to Agent 2) ===
  nixlXferReqH *reqHandle = nullptr;
  nixl_opt_args_t xferArgs; // Use separate optional args for transfer
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  // We don't need to specify backends here if only one (UCX) is relevant

  status = agent1->createXferReq(nixl_xfer_op_t::NIXL_WRITE, regSrcDescs,
                                 regDstDescs, agentName2, reqHandle, &xferArgs);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "createXferReq failed: " << nixlEnumStrings::statusStr(status);
  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";

  status = agent1->postXferReq(reqHandle);
  // Post might return IN_PROG or SUCCESS depending on backend/timing
  ASSERT_TRUE(status == nixl_status_t::NIXL_SUCCESS ||
              status == nixl_status_t::NIXL_IN_PROG)
      << "postXferReq failed: " << nixlEnumStrings::statusStr(status);

  // === Poll for Completion ===
  nixl_notifs_t notifMap;
  nixl_status_t checkStatus = nixl_status_t::NIXL_IN_PROG;
  bool notifReceived = false;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10); // 10 second timeout

  while ((checkStatus != nixl_status_t::NIXL_SUCCESS || !notifReceived) &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    if (checkStatus != nixl_status_t::NIXL_SUCCESS) {
      checkStatus = agent1->getXferStatus(reqHandle);
      ASSERT_NE(checkStatus, nixl_status_t::NIXL_ERR_UNKNOWN)
          << "getXferStatus returned error"; // Check for errors other than
                                             // IN_PROG
    }

    if (!notifReceived) {
      status = agent2->getNotifs(notifMap);
      ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
          << "getNotifs failed: " << nixlEnumStrings::statusStr(status);
      if (notifMap.count(agentName1) > 0 && !notifMap[agentName1].empty()) {
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
  ASSERT_EQ(notifMap.count(agentName1), 1) << "Notification map missing agent1";
  ASSERT_EQ(notifMap[agentName1].size(), 1)
      << "Incorrect number of notifications received";
  ASSERT_EQ(notifMap[agentName1].front(), xferArgs.notifMsg)
      << "Notification message mismatch";

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

  // === Cleanup ===
  status = agent1->releaseXferReq(reqHandle);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "releaseXferReq failed: " << nixlEnumStrings::statusStr(status);

  // Deregister Agent 2's memory
  status = agent2->deregisterMem(*regDlist2, &registrationArgs2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 2 deregisterMem failed: " << nixlEnumStrings::statusStr(status);

  // Invalidate metadata loaded by Agent 1
  status = agent1->invalidateRemoteMD(agentName2);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 1 invalidateRemoteMD(Agent2) failed: "
      << nixlEnumStrings::statusStr(status);

  // Invalidate metadata loaded by Agent 2
  status = agent2->invalidateRemoteMD(agentName1);
  ASSERT_EQ(status, nixl_status_t::NIXL_SUCCESS)
      << "Agent 2 invalidateRemoteMD(Agent1) failed: "
      << nixlEnumStrings::statusStr(status);

  // Agent 1's memory is deregistered by the fixture's TearDown
  // agent2 and regDlist2 unique_ptrs will clean up automatically
}
