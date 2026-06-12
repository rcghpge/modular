//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/unittests/Driver/NIXL/TestCommon.h"

#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <chrono>
#include <netinet/in.h>
#include <random>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace M::MLRT {
std::ostream &operator<<(std::ostream &os, nixl_status_t status) {
  return os << NixlEnumStrings::statusStr(status);
}
} // namespace M::MLRT

namespace M::MLRT::Test {

void NixlAgentWithUCXBackend::registerMemory(nixl_mem_t memType,
                                             nixlBlobDesc desc) {
  nixl_reg_dlist_t descList(memType);
  descList.addDesc(desc);
  nixl_opt_args_t regArgs;
  regArgs.backends = {backend};
  ASSERT_SUCCESS(agent->registerMem(descList, &regArgs));
}

void NixlAgentWithUCXBackend::create(NixlAgentWithUCXBackend &self,
                                     std::string agentName, int port,
                                     std::string backendName, int gpuDeviceId,
                                     int numWorkers,
                                     std::string errorHandlingMode,
                                     int numThreads) {
  self.agentName = agentName;
  self.agent = std::make_unique<nixlAgent>(
      agentName, nixlAgentConfig(/*use_prog_thread=*/true,
                                 /*use_listen_thread=*/true,
                                 /*port=*/port));
  std::vector<nixl_backend_t> plugins;
  ASSERT_SUCCESS(self.agent->getAvailPlugins(plugins));
  auto ucxFound =
      0 < std::count_if(plugins.begin(), plugins.end(),
                        [&](const nixl_backend_t &p) {
                          return llvm::StringRef(p).lower() == backendName;
                        });
  ASSERT_TRUE(ucxFound) << "Plugin not found: " << backendName;

  nixl_b_params_t backendParams;
  nixl_mem_list_t mems;
  ASSERT_SUCCESS(self.agent->getPluginParams(backendName, mems, backendParams));

  if (gpuDeviceId != -1) {
    backendParams["gpu_device_id"] = std::to_string(gpuDeviceId);
  }
  if (numWorkers > 1) {
    backendParams["num_workers"] = std::to_string(numWorkers);
  }
  if (!errorHandlingMode.empty()) {
    backendParams["ucx_error_handling_mode"] = errorHandlingMode;
  }
  if (numThreads > 0) {
    backendParams["num_threads"] = std::to_string(numThreads);
  }

  ASSERT_SUCCESS(
      self.agent->createBackend(backendName, backendParams, self.backend));
  ASSERT_NE(self.backend, nullptr) << "Backend handle is null";
}

int available_port(int startPort, int endPort, int maxAttempts) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(startPort, endPort);

  for (int i = 0; i < maxAttempts; ++i) {
    int port = dis(gen);
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      continue;
    }

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

bool wait_until_true(std::function<bool()> func, int retries) {
  bool result;
  while (!(result = func()) && retries-- > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return result;
}

void exchangeMetadata(NixlAgentWithUCXBackend &nixl1,
                      NixlAgentWithUCXBackend &nixl2) {
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
                  nixl_xfer_dlist_t &dstDescs, nixl_xfer_op_t op) {
  nixlXferReqH *reqHandle = nullptr;
  nixlAgentOptionalArgs xferArgs;
  xferArgs.notifMsg = "notification";
  xferArgs.hasNotif = true;
  ASSERT_SUCCESS(nixl1.agent->createXferReq(
      op, srcDescs, dstDescs, nixl2.agentName, reqHandle, &xferArgs));

  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  if (xferStatus != nixl_status_t::NIXL_SUCCESS &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq failed: " << NixlEnumStrings::statusStr(xferStatus);
  }

  nixl_notifs_t notifMap;
  nixl_status_t checkStatus = nixl_status_t::NIXL_IN_PROG;
  bool notifReceived = false;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(10);

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
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, nixl_status_t::NIXL_SUCCESS)
      << "Transfer did not complete within timeout";
  ASSERT_TRUE(notifReceived) << "Notification not received within timeout";
  ASSERT_EQ(notifMap.count(nixl1.agentName), 1)
      << "Notification map missing agent1";
  ASSERT_EQ(notifMap[nixl1.agentName].size(), 1)
      << "Incorrect number of notifications received";
}

void transferDataWithFailure(NixlAgentWithUCXBackend &nixl1,
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

  ASSERT_NE(reqHandle, nullptr) << "Transfer handle is null";
  nixl_status_t xferStatus = nixl1.agent->postXferReq(reqHandle);
  if (xferStatus != expectedStatus &&
      xferStatus != nixl_status_t::NIXL_IN_PROG) {
    FAIL() << "postXferReq returned unexpected status: "
           << NixlEnumStrings::statusStr(xferStatus);
  }

  nixl_status_t checkStatus = xferStatus;
  const auto startTime = std::chrono::steady_clock::now();
  const auto timeout = std::chrono::seconds(30);

  while (checkStatus == nixl_status_t::NIXL_IN_PROG &&
         (std::chrono::steady_clock::now() - startTime < timeout)) {
    checkStatus = nixl1.agent->getXferStatus(reqHandle);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  ASSERT_EQ(checkStatus, expectedStatus)
      << "Transfer did not complete with expected status. Got: "
      << NixlEnumStrings::statusStr(checkStatus)
      << ", Expected: " << NixlEnumStrings::statusStr(expectedStatus);

  nixl1.agent->releaseXferReq(reqHandle);
}

} // namespace M::MLRT::Test
