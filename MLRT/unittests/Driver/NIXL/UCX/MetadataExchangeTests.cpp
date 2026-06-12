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
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer0,
                ctx0->enqueueCreateBuffer(bufferSize));
  EXPECT_NO_ERR(deviceBuffer0->context().synchronize()); // Need its ptr() below

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
  ASSIGN_NO_ERR(DeviceBufferRef, deviceBuffer0,
                ctx0->enqueueCreateBuffer(bufferSize));
  EXPECT_NO_ERR(deviceBuffer0->context().synchronize()); // Need its ptr() below

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
