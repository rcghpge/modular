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
using namespace M::AsyncRT::NIXL;
using namespace M::AsyncRT::NIXL::Test;

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
