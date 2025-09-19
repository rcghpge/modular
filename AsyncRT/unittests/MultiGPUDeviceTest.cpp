//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

class MultiGPUDeviceTest : public ::testing::Test {};

// EnablePeerAccess is idempotent for the active GPU backend (CUDA/HIP/Metal).
TEST_F(MultiGPUDeviceTest, EnablePeerAccessIdempotent) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 1) << "Should only run on multi-gpu machines";

  bool foundPair = false;
  for (int i = 0; i < num && !foundPair; ++i) {
    ASSIGN_NO_ERR(DeviceRef, di, Device::create(Device::gpuAPI, i));
    for (int j = i + 1; j < num && !foundPair; ++j) {
      ASSIGN_NO_ERR(DeviceRef, dj, Device::create(Device::gpuAPI, j));

      ASSIGN_NO_ERR(bool, canIJ, di->canAccess(*dj));
      ASSIGN_NO_ERR(bool, canJI, dj->canAccess(*di));
      if (!canIJ || !canJI)
        continue;

      EXPECT_NO_ERR(di->enablePeerAccess(*dj));
      EXPECT_NO_ERR(dj->enablePeerAccess(*di));
      // Idempotency: repeat should also succeed.
      EXPECT_NO_ERR(di->enablePeerAccess(*dj));
      EXPECT_NO_ERR(dj->enablePeerAccess(*di));

      foundPair = true;
    }
  }

  if (!foundPair) {
    GTEST_SKIP() << "No GPU device pair with mutual P2P access; skipping.";
  }
}

} // namespace
