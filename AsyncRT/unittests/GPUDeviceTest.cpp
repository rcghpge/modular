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

class GPUDeviceTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      return; // Let individual tests handle errors/skips.

    std::cerr << "##### GPUDeviceTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }
};

// EnablePeerAccess is idempotent for the active GPU backend (CUDA/HIP/Metal).
TEST_F(GPUDeviceTest, EnablePeerAccessIdempotent) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  if (num <= 1) {
    GTEST_SKIP() << "Need >= 2 GPU devices for peer access idempotency";
  }

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

// Enabling peer access to self must fail on the active GPU backend.
TEST_F(GPUDeviceTest, EnablePeerAccessSelfErrors) {
  if (Device::numberOfDevices(Device::gpuAPI) < 1) {
    GTEST_SKIP() << "No GPU device present";
  }

  ASSIGN_NO_ERR(DeviceRef, d0, Device::create(Device::gpuAPI, 0));
  EXPECT_ERR_MSG(d0->enablePeerAccess(*d0), "Cannot enable peer access");
}

} // namespace
