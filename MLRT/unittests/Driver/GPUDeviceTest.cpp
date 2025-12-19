//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
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

// Enabling peer access to self must fail on the active GPU backend.
TEST_F(GPUDeviceTest, EnablePeerAccessSelfErrors) {
  ASSIGN_NO_ERR(DeviceRef, d0, Device::create(Device::gpuAPI, 0));
  // Skip test for Metal
  if (d0->api() == Device::metalAPI) {
    GTEST_SKIP() << "Metal not supported";
  }
  EXPECT_ERR_MSG(d0->enablePeerAccess(*d0), "Cannot enable peer access");
}

} // namespace
