//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "llvm/Support/Process.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <iostream>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

class MultiGPUDeviceTest : public ::testing::Test {};

TEST_F(MultiGPUDeviceTest, DeviceVisibilityControl) {
  std::string const &arch = Device::getAcceleratorArchOrEmpty();
  const char *envVarName = nullptr;
  if (arch.find("nvidia") != std::string::npos) {
    envVarName = "CUDA_VISIBLE_DEVICES";
  } else if (arch.find("amd") != std::string::npos) {
    // Multiple options for AMD, we only support one
    // https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html
    envVarName = "HIP_VISIBLE_DEVICES";
  } else {
    return;
  }
  std::optional<std::string> prevEnvVarValue =
      llvm::sys::Process::GetEnv(envVarName);
  unsetenv(envVarName);
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 1) << "Should only run on multi-gpu machines";
  int const allDeviceCount = num;

  setenv(envVarName, "0", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 1);

  setenv(envVarName, "1,0", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 2);

  setenv(envVarName, "", 1 /*overwrite*/);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, 0) << "Empty string => 0 devices as per CUDA specs";
  // ... we extend this to HIP too

  // Double check
  unsetenv(envVarName);
  num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_EQ(num, allDeviceCount);

  // Restore original setting
  if (prevEnvVarValue) {
    setenv(envVarName, prevEnvVarValue->c_str(), 1 /*overwrite*/);
  }
}

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
