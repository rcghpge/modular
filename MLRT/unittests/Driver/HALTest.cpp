//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
#include "gmock/gmock.h"

#include "MLRT/Driver/HAL/Context.h"
#include "MLRT/Driver/HAL/Device.h"
#include "MLRT/Driver/HAL/Driver.h"

#include "TestUtils.h"

#include <cstdlib>
#include <iostream>

using namespace M::Driver::HAL;
using M::ErrorOr;
using std::cerr;
using std::endl;

class DriverHALTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *const pluginSpec = std::getenv("MODULAR_DRIVER_PLUGINS");
    if (pluginSpec == nullptr) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    std::string const soPath(pluginSpec);
    if (soPath.empty()) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    ErrorOr<DriverRef> driverOr = Driver::create(soPath);
    ASSERT_TRUE(!driverOr.isError()) << driverOr.getError();
    driver_ = driverOr.takeValue();
  }

  void TearDown() override { driver_.reset(); }

  DriverRef driver_;
};

TEST_F(DriverHALTest, SignOfLife) {
  cerr << "Driver plugin name = " << driver_->name()->str() << endl;
  cerr << "Driver plugin path = " << driver_->pluginPath()->str() << endl;
}

TEST_F(DriverHALTest, DeviceCount) {
  ErrorOr<int64_t> deviceCountOr = driver_->deviceCount();
  ASSERT_TRUE(!deviceCountOr.isError());
  int64_t deviceCount = deviceCountOr.takeValue();
  cerr << "Device count: " << deviceCount << endl;
  ASSERT_GT(deviceCount, 0);
}

TEST_F(DriverHALTest, DeviceGet) {
  ASSIGN_NO_ERR(int64_t, deviceCount, driver_->deviceCount());
  for (int64_t i = 0; i < deviceCount; i++) {
    ErrorOr<DeviceRef> deviceOr = driver_->device(i);
    ASSERT_TRUE(!deviceOr.isError());
    DeviceRef device = deviceOr.takeValue();
    ErrorOr<int64_t> idOr = device->id();
    ASSERT_TRUE(!idOr.isError());
    int64_t id = idOr.takeValue();
    ASSERT_EQ(id, i);
    ErrorOr<llvm::StringRef> nameOr = device->name();
    ASSERT_TRUE(!nameOr.isError());
    llvm::StringRef name = nameOr.takeValue();
    cerr << "Retrieved device named '" << name.str() << "' with ID " << id
         << endl;
    ASSERT_EQ(name.str(), "Device #" + std::to_string(id));
  }

  //   ErrorOr<ContextRef> context = device->createContext();
}
