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
    auto apiOr = driver_->property("api");
    if (!apiOr.isError())
      api = *apiOr;
  }

  void TearDown() override { driver_.reset(); }

  DriverRef driver_;
  llvm::StringRef api;
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

class DriverHALDeviceTest : public DriverHALTest {
protected:
  void SetUp() override {
    DriverHALTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    ASSIGN_NO_ERR(int64_t, deviceCount, driver_->deviceCount());
    for (int64_t i = 0; i < deviceCount; i++) {
      ErrorOr<DeviceRef> deviceOr = driver_->device(i);
      ASSERT_TRUE(!deviceOr.isError());
      devices_.push_back(deviceOr.takeValue());
    }
  }

  std::vector<DeviceRef> devices_;
};

TEST_F(DriverHALDeviceTest, DeviceGet) {
  for (size_t i = 0; i < devices_.size(); i++) {
    DeviceRef &device = devices_[i];
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
}

TEST_F(DriverHALTest, PropertyApi) {
  ErrorOr<llvm::StringRef> apiOr = driver_->property("api");
  ASSERT_FALSE(apiOr.isError()) << apiOr.getError();
  ASSERT_NE(0, apiOr->size());
  std::cerr << apiOr->str() << "\n";
}

TEST_F(DriverHALDeviceTest, DeviceCreateContext) {
  for (DeviceRef &device : devices_) {
    ErrorOr<ContextRef> contextOr = device->createContext();
    ASSERT_TRUE(!contextOr.isError()) << contextOr.getError();
    ContextRef context = contextOr.takeValue();
  }
}

class DriverHALContextTest : public DriverHALDeviceTest {
protected:
  void SetUp() override {
    DriverHALDeviceTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (DeviceRef &device : devices_) {
      ErrorOr<ContextRef> contextOr = device->createContext();
      ASSERT_TRUE(!contextOr.isError()) << contextOr.getError();
      contexts.push_back(contextOr.takeValue());
    }
  }

  void TearDown() override {
    contexts.clear();
    DriverHALDeviceTest::TearDown();
  }

  std::vector<ContextRef> contexts;
};

TEST_F(DriverHALContextTest, ContextAllocBufferHOST) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ErrorOr<BufferRef> bufferOr =
        context->allocBuffer(/*byteSize=*/128, Buffer::AllocMode::HOST);
    ASSERT_FALSE(bufferOr.isError());
  }
}

TEST_F(DriverHALContextTest, ContextAllocBufferSYNC) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ErrorOr<BufferRef> bufferOr =
        context->allocBuffer(/*byteSize=*/128, Buffer::AllocMode::SYNC);
    ASSERT_FALSE(bufferOr.isError());
  }
}

TEST_F(DriverHALContextTest, ContextAllocBufferASYNC) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ErrorOr<BufferRef> bufferOr =
        context->allocBuffer(/*byteSize=*/128, Buffer::AllocMode::ASYNC);
    ASSERT_FALSE(bufferOr.isError());
  }
}
