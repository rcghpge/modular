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
    driver = driverOr.takeValue();
    auto apiOr = driver->getProperty("api");
    if (!apiOr.isError())
      api = *apiOr;
  }

  void TearDown() override { driver.reset(); }

  DriverRef driver;
  llvm::StringRef api;
};

TEST_F(DriverHALTest, SignOfLife) {
  cerr << "Driver plugin name = " << driver->getName()->str() << endl;
  cerr << "Driver plugin path = " << driver->getPluginPath()->str() << endl;
}

TEST_F(DriverHALTest, DeviceCount) {
  ErrorOr<int64_t> deviceCountOr = driver->getDeviceCount();
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

    ASSIGN_NO_ERR(int64_t, deviceCount, driver->getDeviceCount());
    for (int64_t i = 0; i < deviceCount; i++) {
      ErrorOr<DeviceRef> deviceOr = driver->getDevice(i);
      ASSERT_TRUE(!deviceOr.isError());
      devices.push_back(deviceOr.takeValue());
    }
  }

  std::vector<DeviceRef> devices;
};

TEST_F(DriverHALDeviceTest, DeviceGet) {
  for (size_t i = 0; i < devices.size(); i++) {
    DeviceRef &device = devices[i];
    ErrorOr<int64_t> idOr = device->getId();
    ASSERT_TRUE(!idOr.isError());
    int64_t id = idOr.takeValue();
    ASSERT_EQ(id, i);
    ErrorOr<llvm::StringRef> nameOr = device->getName();
    ASSERT_TRUE(!nameOr.isError());
    llvm::StringRef name = nameOr.takeValue();
    cerr << "Retrieved device named '" << name.str() << "' with ID " << id
         << endl;
    ASSERT_EQ(name.str(), "Device #" + std::to_string(id));
  }
}

TEST_F(DriverHALTest, PropertyApi) {
  ErrorOr<llvm::StringRef> apiOr = driver->getProperty("api");
  ASSERT_FALSE(apiOr.isError()) << apiOr.getError();
  ASSERT_NE(0, apiOr->size());
  std::cerr << apiOr->str() << "\n";
}

TEST_F(DriverHALDeviceTest, DeviceCreateContext) {
  for (DeviceRef &device : devices) {
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

    for (DeviceRef &device : devices) {
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
  for (ContextRef &context : contexts) {
    ErrorOr<BufferRef> bufferOr = context->allocBufferPinned(/*byteSize=*/128);
    ASSERT_FALSE(bufferOr.isError());
  }
}

TEST_F(DriverHALContextTest, ContextAllocBufferSYNC) {
  for (ContextRef &context : contexts) {
    ErrorOr<BufferRef> bufferOr = context->allocBufferSync(/*byteSize=*/128);
    ASSERT_FALSE(bufferOr.isError());
  }
}

TEST_F(DriverHALContextTest, Queue) {
  if (api == "CUDA" || api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ErrorOr<QueueRef> queueOr = context->createQueue();
    ASSERT_FALSE(queueOr.isError());
  }
}

TEST_F(DriverHALContextTest, CreateEvent) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ErrorOr<EventRef> eventOr = context->createEvent();
    ASSERT_FALSE(eventOr.isError()) << eventOr.getError();
  }
}

TEST_F(DriverHALContextTest, EventSynchronize) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ASSIGN_NO_ERR(EventRef, event, context->createEvent());
    // Synchronizing an unrecorded event should succeed immediately.
    EXPECT_NO_ERR(event->synchronize());
  }
}

TEST_F(DriverHALContextTest, EventIsReady) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (ContextRef &context : contexts) {
    ASSIGN_NO_ERR(EventRef, event, context->createEvent());
    // An unrecorded event should report ready.
    ASSIGN_NO_ERR(bool, ready, event->isReady());
    EXPECT_TRUE(ready);
  }
}

struct DriverHALQueueTest : DriverHALContextTest {
  void SetUp() override {
    DriverHALContextTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    if (api == "CUDA" || api == "HIP" || api == "Metal")
      GTEST_SKIP() << "Unimplemented for " << api;

    for (ContextRef &context : contexts) {
      auto queueOr = context->createQueue();
      ASSERT_FALSE(queueOr.isError()) << queueOr.getError();
      queues.push_back(queueOr.takeValue());
    }
  }

  std::vector<QueueRef> queues;
};

TEST_F(DriverHALQueueTest, AllocBufferAsync) {
  if (api == "CUDA" || api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (QueueRef &queue : queues) {
    ErrorOr<BufferRef> bufferOr = queue->allocBufferAsync(/*byteSize=*/128);
    ASSERT_FALSE(bufferOr.isError());
  }
}

TEST_F(DriverHALQueueTest, CopyToDeviceThenFromDevice) {
  if (api == "CUDA" || api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unsupported on " << api;

  constexpr size_t size = 128;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = i;

  for (size_t i = 0; i < contexts.size(); i++) {
    ContextRef context = contexts[i];
    QueueRef queue = queues[i];

    auto bufferOr = context->allocBufferSync(size);
    ASSIGN_NO_ERR(BufferRef, buffer, context->allocBufferSync(size));
    M::ErrorOrSuccess errorOr =
        queue->copyToDevice(buffer, content.data(), size);
    ASSERT_FALSE(errorOr.isError()) << errorOr.getError();

    std::array<uint8_t, size> result;
    result.fill(0);
    errorOr = queue->copyFromDevice(result.data(), buffer, size);
    ASSERT_FALSE(errorOr.isError()) << errorOr.getError();
    ASSERT_EQ(content, result);
  }
}

TEST_F(DriverHALQueueTest, SetMemory) {
  if (api == "CUDA" || api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unsupported on " << api;

  constexpr size_t size = 128;

  for (size_t i = 0; i < contexts.size(); i++) {
    ContextRef context = contexts[i];
    QueueRef queue = queues[i];

    ASSIGN_NO_ERR(BufferRef, buffer, context->allocBufferSync(size));
    M::ErrorOrSuccess errorOr =
        queue->setMemory(buffer, uint8_t(42), sizeof(uint8_t));
    ASSERT_FALSE(errorOr.isError()) << errorOr.getError();

    std::array<uint8_t, size> result;
    result.fill(0);
    errorOr = queue->copyFromDevice(result.data(), buffer, size);
    ASSERT_FALSE(errorOr.isError()) << errorOr.getError();
    std::array<uint8_t, size> expected;
    expected.fill(42);
    ASSERT_EQ(expected, result);
  }
}
