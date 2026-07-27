//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "HAL/M_driver_api.h"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <iostream>
#include <string>

using std::cerr;
using std::endl;

// Macro for interim lines in a test.
// These build up to the items being tested but
// themselves should have been previously tested.
// So this construct hides their `EXPECT_*()` so that
// the test code only shows items of interest.
#define RUN(stmt)                                                              \
  do {                                                                         \
    M_driver_status_t status = (this->stmt);                                   \
    ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS);                                \
  } while (0)

#define DO_TEST(stmt)                                                          \
  do {                                                                         \
    M_driver_status_t status = (this->stmt);                                   \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

// The ASSERT_SUCCESS helper macro should be used when calling driver entry
// points when the expected result is M_DRIVER_STATUS_SUCCESS and execution
// should NOT continue if the call fails. This is should be preferred over
// EXPECT_SUCCESS by default.
#define ASSERT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    M_driver_status_t status = (EXPR);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

// The EXPECT_SUCCESS helper macro should be used when calling driver entry
// points when the expected result is M_DRIVER_STATUS_SUCCESS and test
// execution should continue if the call fails. Only use instead of
// ASSERT_SUCCESS if there is resource cleanup or other processing required
// after the calling the driver entrypoint.
#define EXPECT_SUCCESS(EXPR)                                                   \
  do {                                                                         \
    M_driver_status_t status = (EXPR);                                         \
    if (status != M_DRIVER_STATUS_SUCCESS)                                     \
      EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS) << getStatusMessage(status);  \
  } while (0)

struct DriverPluginTest : ::testing::Test {
  void SetUp() override {
    const char *pluginSpec = std::getenv("MODULAR_DRIVER_PLUGINS");
    if (pluginSpec == nullptr) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    llvm::StringRef soPath(pluginSpec);
    if (soPath.empty()) {
      GTEST_SKIP() << "MODULAR_DRIVER_PLUGINS environment variable is not set";
      return;
    }
    soHandle = dlopen(pluginSpec, RTLD_NOW);
    ASSERT_TRUE(soHandle != nullptr);

#define M_DRIVER_HAL_SYMBOL(name)                                              \
  name = (M_driver_##name##_func_ptr_t)dlsym(                                  \
      soHandle, ("M_driver_" + std::string(#name)).c_str());                   \
  ASSERT_TRUE(name != nullptr) << "Failed to resolve M_driver_" #name;

#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
    M_driver_version version = {.major = M_DRIVER_INTERFACE_VERSION_MAJOR,
                                .minor = M_DRIVER_INTERFACE_VERSION_MINOR,
                                .patch = M_DRIVER_INTERFACE_VERSION_PATCH};
    DO_TEST(create(&version, &driver));
    ASSERT_EQ(version.major, M_DRIVER_INTERFACE_VERSION_MAJOR);
    ASSERT_EQ(version.minor, M_DRIVER_INTERFACE_VERSION_MINOR);

    const char *apiStr = nullptr;
    M_driver_status_t status = property(driver, M_DRIVER_PROPERTY_API, &apiStr);
    if (status == M_DRIVER_STATUS_SUCCESS && apiStr)
      api = apiStr;
  }

  void TearDown() override {
    if (soHandle == nullptr)
      return;
    DO_TEST(destroy(driver));
#define M_DRIVER_HAL_SYMBOL(name) name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
    dlclose(soHandle);
  }

  std::string getStatusMessage(M_driver_status_t status) {
    char errorMessage[1024];
    M_driver_status_t msgStatus = this->status_message(
        driver, status, errorMessage, sizeof(errorMessage));
    if (msgStatus != M_DRIVER_STATUS_SUCCESS)
      return std::string("Failed to get status message");
    return std::string(errorMessage);
  }

  void *soHandle = nullptr;
#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  M_driver_driver_t driver = nullptr;

  llvm::StringRef api;
};

struct DriverPluginDeviceTest : DriverPluginTest {
  void SetUp() override {
    DriverPluginTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    int64_t deviceCount = 0;
    ASSERT_SUCCESS(device_count(driver, &deviceCount));
    for (int64_t i = 0; i < deviceCount; i++) {
      M_driver_device_t device = nullptr;
      EXPECT_SUCCESS(device_get(driver, i, &device));
      devices.push_back(device);
    }
  }

  std::vector<M_driver_device_t> devices;
};

struct DriverPluginContextTest : DriverPluginDeviceTest {
  void SetUp() override {
    DriverPluginDeviceTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (M_driver_device_t device : devices) {
      M_driver_context_t context = nullptr;
      ASSERT_SUCCESS(context_create(device, &context));
      contexts.push_back(context);
    }
  }

  void TearDown() override {
    for (M_driver_context_t context : contexts) {
      EXPECT_SUCCESS(context_destroy(context));
    }
    DriverPluginTest::TearDown();
  }

  std::vector<M_driver_context_t> contexts;
};

struct DriverPluginQueueTest : DriverPluginContextTest {
  void SetUp() override {
    DriverPluginContextTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    for (M_driver_context_t context : contexts) {
      M_driver_queue_t queue;
      ASSERT_SUCCESS(queue_create(context, &queue));
      queues.push_back(queue);
    }
  }

  void TearDown() override {
    for (size_t i = 0; i < queues.size(); i++)
      EXPECT_SUCCESS(queue_destroy(contexts[i], queues[i]));
    DriverPluginContextTest::TearDown();
  }

  std::vector<M_driver_queue_t> queues;
};
