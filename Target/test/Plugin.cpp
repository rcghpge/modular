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

class DriverPluginTest : public ::testing::Test {
protected:
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
    DO_TEST(create(&driver));

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

  const char *getStatusMessage(M_driver_status_t status) {
    const char *errorMessage = nullptr;
    (void)this->status_message(status, &errorMessage);
    return errorMessage;
  }

  void *soHandle = nullptr;
#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  M_driver_driver_t driver = nullptr;

  llvm::StringRef api;
};

TEST_F(DriverPluginTest, DriverPropertyAPI) {
  const char *api = nullptr;
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->property(driver, M_DRIVER_PROPERTY_API, &api));
  ASSERT_NE(nullptr, api);
  std::cerr << api << "\n";
}

TEST_F(DriverPluginTest, DeviceCount) {
  int64_t deviceCount = 0;
  M_driver_status_t status = this->device_count(driver, &deviceCount);
  EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS);
  cerr << "Device count: " << deviceCount << endl;
  EXPECT_GT(deviceCount, 0);
}

TEST_F(DriverPluginTest, DeviceGet) {
  int64_t deviceCount = 0;
  RUN(device_count(driver, &deviceCount));
  for (int64_t i = 0; i < deviceCount; i++) {
    M_driver_device_t device;
    M_driver_status_t status = this->device_get(driver, i, &device);
    EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS);
    cerr << "Got device 0x" << std::hex << device << endl;
  }
}

TEST_F(DriverPluginTest, DeviceContext) {
  int64_t deviceCount = 0;
  EXPECT_SUCCESS(device_count(driver, &deviceCount));
  ASSERT_GT(deviceCount, 0);
  std::vector<M_driver_context_t> contexts;
  for (int64_t i = 0; i < deviceCount; i++) {
    M_driver_device_t device = nullptr;
    RUN(device_get(driver, i, &device));
    M_driver_context_t context = nullptr;
    DO_TEST(context_create(driver, device, &context));
    cerr << "Created context for device 0x" << std::hex << device << ": 0x"
         << context << endl;
    contexts.push_back(context);
  }
  for (int64_t i = deviceCount - 1; i >= 0; i--) {
    DO_TEST(context_destroy(driver, contexts[i]));
    cerr << "Destroyed context 0x" << contexts[i] << endl;
  }
}

class DriverPluginContextTest : public DriverPluginTest {
public:
  void SetUp() override {
    DriverPluginTest::SetUp();
    if (this->HasFatalFailure() || this->IsSkipped())
      return;

    int64_t deviceCount = 0;
    ASSERT_SUCCESS(device_count(driver, &deviceCount));
    for (int64_t i = 0; i < deviceCount; i++) {
      M_driver_device_t device = nullptr;
      EXPECT_SUCCESS(device_get(driver, i, &device));
      M_driver_context_t context = nullptr;
      ASSERT_SUCCESS(context_create(driver, device, &context));
      contexts.push_back(context);
    }
  }

  void TearDown() override {
    for (M_driver_context_t context : contexts) {
      EXPECT_SUCCESS(context_destroy(driver, context));
    }
    DriverPluginTest::TearDown();
  }

  std::vector<M_driver_context_t> contexts;
};

TEST_F(DriverPluginContextTest, MemoryHost) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t context : contexts) {
    void *memory;
    ASSERT_SUCCESS(
        memory_alloc(context, M_DRIVER_ALLOC_MODE_HOST, 128, &memory));
    ASSERT_SUCCESS(memory_free(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemorySync) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t context : contexts) {
    void *memory;
    ASSERT_SUCCESS(
        memory_alloc(context, M_DRIVER_ALLOC_MODE_SYNC, 128, &memory));
    ASSERT_SUCCESS(memory_free(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAsync) {
  if (api == "CUDA" || api == "HIP")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t context : contexts) {
    void *memory;
    ASSERT_SUCCESS(
        memory_alloc(context, M_DRIVER_ALLOC_MODE_ASYNC, 128, &memory));
    ASSERT_SUCCESS(memory_free(context, memory));
  }
}
