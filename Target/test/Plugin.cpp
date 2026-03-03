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
    EXPECT_EQ(status, M_DRIVER_STATUS_SUCCESS);                                \
  } while (0)

#define DO_TEST(stmt)                                                          \
  do {                                                                         \
    M_driver_status_t status = (this->stmt);                                   \
    if (status != M_DRIVER_STATUS_SUCCESS) {                                   \
      const char *errorMessage = nullptr;                                      \
      (void)this->status_message(status, &errorMessage);                       \
      ASSERT_EQ(status, M_DRIVER_STATUS_SUCCESS) << errorMessage;              \
    }                                                                          \
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

  void *soHandle = nullptr;
#define M_DRIVER_HAL_SYMBOL(name) M_driver_##name##_func_ptr_t name = nullptr;
#include "HAL/M_driver_hal_symbols.inc"
#undef M_DRIVER_HAL_SYMBOL
  M_driver_driver_t driver = 0;
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
  RUN(device_count(driver, &deviceCount));
  std::vector<M_driver_context_t> contexts;
  for (int64_t i = 0; i < deviceCount; i++) {
    M_driver_device_t device = 0x0ULL;
    RUN(device_get(driver, i, &device));
    M_driver_context_t context = 0x0ULL;
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
