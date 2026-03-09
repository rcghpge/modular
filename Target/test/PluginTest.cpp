//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Fixture.h"

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
    DO_TEST(context_create(device, &context));
    cerr << "Created context for device 0x" << std::hex << device << ": 0x"
         << context << endl;
    contexts.push_back(context);
  }
  for (int64_t i = deviceCount - 1; i >= 0; i--) {
    DO_TEST(context_destroy(contexts[i]));
    cerr << "Destroyed context 0x" << contexts[i] << endl;
  }
}

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

TEST_F(DriverPluginContextTest, QueueCreateDestroy) {
  if (api == "CUDA" || api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}
