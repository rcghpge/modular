//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Fixture.h"
#include <cstring>

TEST_F(DriverPluginTest, StandardStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_SUCCESS, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_UNKNOWN_ERROR, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);
}

TEST_F(DriverPluginTest, BadStatusCode) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(10, errorMessage, sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(-M_DRIVER_NUM_STANDARD_STATUS_CODES,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(M_DRIVER_STATUS_CUSTOM_CODES_BASE + 1,
                                 errorMessage, sizeof(errorMessage)));
  // Conservative number that we expect no plugin to reach or exceed
  int64_t constexpr assumedNumCustomStatusCodes = 50000;
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(M_DRIVER_STATUS_CUSTOM_CODES_BASE -
                                     assumedNumCustomStatusCodes,
                                 errorMessage, sizeof(errorMessage)));
}

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

TEST_F(DriverPluginContextTest, MemoryAllocSync) {
  for (M_driver_context_t context : contexts) {
    void *memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_sync(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocPinned) {
  for (M_driver_context_t context : contexts) {
    void *memory;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_pinned(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocReturnsNonNull) {
  for (M_driver_context_t context : contexts) {
    void *memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    EXPECT_NE(memory, nullptr);
    ASSERT_SUCCESS(memory_free_sync(context, memory));

    memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 128, &memory));
    EXPECT_NE(memory, nullptr);
    ASSERT_SUCCESS(memory_free_pinned(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocZeroSize) {
  for (M_driver_context_t context : contexts) {
    void *memory = reinterpret_cast<void *>(-1);
    ASSERT_SUCCESS(memory_alloc_pinned(context, 0, &memory));
    EXPECT_EQ(memory, nullptr);

    memory = reinterpret_cast<void *>(-1);
    ASSERT_SUCCESS(memory_alloc_sync(context, 0, &memory));
    EXPECT_EQ(memory, nullptr);
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocBadContext) {
  EXPECT_EQ(memory_alloc_sync(nullptr, 128, nullptr),
            M_DRIVER_STATUS_BAD_CONTEXT);
  EXPECT_EQ(memory_alloc_pinned(nullptr, 128, nullptr),
            M_DRIVER_STATUS_BAD_CONTEXT);
}

TEST_F(DriverPluginContextTest, MemoryAllocNullOut) {
  for (M_driver_context_t context : contexts) {
    EXPECT_EQ(memory_alloc_sync(context, 128, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
    EXPECT_EQ(memory_alloc_pinned(context, 128, nullptr),
              M_DRIVER_STATUS_INVALID_ARG);
  }
}

TEST_F(DriverPluginContextTest, MemoryFreeNull) {
  for (M_driver_context_t context : contexts) {
    ASSERT_SUCCESS(memory_free_pinned(context, nullptr));
    ASSERT_SUCCESS(memory_free_sync(context, nullptr));
  }
}

TEST_F(DriverPluginContextTest, MemoryFreeBadContext) {
  EXPECT_EQ(memory_free_pinned(nullptr, nullptr), M_DRIVER_STATUS_BAD_CONTEXT);
  EXPECT_EQ(memory_free_sync(nullptr, nullptr), M_DRIVER_STATUS_BAD_CONTEXT);
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

TEST_F(DriverPluginContextTest, EventCreateDestroy) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t ctx : contexts) {
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, &event));
    ASSERT_NE(event, nullptr);
    EXPECT_SUCCESS(event_destroy(ctx, event));
  }
}

TEST_F(DriverPluginContextTest, EventSynchronize) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t ctx : contexts) {
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, &event));
    EXPECT_SUCCESS(event_synchronize(ctx, event));
    EXPECT_SUCCESS(event_destroy(ctx, event));
  }
}

TEST_F(DriverPluginContextTest, EventIsReady) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;
  for (M_driver_context_t ctx : contexts) {
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, &event));
    bool isReady = false;
    EXPECT_SUCCESS(is_event_ready(ctx, event, &isReady));
    EXPECT_TRUE(isReady);
    EXPECT_SUCCESS(event_destroy(ctx, event));
  }
}
