//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Fixture.h"
#include <cstring>

#include "llvm/ADT/ScopeExit.h"

TEST_F(DriverPluginTest, StandardStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_SUCCESS, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);

  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_INCOMPATIBLE_VERSION,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Incompatible version");

  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNKNOWN_ERROR,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_GT(strlen(errorMessage), 0);
}

TEST_F(DriverPluginTest, BadStatusCode) {
  char errorMessage[1024];
  M_driver_status_t kPositiveStatusCode = 10; // Arbitrarily chosen
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, kPositiveStatusCode, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, -M_DRIVER_NUM_STANDARD_STATUS_CODES,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver, M_DRIVER_STATUS_CUSTOM_CODES_BASE + 1,
                                 errorMessage, sizeof(errorMessage)));
  // Conservative number that we expect no plugin to reach or exceed
  int64_t constexpr assumedNumCustomStatusCodes = 50000;
  ASSERT_EQ(M_DRIVER_STATUS_ARGUMENT_OUT_OF_RANGE,
            this->status_message(driver,
                                 M_DRIVER_STATUS_CUSTOM_CODES_BASE -
                                     assumedNumCustomStatusCodes,
                                 errorMessage, sizeof(errorMessage)));
}

TEST_F(DriverPluginTest, DriverPropertyVersion) {
  // Good version
  M_driver_version version = {
      .major = 0xFFFFFFFFu, .minor = 0xFFFFFFFFu, .patch = 0xFFFFFFFFu};
  ASSERT_EQ(
      M_DRIVER_STATUS_SUCCESS,
      this->property(driver, M_DRIVER_PROPERTY_VERSION, (void *)&version));
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_MAJOR, version.major);
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_MINOR, version.minor);
  EXPECT_EQ(M_DRIVER_INTERFACE_VERSION_PATCH, version.patch);
  std::cerr << "Version: " << version.major << "." << version.minor << "."
            << version.patch << "\n";

  // Bad version
  version.major = 0xFFFFFFFFu;
  version.minor = 0xFFFFFFFFu;
  version.patch = 0xFFFFFFFFu;
  M_driver_driver_t noDriver = nullptr;
  EXPECT_EQ(M_DRIVER_STATUS_INCOMPATIBLE_VERSION,
            this->create(&version, &noDriver));
  EXPECT_EQ(noDriver, nullptr);
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
    M_driver_memory_t memory;
    ASSERT_SUCCESS(memory_alloc_sync(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_sync(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocPinned) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 128, &memory));
    ASSERT_SUCCESS(memory_free_pinned(context, memory));
  }
}

TEST_F(DriverPluginContextTest, MemoryAllocReturnsNonNull) {
  for (M_driver_context_t context : contexts) {
    M_driver_memory_t memory = nullptr;
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
    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_pinned(context, 0, &memory));
    EXPECT_EQ(memory, nullptr);

    memory = nullptr;
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
              M_DRIVER_STATUS_BAD_MEMORY);
    EXPECT_EQ(memory_alloc_pinned(context, 128, nullptr),
              M_DRIVER_STATUS_BAD_MEMORY);
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
  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));
    ASSERT_NE(nullptr, queue);
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, QueueCopySync) {
  constexpr size_t size = 128;
  std::array<uint8_t, size> src;
  for (size_t i = 0; i < size; i++)
    src[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    ASSERT_NE(memory, nullptr);

    M_driver_memory_view toView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, src.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    EXPECT_EQ(src, dst);

    ASSERT_SUCCESS(memory_free_sync(context, memory));
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, QueueAsyncAllocSync) {
  if (api == "HIP" || api == "Metal")
    GTEST_SKIP() << "Unimplemented for " << api;

  constexpr size_t size = 128;
  std::array<uint8_t, size> src;
  for (size_t i = 0; i < size; i++)
    src[i] = static_cast<uint8_t>(i);

  for (M_driver_context_t context : contexts) {
    M_driver_queue_t queue = nullptr;
    ASSERT_SUCCESS(queue_create(context, &queue));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_async(queue, size, &memory));
    ASSERT_NE(memory, nullptr);
    ASSERT_SUCCESS(queue_synchronize(queue));

    M_driver_memory_view toView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, src.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    EXPECT_EQ(src, dst);

    ASSERT_SUCCESS(memory_free_async(queue, memory));
    ASSERT_SUCCESS(queue_synchronize(queue));
    ASSERT_SUCCESS(queue_destroy(context, queue));
  }
}

TEST_F(DriverPluginContextTest, EventCreateDestroy) {
  for (M_driver_context_t ctx : contexts) {
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, M_DRIVER_EVENT_FLAG_NONE, &event));
    ASSERT_NE(event, nullptr);
    EXPECT_SUCCESS(event_destroy(ctx, event));
  }
}

TEST_F(DriverPluginContextTest, EventRecordWait) {
  constexpr size_t numElems = 256;
  constexpr size_t byteSize = numElems * sizeof(unsigned int);
  std::array<unsigned int, numElems> src;
  for (unsigned int i = 0; i < numElems; i++)
    src[i] = i;

  for (M_driver_context_t ctx : contexts) {
    M_driver_queue_t queueA = nullptr;
    M_driver_queue_t queueB = nullptr;
    ASSERT_SUCCESS(queue_create(ctx, &queueA));
    ASSERT_SUCCESS(queue_create(ctx, &queueB));

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(ctx, byteSize, &memory));

    M_driver_memory_view toView{memory, /*offset=*/0, byteSize};
    ASSERT_SUCCESS(queue_copy_to_device(queueA, &toView, src.data()));
    M_driver_event_t event = nullptr;
    ASSERT_SUCCESS(event_create(ctx, M_DRIVER_EVENT_FLAG_CPU_VISIBLE, &event));
    ASSERT_SUCCESS(queue_record_event(queueA, event));

    ASSERT_SUCCESS(queue_wait_for_events(queueB, &event, 1));
    std::array<unsigned int, numElems> dst;
    dst.fill(0);
    M_driver_memory_view fromView{memory, /*offset=*/0, byteSize};
    ASSERT_SUCCESS(queue_copy_from_device(queueB, dst.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queueB));

    EXPECT_EQ(src, dst);

    bool isReady = false;
    EXPECT_SUCCESS(is_event_ready(ctx, event, &isReady));
    EXPECT_TRUE(isReady);

    EXPECT_SUCCESS(event_destroy(ctx, event));
    EXPECT_SUCCESS(memory_free_sync(ctx, memory));
    EXPECT_SUCCESS(queue_destroy(ctx, queueB));
    EXPECT_SUCCESS(queue_destroy(ctx, queueA));
  }
}

TEST_F(DriverPluginQueueTest, CopyToDeviceThenFromDevice) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < content.size(); i++)
    content[i] = i;

  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    M_driver_memory_view dstView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, /*dst=*/&dstView,
                                        /*src=*/content.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view srcView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, /*dst=*/result.data(),
                                          /*src=*/&srcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    ASSERT_EQ(content, result);
  }
}

TEST_F(DriverPluginQueueTest, SetMemory) {
  constexpr size_t size = 128;
  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t memory = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &memory));
    auto defer = llvm::scope_exit([&]() { memory_free_sync(context, memory); });

    M_driver_memory_view setView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_set_memory(queue, /*dst=*/&setView, uint8_t(42),
                                    sizeof(uint8_t)));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    M_driver_memory_view srcView{memory, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, /*dst=*/result.data(),
                                          /*src=*/&srcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> expected;
    expected.fill(42);

    ASSERT_EQ(expected, result);
  }
}

TEST_F(DriverPluginQueueTest, CopyIntraDevice) {
  constexpr size_t size = 256;
  std::array<uint8_t, size> content;
  for (size_t i = 0; i < size; i++)
    content[i] = static_cast<uint8_t>(i);

  for (size_t i = 0; i < contexts.size(); i++) {
    M_driver_context_t context = contexts[i];
    M_driver_queue_t queue = queues[i];

    M_driver_memory_t srcMem = nullptr;
    M_driver_memory_t dstMem = nullptr;
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &srcMem));
    ASSERT_SUCCESS(memory_alloc_sync(context, size, &dstMem));
    auto defer = llvm::scope_exit([&]() {
      memory_free_sync(context, dstMem);
      memory_free_sync(context, srcMem);
    });

    // Sync between each pair so the test is correct regardless of whether
    // the plugin's raw queue is a stream.
    M_driver_memory_view toView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_to_device(queue, &toView, content.data()));
    ASSERT_SUCCESS(queue_synchronize(queue));

    M_driver_memory_view intraDstView{dstMem, /*offset=*/0, size};
    M_driver_memory_view intraSrcView{srcMem, /*offset=*/0, size};
    ASSERT_SUCCESS(
        queue_copy_intra_device(queue, &intraDstView, &intraSrcView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    std::array<uint8_t, size> result;
    result.fill(0);
    M_driver_memory_view fromView{dstMem, /*offset=*/0, size};
    ASSERT_SUCCESS(queue_copy_from_device(queue, result.data(), &fromView));
    ASSERT_SUCCESS(queue_synchronize(queue));

    ASSERT_EQ(content, result);
  }
}
