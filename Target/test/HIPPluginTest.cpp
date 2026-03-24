//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../lib/HIP/Error.h"
#include "Fixture.h"

// Extend test-cases per-plugin beyond the common ones in PluginTest.cpp

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_UNIMPLEMENTED, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(M_DRIVER_STATUS_BAD_HANDLE, errorMessage,
                                 sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Bad handle");
}

TEST_F(DriverPluginContextTest, EventCreateDestroyMultiple) {
  for (M_driver_context_t ctx : contexts) {
    constexpr int kNumEvents = 8;
    M_driver_event_t events[kNumEvents];
    for (int i = 0; i < kNumEvents; i++) {
      events[i] = nullptr;
      ASSERT_SUCCESS(event_create(ctx, &events[i]));
      ASSERT_NE(events[i], nullptr);
    }
    for (int i = 0; i < kNumEvents; i++) {
      EXPECT_SUCCESS(event_destroy(ctx, events[i]));
    }
  }
}

TEST_F(DriverPluginContextTest, EventNullContext) {
  M_driver_event_t event = nullptr;
  ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE, event_create(nullptr, &event));

  ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE, event_destroy(nullptr, event));

  ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE, event_synchronize(nullptr, event));

  bool isReady = false;
  ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE,
            is_event_ready(nullptr, event, &isReady));
}

TEST_F(DriverPluginContextTest, EventDestroyNull) {
  for (M_driver_context_t ctx : contexts) {
    ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE, event_destroy(ctx, nullptr));
  }
}

TEST_F(DriverPluginContextTest, EventSynchronizeNull) {
  for (M_driver_context_t ctx : contexts) {
    ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE, event_synchronize(ctx, nullptr));
  }
}

TEST_F(DriverPluginContextTest, EventIsReadyNull) {
  for (M_driver_context_t ctx : contexts) {
    bool isReady = false;
    ASSERT_EQ(M_DRIVER_STATUS_BAD_HANDLE,
              is_event_ready(ctx, nullptr, &isReady));
  }
}
