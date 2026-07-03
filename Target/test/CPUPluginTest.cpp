//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "../lib/CPU/Common.h"
#include "Fixture.h"
#include "HAL/M_driver_api.h"

// CPU-plugin-specific cases beyond the shared ones in PluginTest.cpp.

TEST_F(DriverPluginTest, CustomStatusMessage) {
  char errorMessage[1024];
  ASSERT_EQ(M_DRIVER_STATUS_SUCCESS,
            this->status_message(driver, M_DRIVER_STATUS_UNIMPLEMENTED,
                                 errorMessage, sizeof(errorMessage)));
  ASSERT_STREQ(errorMessage, "Unimplemented");
}

TEST_F(DriverPluginTest, ApiIsCpu) { ASSERT_EQ(api, "cpu"); }

TEST_F(DriverPluginQueueTest, QueueIsStream) {
  for (M_driver_queue_t queue : queues) {
    bool isStream = false;
    ASSERT_SUCCESS(queue_is_stream(queue, &isStream));
    EXPECT_TRUE(isStream);
  }
}
