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
