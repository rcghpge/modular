//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/utils/stream/metadata_stream.h"
#include "gtest/gtest.h"

#include <cerrno>
#include <sys/socket.h>

using namespace M::MLRT;

// Regression test for macOS compatibility
TEST(MetadataStreamTest, ListenerSocketIsNonBlocking) {
  nixlMDStreamListener listener(19876);
  listener.setupListener();

  int csock = listener.acceptClient();

  EXPECT_EQ(csock, -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK)
      << "Expected EAGAIN or EWOULDBLOCK, got errno=" << errno;
}
