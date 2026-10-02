//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "stream/metadata_stream.h"
#include "gtest/gtest.h"

#include <cerrno>
#include <sys/socket.h>

// Regression test for macOS compatibility
TEST(MetadataStreamTest, ListenerSocketIsNonBlocking) {
  nixlMDStreamListener listener(19876);
  listener.setupListener();

  nixl::scopedFd csock = listener.acceptClient();

  EXPECT_FALSE(csock.valid());
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK)
      << "Expected EAGAIN or EWOULDBLOCK, got errno=" << errno;
}
