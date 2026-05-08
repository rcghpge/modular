//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"
#include "gtest/gtest.h"

#include <cstdint>

using namespace M::MLRT;

namespace {

// addXferId returns true while there is space and false once the
// fixed-size xfer_ids array is full. Callers must consult this return
// value to drive their submission accounting; conflating it with the
// number of RDMAs they posted will leave the sender's completion
// tracker wedged on the unaccounted tail.
TEST(BinaryNotificationTests, AddXferIdReturnsFalseAtCapacity) {
  BinaryNotification notif;
  notif.clear();
  ASSERT_EQ(notif.xfer_id_count, 0u);

  for (uint32_t i = 0; i < NIXL_LIBFABRIC_MAX_XFER_IDS; ++i) {
    EXPECT_TRUE(notif.addXferId(i)) << "addXferId returned false at index " << i
                                    << " (capacity should be at least "
                                    << NIXL_LIBFABRIC_MAX_XFER_IDS << ")";
  }
  EXPECT_EQ(notif.xfer_id_count, NIXL_LIBFABRIC_MAX_XFER_IDS);

  // Past the cap: every further attempt should report failure and the
  // count must not advance. This is the contract the libfabric backend
  // relies on to detect overflow without scanning.
  for (uint32_t i = 0; i < 16; ++i) {
    EXPECT_FALSE(notif.addXferId(NIXL_LIBFABRIC_MAX_XFER_IDS + i));
    EXPECT_EQ(notif.xfer_id_count, NIXL_LIBFABRIC_MAX_XFER_IDS);
  }
}

// Stored xfer_ids round-trip through getXferIds() up to the capacity.
// Anything past the capacity is dropped; the backend reports the drop
// count separately via its submitted_count tracking.
TEST(BinaryNotificationTests, GetXferIdsReturnsExactlyAccepted) {
  BinaryNotification notif;
  notif.clear();

  // Mix accepted and dropped writes.
  const uint32_t accepted = NIXL_LIBFABRIC_MAX_XFER_IDS;
  const uint32_t attempted = accepted + 32;
  for (uint32_t i = 0; i < attempted; ++i) {
    notif.addXferId(i);
  }

  auto ids = notif.getXferIds();
  EXPECT_EQ(ids.size(), accepted);
  for (uint32_t i = 0; i < accepted; ++i) {
    EXPECT_TRUE(ids.count(i)) << "expected xfer_id " << i << " in set";
  }
  for (uint32_t i = accepted; i < attempted; ++i) {
    EXPECT_FALSE(ids.count(i))
        << "xfer_id " << i << " was dropped at the boundary "
        << "but reappeared in getXferIds()";
  }
}

// clear() rewinds the count so a notification can be reused without
// reallocating. Subsequent addXferId calls fill from index 0 again.
TEST(BinaryNotificationTests, ClearResetsCountAndAcceptsAgain) {
  BinaryNotification notif;
  notif.clear();
  for (uint32_t i = 0; i < 100; ++i) {
    ASSERT_TRUE(notif.addXferId(i));
  }
  EXPECT_EQ(notif.xfer_id_count, 100u);

  notif.clear();
  EXPECT_EQ(notif.xfer_id_count, 0u);

  for (uint32_t i = 0; i < NIXL_LIBFABRIC_MAX_XFER_IDS; ++i) {
    EXPECT_TRUE(notif.addXferId(i));
  }
  EXPECT_FALSE(notif.addXferId(NIXL_LIBFABRIC_MAX_XFER_IDS));
}

// The notification must fit in a single control-pool buffer so it can
// be sent via libfabric's tagged SEND/RECV path. The pool buffer size
// is sized off NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE; if the
// notification grows past that, the SEND will be rejected at runtime
// with a confusing error. Catch the mismatch at build/test time.
TEST(BinaryNotificationTests, FitsInControlBuffer) {
  EXPECT_LE(sizeof(BinaryNotification),
            static_cast<size_t>(NIXL_LIBFABRIC_SEND_RECV_BUFFER_SIZE));
}

} // namespace
