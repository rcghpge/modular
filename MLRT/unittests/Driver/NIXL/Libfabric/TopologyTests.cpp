//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_common.h"
#include "MLRT/Driver/NIXL/utils/libfabric/libfabric_topology.h"
#include "gtest/gtest.h"

using namespace M::MLRT;

TEST(LibfabricTopologyTests, DiscoversProviderAndDevices) {
  if (LibfabricUtils::getAvailableEfaDevices().empty()) {
    GTEST_SKIP()
        << "libfabric topology discovery requires available EFA devices";
  }

  auto topology = nixlLibfabricTopology::create();
  ASSERT_FALSE(topology.isError()) << topology.getError();

  EXPECT_TRUE(topology.get()->isTopologyDiscovered());
  EXPECT_FALSE(topology.get()->getAllEfaDevices().empty());
}
