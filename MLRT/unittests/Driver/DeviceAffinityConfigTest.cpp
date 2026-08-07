//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Kept in its own binary: runtime.device_task_cpu_ids is read once and cached
// for the process, so the env var must be set before any other test can
// trigger that read. The BUILD target supplies it.
//
// The configured list is "3,9999" against a pool of 8 workers: device 0 names
// a valid worker, device 1 names one past the end of the pool.

#include "MLRT/Driver/DeviceContext/DeviceAffinity.h"

#include "gtest/gtest.h"

using namespace M::MLRT;

namespace {

constexpr size_t kNumWorkers = 8;

// A configured ID that fits the pool is used as-is, ahead of the later tiers.
TEST(DeviceAffinityConfig, UsesConfiguredWorkerIdInRange) {
  EXPECT_EQ(taskIdForDevice(0, kNumWorkers), 3);
}

// A worker ID is an index into the worker array, so a configured ID past the
// end of the pool is an ill-formed mapping. Report it rather than silently
// placing the device somewhere else.
TEST(DeviceAffinityConfigDeathTest, RejectsConfiguredWorkerIdPastPool) {
  EXPECT_DEATH(taskIdForDevice(1, kNumWorkers), "is not a valid worker ID");
}

} // namespace
