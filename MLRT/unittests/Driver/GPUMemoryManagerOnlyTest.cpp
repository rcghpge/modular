//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <memory>

using namespace M::Driver;
using M::ErrorOr;

namespace {

class GPUMemoryManagerOnlyTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError()) {
      GTEST_SKIP() << "GPU not available, skipping tests";
    }
#if MODULAR_MACOS
    GTEST_SKIP() << "MemoryManager not used on macOS, skipping tests";
#endif
  }

  ErrorOr<DeviceRef> createTestDevice() {
    return Device::create(Device::gpuAPI);
  }

  ErrorOr<DeviceContextRef> createTestDeviceContext() {
    return DeviceContext::create(Device::gpuAPI);
  }
};

TEST_F(GPUMemoryManagerOnlyTest, RejectsAllocationLargerThanMaxCacheSize) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, createTestDeviceContext());
  MemoryManagerFreeList &freeList = ctx->freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  constexpr size_t kMaxCacheSize = 16 * 1024 * 1024;

  EXPECT_ERR_MSG(freeList.allocate(kMaxCacheSize + 1024, mode),
                 "Memory manager cannot satisfy allocation");
}

TEST_F(GPUMemoryManagerOnlyTest, RejectsWhenGpuMemoryManagerIsDisabled) {
  ASSIGN_NO_ERR(DeviceRef, device, createTestDevice());
  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());

  std::unique_ptr<MemoryManager> cache(
      MemoryManager::onDevice(*device, memInfo, device->allocationAlignment(),
                              /*memoryLimit=*/0));

  EXPECT_ERR_MSG(cache->freeList()->allocate(64, Device::AllocMode::SYNC),
                 "memory manager is disabled");
}

} // namespace
