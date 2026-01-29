//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/Virtual/VirtualDeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDACompilationDevice.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <chrono>

namespace M::Driver {

class VirtualDeviceContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a real CPU device to back the virtual devices
    ErrorOr<DeviceRef> cpuDeviceErr = Device::create(Device::cpuAPI, 0);
    if (cpuDeviceErr.isError()) {
      GTEST_SKIP() << "Failed to create CPU device: "
                   << cpuDeviceErr.getError();
    }
    realDevice_ = cpuDeviceErr.takeValue();
  }

  // Helper to create a CUDA compilation device for testing
  std::unique_ptr<CUDACompilationDevice>
  createCompilationDevice(const char *arch = "sm_80") {
    auto result = CUDACompilationDevice::create(arch);
    if (result.isError()) {
      return nullptr;
    }
    return std::move(*result);
  }

  DeviceRef realDevice_;
};

TEST_F(VirtualDeviceContextTest, DeviceCreation) {
  // Test creating virtual devices with different IDs
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", createCompilationDevice());
  VirtualDeviceRef device1 = VirtualDeviceRef::create(
      1, "VirtualGPU-1", createCompilationDevice(), 8LL * 1024 * 1024 * 1024);

  EXPECT_EQ(device0->id(), 0);
  EXPECT_EQ(device1->id(), 1);
  EXPECT_EQ(device0->name(), "VirtualGPU-0");
  EXPECT_EQ(device1->name(), "VirtualGPU-1");
  // Virtual device should use CUDA API (from compilation device)
  EXPECT_EQ(device0->api(), Device::cudaAPI);
  EXPECT_EQ(device1->api(), Device::cudaAPI);
}

TEST_F(VirtualDeviceContextTest, DeviceCompatibility) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());

  // Virtual devices should always be compatible
  EXPECT_NO_ERR(device->isCompatible());

  // Check API version
  ASSIGN_NO_ERR(int, apiVersion, device->getApiVersion());
  EXPECT_GE(apiVersion, 0);
}

TEST_F(VirtualDeviceContextTest, PeerAccess) {
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", createCompilationDevice());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", createCompilationDevice());

  // Virtual devices should report false for peer access
  ASSIGN_NO_ERR(bool, virtualCanAccess, device0->canAccess(*device1));
  EXPECT_FALSE(virtualCanAccess);
}

TEST_F(VirtualDeviceContextTest, MemoryInfo) {
  VirtualDeviceRef device = VirtualDeviceRef::create(
      0, "VirtualGPU", createCompilationDevice(), 16LL * 1024 * 1024 * 1024);

  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());
  EXPECT_GT(memInfo.total, 0);
  EXPECT_EQ(memInfo.free, memInfo.total);
  EXPECT_EQ(memInfo.total, 16LL * 1024 * 1024 * 1024);
}

TEST_F(VirtualDeviceContextTest, MemoryInfoDefaultLimit) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());

  // Virtual device without explicit memory limit should return default (80GB)
  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());

  // Should return positive values
  EXPECT_GT(memInfo.total, 0);
  EXPECT_GT(memInfo.free, 0);
  // Default should be 192GB (typical for B200)
  EXPECT_EQ(memInfo.total, 192LL * 1024 * 1024 * 1024);
}

TEST_F(VirtualDeviceContextTest, ContextCreation) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());

  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  EXPECT_NE(ctx.getPointer(), nullptr);
  EXPECT_EQ(&ctx->device(), device.getPointer());
}

TEST_F(VirtualDeviceContextTest, Synchronization) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Synchronization should be a no-op and succeed
  EXPECT_NO_ERR(ctx->synchronize());
}

TEST_F(VirtualDeviceContextTest, PeerAccessContext) {
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", createCompilationDevice());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", createCompilationDevice());
  ASSIGN_NO_ERR(DeviceContextRef, ctx0, device0->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, device1->createContext());

  // Enabling peer access should succeed (no-op)
  EXPECT_NO_ERR(ctx0->enablePeerAccess(*ctx1));
}

TEST_F(VirtualDeviceContextTest, CastOperations) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());

  // Test Device cast
  const Device &deviceBase = *device;
  const VirtualDevice &deviceCast = VirtualDevice::cast(deviceBase);
  EXPECT_EQ(&deviceCast, device.getPointer());

  // Test DeviceContext cast
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  const DeviceContext &ctxBase = *ctx;
  const VirtualDeviceContext &ctxCast = VirtualDeviceContext::cast(ctxBase);
  EXPECT_EQ(&ctxCast, ctx.getPointer());

  // Test DeviceContext verified_cast
  const DeviceContext *ctxPtr = ctx.getPointer();
  const VirtualDeviceContext *ctxVerified =
      VirtualDeviceContext::verified_cast(ctxPtr);
  EXPECT_EQ(ctxVerified, ctx.getPointer());

  // Test verified_cast with nullptr
  const VirtualDeviceContext *nullCast =
      VirtualDeviceContext::verified_cast(nullptr);
  EXPECT_EQ(nullCast, nullptr);
}

TEST_F(VirtualDeviceContextTest, MultipleVirtualDevices) {
  // Test creating multiple virtual devices and contexts
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", createCompilationDevice());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", createCompilationDevice());
  VirtualDeviceRef device2 =
      VirtualDeviceRef::create(2, "VirtualGPU-2", createCompilationDevice());

  ASSIGN_NO_ERR(DeviceContextRef, ctx0, device0->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, device1->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx2, device2->createContext());

  // All should be usable
  EXPECT_NO_ERR(ctx0->synchronize());
  EXPECT_NO_ERR(ctx1->synchronize());
  EXPECT_NO_ERR(ctx2->synchronize());
}

} // namespace M::Driver
