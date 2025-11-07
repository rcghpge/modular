//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/lib/DeviceContext/Virtual/VirtualDeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <chrono>

namespace M::AsyncRT {

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

  DeviceRef realDevice_;
};

TEST_F(VirtualDeviceContextTest, DeviceCreation) {
  // Test creating virtual devices with different IDs
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", realDevice_.getPointer());
  VirtualDeviceRef device1 = VirtualDeviceRef::create(
      1, "VirtualGPU-1", realDevice_.getPointer(), 8LL * 1024 * 1024 * 1024);

  EXPECT_EQ(device0->id(), 0);
  EXPECT_EQ(device1->id(), 1);
  EXPECT_EQ(device0->name(), "VirtualGPU-0");
  EXPECT_EQ(device1->name(), "VirtualGPU-1");
  // Virtual device should use the same API as the real device
  EXPECT_EQ(device0->api(), realDevice_->api());
  EXPECT_EQ(device1->api(), realDevice_->api());
}

TEST_F(VirtualDeviceContextTest, DeviceCompatibility) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual devices should always be compatible
  EXPECT_NO_ERR(device->isCompatible());

  // Check API version
  ASSIGN_NO_ERR(int, apiVersion, device->getApiVersion());
  EXPECT_GE(apiVersion, 0);

  // Check compute capability
  ASSIGN_NO_ERR(int, computeCap, device->computeCapability());
  EXPECT_GE(computeCap, 0);
}

TEST_F(VirtualDeviceContextTest, ApiVersionMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device API version should match the real device's API version
  ASSIGN_NO_ERR(int, virtualApiVersion, virtualDevice->getApiVersion());
  ASSIGN_NO_ERR(int, realApiVersion, realDevice_->getApiVersion());

  EXPECT_EQ(virtualApiVersion, realApiVersion);
}

TEST_F(VirtualDeviceContextTest, DriverVersionMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device driver version should match the real device's driver version
  ErrorOr<DriverVersion &> virtualDriverVersionErr =
      virtualDevice->getDriverVersion();
  ErrorOr<DriverVersion &> realDriverVersionErr =
      realDevice_->getDriverVersion();

  EXPECT_FALSE(virtualDriverVersionErr.isError());
  EXPECT_FALSE(realDriverVersionErr.isError());

  // Both should return the same reference
  EXPECT_EQ(&virtualDriverVersionErr.get(), &realDriverVersionErr.get());
}

TEST_F(VirtualDeviceContextTest, IsCompatibleMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device compatibility should match the real device's compatibility
  ErrorOrSuccess virtualCompatible = virtualDevice->isCompatible();
  ErrorOrSuccess realCompatible = realDevice_->isCompatible();

  EXPECT_EQ(virtualCompatible.isError(), realCompatible.isError());
}

TEST_F(VirtualDeviceContextTest, ComputeCapabilityMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device compute capability should match the real device's compute
  // capability
  ASSIGN_NO_ERR(int, virtualComputeCap, virtualDevice->computeCapability());
  ASSIGN_NO_ERR(int, realComputeCap, realDevice_->computeCapability());

  EXPECT_EQ(virtualComputeCap, realComputeCap);
}

TEST_F(VirtualDeviceContextTest, CanAccessMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device canAccess should match the real device's canAccess
  ASSIGN_NO_ERR(bool, virtualCanAccess, virtualDevice->canAccess(*realDevice_));
  ASSIGN_NO_ERR(bool, realCanAccess, realDevice_->canAccess(*realDevice_));

  EXPECT_EQ(virtualCanAccess, realCanAccess);
}

TEST_F(VirtualDeviceContextTest, EnablePeerAccessMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device enablePeerAccess should match the real device's
  // enablePeerAccess
  ErrorOrSuccess virtualEnablePeer =
      virtualDevice->enablePeerAccess(*realDevice_);
  ErrorOrSuccess realEnablePeer = realDevice_->enablePeerAccess(*realDevice_);

  EXPECT_EQ(virtualEnablePeer.isError(), realEnablePeer.isError());
}

TEST_F(VirtualDeviceContextTest, GetAttributeMatchesRealDevice) {
  VirtualDeviceRef virtualDevice =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device getAttribute should match the real device's getAttribute
  ASSIGN_NO_ERR(int, virtualAttr,
                virtualDevice->getAttribute(Device::Attribute::WARP_SIZE));
  ASSIGN_NO_ERR(int, realAttr,
                realDevice_->getAttribute(Device::Attribute::WARP_SIZE));

  EXPECT_EQ(virtualAttr, realAttr);
}

TEST_F(VirtualDeviceContextTest, PeerAccess) {
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", realDevice_.getPointer());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", realDevice_.getPointer());

  // Virtual devices delegate peer access queries to the real device
  // Since both virtual devices share the same real CPU device, check
  // what the real device reports about accessing itself
  ASSIGN_NO_ERR(bool, realCanAccess, realDevice_->canAccess(*realDevice_));
  ASSIGN_NO_ERR(bool, virtualCanAccess, device0->canAccess(*device1));

  // Virtual devices should report the same as the real device
  EXPECT_EQ(virtualCanAccess, realCanAccess);
}

TEST_F(VirtualDeviceContextTest, MemoryInfo) {
  VirtualDeviceRef device = VirtualDeviceRef::create(
      0, "VirtualGPU", realDevice_.getPointer(), 16LL * 1024 * 1024 * 1024);

  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());
  EXPECT_GT(memInfo.total, 0);
  EXPECT_EQ(memInfo.free, memInfo.total);
  EXPECT_EQ(memInfo.total, 16LL * 1024 * 1024 * 1024);
}

TEST_F(VirtualDeviceContextTest, MemoryInfoDefaultLimit) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  // Virtual device without explicit memory limit should query the real device
  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());

  // Should return positive values from the real device
  EXPECT_GT(memInfo.total, 0);
  EXPECT_GT(memInfo.free, 0);
  // For CPU devices, total should be much larger than the old 8GB default
  EXPECT_GT(memInfo.total, 8LL * 1024 * 1024 * 1024);
}

TEST_F(VirtualDeviceContextTest, ContextCreation) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());
  EXPECT_NE(ctx.getPointer(), nullptr);
  EXPECT_EQ(&ctx->device(), device.getPointer());
}

TEST_F(VirtualDeviceContextTest, Synchronization) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Synchronization should be a no-op and succeed
  EXPECT_NO_ERR(ctx->synchronize());
}

TEST_F(VirtualDeviceContextTest, EventCreation) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Test enqueue event
  ASSIGN_NO_ERR(DeviceEventRef, event1, ctx->enqueueEvent());
  EXPECT_NE(event1.getPointer(), nullptr);

  // Test event create with flags
  ASSIGN_NO_ERR(DeviceEventRef, event2, ctx->eventCreate(0));
  EXPECT_NE(event2.getPointer(), nullptr);
}

TEST_F(VirtualDeviceContextTest, EventOperations) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  ASSIGN_NO_ERR(DeviceEventRef, event, ctx->enqueueEvent());

  // Event should always be ready
  ASSIGN_NO_ERR(bool, isReady, event->isReady());
  EXPECT_TRUE(isReady);

  // Event synchronization should be a no-op
  EXPECT_NO_ERR(event->synchronize());
}

TEST_F(VirtualDeviceContextTest, ContextEventWait) {
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", realDevice_.getPointer());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx0, device0->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, device1->createContext());

  ASSIGN_NO_ERR(DeviceEventRef, event, ctx0->enqueueEvent());

  // Context should be able to wait for events (no-op)
  EXPECT_NO_ERR(ctx1->enqueueWaitFor(*event));
  EXPECT_NO_ERR(ctx1->enqueueWaitFor(*ctx0));
}

TEST_F(VirtualDeviceContextTest, TimerOperations) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Start a timer
  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx->startTimer());
  EXPECT_NE(timer.getPointer(), nullptr);

  // Stop the timer (should return 0 duration)
  ASSIGN_NO_ERR(std::chrono::nanoseconds, duration, ctx->stopTimer(*timer));
  EXPECT_EQ(duration.count(), 0);
}

TEST_F(VirtualDeviceContextTest, PeerAccessContext) {
  VirtualDeviceRef device0 =
      VirtualDeviceRef::create(0, "VirtualGPU-0", realDevice_.getPointer());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", realDevice_.getPointer());
  ASSIGN_NO_ERR(DeviceContextRef, ctx0, device0->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, device1->createContext());

  // Enabling peer access should succeed (no-op)
  EXPECT_NO_ERR(ctx0->enablePeerAccess(*ctx1));
}

TEST_F(VirtualDeviceContextTest, CastOperations) {
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", realDevice_.getPointer());

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
      VirtualDeviceRef::create(0, "VirtualGPU-0", realDevice_.getPointer());
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", realDevice_.getPointer());
  VirtualDeviceRef device2 =
      VirtualDeviceRef::create(2, "VirtualGPU-2", realDevice_.getPointer());

  ASSIGN_NO_ERR(DeviceContextRef, ctx0, device0->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx1, device1->createContext());
  ASSIGN_NO_ERR(DeviceContextRef, ctx2, device2->createContext());

  // All should be usable
  EXPECT_NO_ERR(ctx0->synchronize());
  EXPECT_NO_ERR(ctx1->synchronize());
  EXPECT_NO_ERR(ctx2->synchronize());
}

} // namespace M::AsyncRT
