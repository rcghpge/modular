//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/lib/Driver/DeviceContext/Virtual/VirtualDeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDACompilationDevice.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <chrono>

namespace M::Driver {

// PTX for a simple vec_add kernel, used for compilation tests.
static const char *vecAddPtx = R"PTX(
// From max/kernels/test/gpu/vec_add.ptx

.version 6.3
.target sm_75
.address_size 64

	// .globl	vec_add

.visible .entry vec_add(
	.param .b64 vec_add_param_0,
	.param .b64 vec_add_param_1,
	.param .b64 vec_add_param_2,
	.param .b64 vec_add_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<28>;
	.reg .f32 	%f<4>;
	.reg .b64 	%rd<12>;

	mov.u32 	%r1, %tid.x;
	cvt.s64.s32 	%rd6, %r1;
	ld.param.u64 	%rd8, [vec_add_param_3];
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %ctaid.x;
	mul.wide.s32 	%rd9, %r2, %r3;
	add.s64 	%rd10, %rd9, %rd6;
	setp.ge.s64 	%p1, %rd10, %rd8;
	@%p1 bra 	$L__BB0_2;
	ld.param.u64 	%rd4, [vec_add_param_0];
	ld.param.u64 	%rd5, [vec_add_param_1];
	ld.param.u64 	%rd7, [vec_add_param_2];
	shl.b64 	%rd11, %rd10, 2;
	add.s64 	%rd1, %rd4, %rd11;
	add.s64 	%rd2, %rd7, %rd11;
	add.s64 	%rd3, %rd5, %rd11;
	ld.u8 	%r4, [%rd3];
	ld.u8 	%r5, [%rd3+1];
	shl.b32 	%r6, %r5, 8;
	or.b32  	%r7, %r6, %r4;
	ld.u8 	%r8, [%rd3+2];
	shl.b32 	%r9, %r8, 16;
	ld.u8 	%r10, [%rd3+3];
	shl.b32 	%r11, %r10, 24;
	or.b32  	%r12, %r11, %r9;
	or.b32  	%r13, %r12, %r7;
	mov.b32 	%f1, %r13;
	ld.u8 	%r14, [%rd2];
	ld.u8 	%r15, [%rd2+1];
	shl.b32 	%r16, %r15, 8;
	or.b32  	%r17, %r16, %r14;
	ld.u8 	%r18, [%rd2+2];
	shl.b32 	%r19, %r18, 16;
	ld.u8 	%r20, [%rd2+3];
	shl.b32 	%r21, %r20, 24;
	or.b32  	%r22, %r21, %r19;
	or.b32  	%r23, %r22, %r17;
	mov.b32 	%f2, %r23;
	add.rn.f32 	%f3, %f1, %f2;
	mov.b32 	%r24, %f3;
	st.u8 	[%rd1], %r24;
	shr.u32 	%r25, %r24, 16;
	st.u8 	[%rd1+2], %r25;
	shr.u32 	%r26, %r24, 24;
	st.u8 	[%rd1+3], %r26;
	shr.u32 	%r27, %r24, 8;
	st.u8 	[%rd1+1], %r27;
$L__BB0_2:
	ret;
}
)PTX";

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
  VirtualDeviceRef device1 =
      VirtualDeviceRef::create(1, "VirtualGPU-1", createCompilationDevice());

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
  VirtualDeviceRef device =
      VirtualDeviceRef::create(0, "VirtualGPU", createCompilationDevice());

  ASSIGN_NO_ERR(Device::MemInfo, memInfo, device->getMemoryInfo());
  // Virtual device returns simulated memory info (192GB, typical for B200)
  EXPECT_GT(memInfo.total, 0);
  EXPECT_EQ(memInfo.free, memInfo.total);
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

TEST_F(VirtualDeviceContextTest, PtxasFallbackBadPath) {
  ScopedEnvVar env("MODULAR_NVPTX_COMPILER_PATH", "/nonexistent/path/ptxas");
  auto device = createCompilationDevice("sm_80");
  ASSERT_NE(device, nullptr);
  EXPECT_ERR_MSG(device->compileFunction(vecAddPtx, "", 3), "does not exist");
}

TEST_F(VirtualDeviceContextTest, PtxasFallbackNotExecutable) {
  // Use /etc/passwd: a regular file that exists but is not executable.
  // /dev/null won't work because it's a device file, not a regular file.
  ScopedEnvVar env("MODULAR_NVPTX_COMPILER_PATH", "/etc/passwd");
  auto device = createCompilationDevice("sm_80");
  ASSERT_NE(device, nullptr);
  EXPECT_ERR_MSG(device->compileFunction(vecAddPtx, "", 3), "not executable");
}

TEST_F(VirtualDeviceContextTest, CompileFunctionDefaultPath) {
#if MODULAR_LINUX
  auto device = createCompilationDevice("sm_80");
  ASSERT_NE(device, nullptr);
  ASSIGN_NO_ERR(M::BufferRef, cubin, device->compileFunction(vecAddPtx, "", 3));
  EXPECT_GT(cubin->getBufferSize(), 0u);
#else
  GTEST_SKIP() << "NVPTX compiler not supported on this platform";
#endif
}

} // namespace M::Driver
