//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace M::Driver;
using M::ErrorOr;

namespace {

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
	add.f32 	%f3, %f2, %f1;
	st.f32 	[%rd1], %f3;
$L__BB0_2:
	ret;
}
)PTX";

class GraphCaptureTest : public ::testing::Test {};

TEST_F(GraphCaptureTest, BeginAndEndCapture) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);
}

TEST_F(GraphCaptureTest, DoubleBeginCaptureErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  EXPECT_ERR(stream->beginCapture());

  // Clean up
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
}

TEST_F(GraphCaptureTest, EndCaptureWithoutBeginErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_ERR(stream->endCapture());
}

TEST_F(GraphCaptureTest, CaptureAndReplayEmpty) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());

  // Replay the captured graph
  EXPECT_NO_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, ReplayMultipleTimes) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());

  // Replay multiple times
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, CPUDeviceReturnsNotSupported) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  EXPECT_ERR(stream->beginCapture());
  EXPECT_ERR(stream->endCapture());
}

TEST_F(GraphCaptureTest, ReplayOnCPUStreamErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, cudaCtx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  const auto &cudaDevice = CUDADevice::cast(cudaCtx->device());
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());

  DeviceGraphRef graph =
      CUDADeviceGraphRef::create(cudaDevice, /*exec=*/nullptr,
                                 DeviceStreamRef::copy(cpuStream.getPointer()));
  EXPECT_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, LaunchTraceCapturesKernelLaunches) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  DeviceFunction::SignatureBuilder sig = DeviceFunction::SignatureBuilder()
                                             .addBuffer()  // out
                                             .addBuffer()  // in0
                                             .addBuffer()  // in1
                                             .addUInt64(); // size

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                ctx->loadFunction("vec_add_module", "vec_add", sig, vecAddPtx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {&outPtr, &in0Ptr, &in1Ptr, (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  stream->beginLaunchTrace();
  EXPECT_NO_ERR(stream->enqueueFunctionExecDirect(
      *deviceFunc,
      /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
      /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
      /*sharedMemBytes=*/0,
      /*attrs=*/nullptr, /*numAttrs=*/0, args));
  EXPECT_NO_ERR(stream->synchronize());
  std::vector<uint64_t> baseline = stream->takeLaunchTrace();
  EXPECT_FALSE(baseline.empty());

  stream->beginLaunchTrace();
  EXPECT_NO_ERR(stream->enqueueFunctionExecDirect(
      *deviceFunc,
      /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
      /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
      /*sharedMemBytes=*/0,
      /*attrs=*/nullptr, /*numAttrs=*/0, args));
  EXPECT_NO_ERR(stream->synchronize());
  std::vector<uint64_t> repeat = stream->takeLaunchTrace();
  EXPECT_EQ(baseline, repeat);

  stream->beginLaunchTrace();
  EXPECT_NO_ERR(stream->enqueueFunctionExecDirect(
      *deviceFunc,
      /*gridX=*/gridDim + 1, /*gridY=*/1, /*gridZ=*/1,
      /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
      /*sharedMemBytes=*/0,
      /*attrs=*/nullptr, /*numAttrs=*/0, args));
  EXPECT_NO_ERR(stream->synchronize());
  std::vector<uint64_t> changed = stream->takeLaunchTrace();
  EXPECT_NE(baseline, changed);
}

// Test: Capture using context->stream() (default stream pattern used by
// device graph capture in the engine API).
TEST_F(GraphCaptureTest, CaptureOnDefaultStream) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));

  // Get the default stream.
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);

  // Replay on the same stream
  EXPECT_NO_ERR(graph->replay());
}

} // namespace
