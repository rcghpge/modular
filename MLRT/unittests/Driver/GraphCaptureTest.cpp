//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;
using M::success;

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

TEST_F(GraphCaptureTest, CaptureProducesGraphs) {
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
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, [&]() -> ErrorOrSuccess {
        return stream->enqueueFunctionExecDirect(
            *deviceFunc,
            /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
            /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
            /*sharedMemBytes=*/0,
            /*attrs=*/nullptr, /*numAttrs=*/0, args);
      });

  // Successfully ran the capture process.
  EXPECT_FALSE(captured.isError());

  // One graph was produced.
  EXPECT_EQ(captured->numGraphs(), 1);

  // The graph can be replayed.
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
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

// Test: debugVerifyReplay succeeds when execution matches captured graph.
TEST_F(GraphCaptureTest, DebugVerifyReplaySuccess) {
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
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  auto execute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Capture the execution
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, execute);
  EXPECT_FALSE(captured.isError());

  // Verify that replaying the same execution matches the captured graph
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute));
}

// Test: debugVerifyReplay detects when execution differs from captured graph.
TEST_F(GraphCaptureTest, DebugVerifyReplayDetectsMismatch) {
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
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture with one kernel launch
  auto captureExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with TWO kernel launches (should mismatch)
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    auto err1 = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
    if (err1.isError())
      return err1;

    // Second launch (creates a mismatch)
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the mismatch
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay detects different grid dimensions.
TEST_F(GraphCaptureTest, DebugVerifyReplayDetectsDifferentGridDim) {
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
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture with one grid dimension
  auto captureExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with different grid dimension
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim * 2, /*gridY=*/1, /*gridZ=*/1, // Different grid!
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the mismatch in launch parameters
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay handles execution failure gracefully.
TEST_F(GraphCaptureTest, DebugVerifyReplayWithExecutionFailure) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Capture an empty graph
  auto captureExecute = [&]() -> ErrorOrSuccess { return success(); };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify with an execution that fails
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    return M::Error("Simulated execution failure");
  };

  // Should propagate the execution error
  EXPECT_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: debugVerifyReplay handles empty graph (no operations).
TEST_F(GraphCaptureTest, DebugVerifyReplayEmptyGraph) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Capture and verify empty execution
  auto emptyExecute = [&]() -> ErrorOrSuccess { return success(); };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, emptyExecute);
  EXPECT_FALSE(captured.isError());

  // Verify empty execution should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(emptyExecute));
}

// Test: debugVerifyReplay handles different execution order (simulates AsyncRT
// non-determinism). Verifies that traces are checked as permutations, not exact
// order matches.
TEST_F(GraphCaptureTest, DebugVerifyReplayHandlesReorderedExecution) {
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
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut1,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut2,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *out1Ptr = bufOut1->ptr();
  void *out2Ptr = bufOut2->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  // Capture: Execute kernel A with gridDim, then kernel B with gridDim/2
  // Use different grid dimensions so kernels have different hashes
  auto captureExecute = [&]() -> ErrorOrSuccess {
    // Kernel A: bufOut1 = bufIn0 + bufIn1 (full grid)
    void *argsA[] = {(void *)&out1Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    auto errA = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsA);
    if (errA.isError())
      return errA;

    // Kernel B: bufOut2 = bufIn0 + bufIn1 (half grid)
    void *argsB[] = {(void *)&out2Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsB);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, captureExecute);
  EXPECT_FALSE(captured.isError());

  // Verify: Execute kernel B then kernel A (different order)
  // This simulates AsyncRT non-deterministic scheduling
  auto verifyExecute = [&]() -> ErrorOrSuccess {
    // Kernel B first (half grid)
    void *argsB[] = {(void *)&out2Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    auto errB = stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim / 2, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsB);
    if (errB.isError())
      return errB;

    // Kernel A second (full grid)
    void *argsA[] = {(void *)&out1Ptr, (void *)&in0Ptr, (void *)&in1Ptr,
                     (void *)&length};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, argsA);
  };

  // Should succeed since traces are permutations of each other (same
  // operations, different order)
  EXPECT_NO_ERR(captured->debugVerifyReplay(verifyExecute));
}

// Test: Graph capture and replay works correctly with sync mode enabled.
TEST_F(GraphCaptureTest, CaptureWithSyncMode) {
  // Set MODULAR_DEVICE_CONTEXT_SYNC_MODE=true before creating context
  ScopedEnvVar env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "true");

  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));

  // Verify that sync mode is enabled
  EXPECT_TRUE(ctx->syncMode());

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
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, [&]() -> ErrorOrSuccess {
        return stream->enqueueFunctionExecDirect(
            *deviceFunc,
            /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
            /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
            /*sharedMemBytes=*/0,
            /*attrs=*/nullptr, /*numAttrs=*/0, args);
      });

  // Successfully ran the capture process with sync mode enabled.
  EXPECT_FALSE(captured.isError());

  // One graph was produced.
  EXPECT_EQ(captured->numGraphs(), 1);

  // The graph can be replayed multiple times with sync mode enabled.
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
  EXPECT_NO_ERR(captured->replay());
}

} // namespace
