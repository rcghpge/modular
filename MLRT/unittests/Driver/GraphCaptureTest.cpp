//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"
#include "tools/cpp/runfiles/runfiles.h"

#include "gtest/gtest.h"

#include "llvm/Support/MemoryBuffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

using namespace M::Driver;
using bazel::tools::cpp::runfiles::Runfiles;
using M::ErrorOr;
using M::ErrorOrSuccess;
using M::success;

namespace {

static const char *smallScalarPtx = R"PTX(
// Kernel that takes sub-pointer-sized scalar parameters to test that
// parameters < 8 bytes ARE included in trace hash.

.version 6.3
.target sm_75
.address_size 64

.visible .entry small_scalar_kernel(
	.param .b64 ptr,           // pointer (8 bytes - excluded from hash)
	.param .u32 scalar32,      // 32-bit scalar (4 bytes - HASHED)
	.param .u32 grid_size      // 32-bit scalar (4 bytes - HASHED)
)
{
	ret;
}
)PTX";

static const char *mixedArgsPtx = R"PTX(
// Kernel that takes pointer-sized scalar arguments to test that
// parameters >= 8 bytes are conservatively excluded from hash.

.version 6.3
.target sm_75
.address_size 64

.visible .entry mixed_args(
	.param .b64 mixed_args_param_0,  // pointer (8 bytes)
	.param .b64 mixed_args_param_1,  // scalar (8 bytes - NOT hashed)
	.param .b64 mixed_args_param_2,  // pointer (8 bytes)
	.param .b64 mixed_args_param_3   // scalar (8 bytes - NOT hashed)
)
{
	ret;
}
)PTX";

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

static Runfiles *getRunfiles() {
  static std::unique_ptr<Runfiles> runfiles = [] {
    std::string error;
    auto result = std::unique_ptr<Runfiles>(Runfiles::CreateForTest(&error));
    if (!result)
      ADD_FAILURE() << "Failed to create runfiles: " << error;
    return result;
  }();
  return runfiles.get();
}

static ErrorOr<std::string> loadHIPCodeObject(std::string_view stem,
                                              const Device &device) {
  if (device.api() != Device::hipAPI)
    RETURN_ERROR("loadHIPCodeObject only supports HIP devices");

  DEFINE_OR_RETURN_ERROR(const char *, archName, device.getArchName());
  std::string_view arch(archName);

  if (arch.size() < 6 || arch.substr(0, 6) != "gfx950")
    RETURN_ERROR("Unsupported HIP arch for GraphCaptureTest: " +
                 std::string(arch));

  constexpr std::string_view archTag = "gfx950";

  Runfiles *runfiles = getRunfiles();
  if (!runfiles)
    RETURN_ERROR("Runfiles not available");

  // Dig around under the runfiles for the HSA code object (sorry).
  std::string runfilePath = "_main/MLRT/unittests/Driver/graph_capture_";
  runfilePath.append(stem.data(), stem.size());
  runfilePath.push_back('_');
  runfilePath.append(archTag.data(), archTag.size());
  runfilePath.append(".hsaco");
  std::string codeObjectPath = runfiles->Rlocation(runfilePath);
  if (codeObjectPath.empty())
    RETURN_ERROR("Failed to locate runfile: " + runfilePath);

  auto bufferOrErr = llvm::MemoryBuffer::getFile(codeObjectPath);
  if (!bufferOrErr)
    RETURN_ERROR("Failed to open code object: " + codeObjectPath);

  return (*bufferOrErr)->getBuffer().str();
}

static ErrorOr<DeviceFunctionRef>
loadKernelForActiveAPI(const DeviceContext &ctx, std::string_view moduleName,
                       std::string_view functionName, std::string_view cudaPtx,
                       std::string_view hipCodeStem) {
  std::string moduleNameStr(moduleName);
  std::string functionNameStr(functionName);

  if (ctx.device().api() == Device::cudaAPI)
    return ctx.loadFunction(moduleNameStr, functionNameStr,
                            std::string(cudaPtx));
  if (ctx.device().api() == Device::hipAPI) {
    DEFINE_OR_RETURN_ERROR(std::string, hsacoData,
                           loadHIPCodeObject(hipCodeStem, ctx.device()));
    return ctx.loadFunction(moduleNameStr, functionNameStr, hsacoData);
  }

  RETURN_ERROR("GraphCaptureTest requires CUDA or HIP backend");
}

static bool supportsLaunchTraceVerification(const DeviceContext &ctx) {
  return ctx.device().api() == Device::cudaAPI;
}

class GraphCaptureTest : public ::testing::Test {};

TEST_F(GraphCaptureTest, BeginAndEndCapture) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);
}

TEST_F(GraphCaptureTest, DoubleBeginCaptureErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  EXPECT_ERR(stream->beginCapture());

  // Clean up
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
}

TEST_F(GraphCaptureTest, EndCaptureWithoutBeginErrors) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_ERR(stream->endCapture());
}

TEST_F(GraphCaptureTest, CaptureAndReplayEmpty) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());

  // Replay the captured graph
  EXPECT_NO_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, ReplayMultipleTimes) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
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
  ASSIGN_NO_ERR(DeviceContextRef, gpuCtx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, gpuStream, gpuCtx->createStream());
  ASSIGN_NO_ERR(DeviceContextRef, cpuCtx,
                DeviceContext::create(Device::cpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, cpuStream, cpuCtx->stream());

  EXPECT_NO_ERR(gpuStream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, gpuStream->endCapture());
  graph->setStream(DeviceStreamRef::copy(cpuStream.getPointer()));
  EXPECT_ERR(graph->replay());
}

TEST_F(GraphCaptureTest, CaptureProducesGraphs) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  // Verify that sync mode is enabled
  EXPECT_TRUE(ctx->syncMode());

  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

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

// Test: Scalar parameter changes are detected in trace.
TEST_F(GraphCaptureTest, TraceDetectsScalarParameterChange) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  const uint32_t blockDim = 256;

  // Capture with length=1024
  uint64_t captureLength = 1024;
  auto captureExecute = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&captureLength};
    const uint32_t gridDim = (captureLength + blockDim - 1) / blockDim;
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

  // Replay with different length=2048 should detect mismatch
  uint64_t replayLength = 2048;
  auto replayExecute = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&replayLength};
    const uint32_t gridDim = (replayLength + blockDim - 1) / blockDim;
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the scalar parameter change
  EXPECT_ERR(captured->debugVerifyReplay(replayExecute));

  // Replay with same length=1024 should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(captureExecute));
}

// Test: Pointer reallocation doesn't affect trace (existing behavior).
TEST_F(GraphCaptureTest, TraceStableAcrossBufferReallocation) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Allocate first set of buffers
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut1,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn01,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn11,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr1 = bufOut1->ptr();
  void *in0Ptr1 = bufIn01->ptr();
  void *in1Ptr1 = bufIn11->ptr();
  void *args1[] = {(void *)&outPtr1, (void *)&in0Ptr1, (void *)&in1Ptr1,
                   (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  auto execute1 = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args1);
  };

  // Capture with first set of buffers
  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Free first set of buffers and allocate new ones (likely at different
  // addresses)
  bufOut1.reset();
  bufIn01.reset();
  bufIn11.reset();

  ASSIGN_NO_ERR(DeviceBufferRef, bufOut2,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn02,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn12,
                ctx->enqueueCreateBuffer(length * elemSize));

  void *outPtr2 = bufOut2->ptr();
  void *in0Ptr2 = bufIn02->ptr();
  void *in1Ptr2 = bufIn12->ptr();
  void *args2[] = {(void *)&outPtr2, (void *)&in0Ptr2, (void *)&in1Ptr2,
                   (void *)&length};

  auto execute2 = [&]() -> ErrorOrSuccess {
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args2);
  };

  // Execute with new buffers should match trace (pointer values excluded)
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute2));
}

// Test: Mixed pointer and scalar parameters - changing scalar is detected.
TEST_F(GraphCaptureTest, TraceMixedPointerAndScalarParams) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                loadKernelForActiveAPI(*ctx, "vec_add_module", "vec_add",
                                       vecAddPtx, "vec_add"));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();

  const uint32_t blockDim = 256;

  // Capture with length=512
  uint64_t length1 = 512;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&length1};
    const uint32_t gridDim = (length1 + blockDim - 1) / blockDim;
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Replay with different scalar (length=1024) should mismatch
  uint64_t length2 = 1024;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                    (void *)&length2};
    const uint32_t gridDim = (length2 + blockDim - 1) / blockDim;
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Changing the scalar parameter should trigger mismatch
  EXPECT_ERR(captured->debugVerifyReplay(execute2));
}

// Test: Sub-pointer-sized scalars ARE included in the trace hash.
// This verifies that the size-based detection correctly identifies and hashes
// parameters smaller than pointer size (e.g., uint32_t, float).
TEST_F(GraphCaptureTest, TraceDetectsSubPointerSizedScalarChanges) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Load kernel with mixed parameter sizes: (b64 pointer, u32, u32)
  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc,
                ctx->loadFunction("small_scalar_module", "small_scalar_kernel",
                                  smallScalarPtx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, buf,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *bufPtr = buf->ptr();

  // Keep grid dimensions CONSTANT - we're only testing parameter value changes
  const uint32_t gridDim = 1;
  const uint32_t blockDim = 256;

  // Test 1: Capture with initial scalar values
  uint32_t scalar1 = 12345;
  uint32_t scalar2 = 67890;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1, (void *)&scalar2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Test 2: Change the first u32 scalar - should be DETECTED
  // Grid dimensions remain the same, ONLY the parameter value changes
  uint32_t scalar1Changed = 99999;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1Changed, (void *)&scalar2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should detect the change in the u32 parameter (4 bytes < 8 bytes)
  EXPECT_ERR(captured->debugVerifyReplay(execute2));

  // Test 3: Change the second u32 scalar - should also be DETECTED
  uint32_t scalar2Changed = 11111;
  auto execute3 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&bufPtr, (void *)&scalar1, (void *)&scalar2Changed};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc, gridDim, /*gridY=*/1, /*gridZ=*/1, blockDim,
        /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0, /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  EXPECT_ERR(captured->debugVerifyReplay(execute3));

  // Test 4: Replay with original values should succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute1));
}

// Test: Size-based scalar detection conservatively excludes pointer-sized
// scalars to avoid false positives. Demonstrates that uint64_t changes are
// NOT detected (trade-off for reliability).
TEST_F(GraphCaptureTest, TraceDistinguishesScalarsFromPointerBitPatterns) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));
  if (!supportsLaunchTraceVerification(*ctx))
    GTEST_SKIP() << "Launch trace verification is currently CUDA-only";
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->createStream());

  // Signature: (ptr, uint64_t, ptr, uint64_t)
  // All scalar parameters are 8 bytes (pointer-sized), so they are NOT hashed
  ASSIGN_NO_ERR(
      DeviceFunctionRef, deviceFunc,
      ctx->loadFunction("mixed_args_module", "mixed_args", mixedArgsPtx));

  const size_t elemSize = sizeof(float);
  ASSIGN_NO_ERR(DeviceBufferRef, buf1,
                ctx->enqueueCreateBuffer(1024 * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, buf2,
                ctx->enqueueCreateBuffer(1024 * elemSize));

  void *ptr1 = buf1->ptr();
  void *ptr2 = buf2->ptr();

  const uint32_t blockDim = 256;
  const uint32_t gridDim = 1;

  // Capture with initial scalar values
  uint64_t scalar1Value1 = 0x7fff00001000ULL;
  uint64_t scalar2Value1 = 0x7fff00002000ULL;
  auto execute1 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&ptr1, (void *)&scalar1Value1, (void *)&ptr2,
                    (void *)&scalar2Value1};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  ErrorOr<DeviceGraphCollection> captured =
      DeviceGraph::capture({stream}, execute1);
  EXPECT_FALSE(captured.isError());

  // Verify that changing pointer-sized (8-byte) scalar parameters is NOT
  // detected. This demonstrates the conservative approach: we exclude
  // pointer-sized parameters to avoid false positives from uint64_t values
  // that might accidentally look like valid CUDA memory addresses.
  uint64_t scalar1Value2 = scalar1Value1 + 0x1000;
  uint64_t scalar2Value2 = scalar2Value1 + 0x2000;
  auto execute2 = [&]() -> ErrorOrSuccess {
    void *args[] = {(void *)&ptr1, (void *)&scalar1Value2, (void *)&ptr2,
                    (void *)&scalar2Value2};
    return stream->enqueueFunctionExecDirect(
        *deviceFunc,
        /*gridX=*/gridDim, /*gridY=*/1, /*gridZ=*/1,
        /*blockX=*/blockDim, /*blockY=*/1, /*blockZ=*/1,
        /*sharedMemBytes=*/0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args);
  };

  // Should NOT detect the change in pointer-sized scalars (conservative)
  // This is the expected trade-off: we miss uint64_t changes but avoid false
  // positives from pointer-like bit patterns.
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute2));

  // Replay with original scalar values should also succeed
  EXPECT_NO_ERR(captured->debugVerifyReplay(execute1));
}

} // namespace
