//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/CUDADeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

using namespace M::Driver;
using M::ErrorOr;

namespace {

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

// Test: Capture using context->stream() (default stream pattern used by
// ExecutionTrace)
TEST_F(GraphCaptureTest, CaptureOnDefaultStream) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /*id=*/0));

  // Get the default stream (this is what ExecutionTrace does)
  ASSIGN_NO_ERR(DeviceStreamRef, stream, ctx->stream());

  EXPECT_NO_ERR(stream->beginCapture());
  ASSIGN_NO_ERR(DeviceGraphRef, graph, stream->endCapture());
  EXPECT_TRUE(graph);

  // Replay on the same stream
  EXPECT_NO_ERR(graph->replay());
}

} // namespace
