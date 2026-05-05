//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file tests the DeviceGraphBuilder interface for explicit device graph
// construction.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceGraphBuilder.h"
#include "DeviceGraphTestUtils.h"
#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

using namespace M::Driver;
using M::ErrorOr;

namespace {

TEST(DeviceGraphBuilderTest, VecAddKernelNode) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunc, loadVecAdd(*ctx));

  const uint64_t length = 1024;
  const size_t elemSize = sizeof(float);

  // Initialize host input buffers.
  llvm::SmallVector<float> hostIn0(length), hostIn1(length),
      hostOut(length, 0.0f);
  for (uint64_t i = 0; i < length; ++i) {
    hostIn0[i] = static_cast<float>(i);
    hostIn1[i] = static_cast<float>(length - i);
  }

  ASSIGN_NO_ERR(DeviceBufferRef, bufOut,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn0,
                ctx->enqueueCreateBuffer(length * elemSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufIn1,
                ctx->enqueueCreateBuffer(length * elemSize));

  EXPECT_NO_ERR(bufIn0->copyFrom(hostIn0.data()));
  EXPECT_NO_ERR(bufIn1->copyFrom(hostIn1.data()));

  void *outPtr = bufOut->ptr();
  void *in0Ptr = bufIn0->ptr();
  void *in1Ptr = bufIn1->ptr();
  void *args[] = {(void *)&outPtr, (void *)&in0Ptr, (void *)&in1Ptr,
                  (void *)&length};

  const uint32_t blockDim = 256;
  const uint32_t gridDim = (length + blockDim - 1) / blockDim;

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  DeviceFunctionLaunchParams params;
  params.gridX = gridDim;
  params.gridY = 1;
  params.gridZ = 1;
  params.blockX = blockDim;
  params.blockY = 1;
  params.blockZ = 1;
  params.sharedMemBytes = 0;
  params.args = args;
  params.argCount = 4;

  EXPECT_NO_ERR(builder->addFunctionDirect(*deviceFunc, params));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(bufOut->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i = 0; i < length; ++i)
    EXPECT_FLOAT_EQ(hostOut[i], hostIn0[i] + hostIn1[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyHostToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i = 0; i < length; ++i)
    hostIn[i] = static_cast<float>(i);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyHostToDevice(*devBuf, hostIn.data()));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i = 0; i < length; ++i)
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToHost) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i = 0; i < length; ++i)
    hostIn[i] = static_cast<float>(2 * i + 1);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(devBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(hostOut.data(), *devBuf));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i = 0; i < length; ++i)
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i = 0; i < length; ++i)
    hostIn[i] = static_cast<float>(i * i);

  ASSIGN_NO_ERR(DeviceBufferRef, srcBuf, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, dstBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(srcBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyDeviceToDevice(*dstBuf, *srcBuf));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(dstBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i = 0; i < length; ++i)
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddSetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));
  llvm::SmallVector<uint8_t> hostBuf(byteSize, 0);

  // Exercise valSize = 1, 2, 4, and symmetric 8 in a single graph by
  // recording the full sequence and replaying once. Each subsequent set
  // overwrites the previous one, so verification is against the final state.
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/123, /*valSize=*/1));
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/0xbeef, /*valSize=*/2));
  EXPECT_NO_ERR(
      builder->addSetMemory(*devBuf, /*val=*/0xdeadbeefu, /*valSize=*/4));
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/0x0101010101010101ull,
                                      /*valSize=*/8));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostBuf.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // Final state is the symmetric 8-byte pattern (0x01 in every byte).
  for (size_t i = 0; i < byteSize; ++i)
    EXPECT_EQ(hostBuf[i], 0x01u) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddSetMemoryRejectsAsymmetric8Byte) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 64;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  // High and low 32-bit halves differ; cannot be expressed as a single node.
  EXPECT_ERR_MSG(builder->addSetMemory(*devBuf,
                                       /*val=*/0x0123456789abcdefull,
                                       /*valSize=*/8),
                 "single graph node");
}

// Two graph operations that mutate the same buffer with different values.
// If the builder doesn't sequence consecutive nodes, both writes are graph
// roots and execute concurrently — the final state of the buffer is then a
// race between the memset and the host-to-device copy. With sequencing, the
// copy is forced to run after the memset and the buffer must hold the copy's
// pattern.
TEST(DeviceGraphBuilderTest, SequencesConsecutiveOpsOnSameBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 4096;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  llvm::SmallVector<uint8_t> hostIn(byteSize, 0xBBu);
  llvm::SmallVector<uint8_t> hostOut(byteSize, 0);

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/0xAAu, /*valSize=*/1));
  EXPECT_NO_ERR(builder->addCopyHostToDevice(*devBuf, hostIn.data()));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // The copy must observe a fully-written memset and overwrite every byte.
  for (size_t i = 0; i < byteSize; ++i)
    EXPECT_EQ(hostOut[i], 0xBBu) << "at index " << i;
}

} // namespace
