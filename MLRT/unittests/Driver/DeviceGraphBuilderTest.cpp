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

#include "llvm/ADT/Sequence.h"
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
  for (uint64_t i : llvm::seq(length)) {
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

  EXPECT_NO_ERR(
      builder->addFunctionDirect(*deviceFunc, params, /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(bufOut->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn0[i] + hostIn1[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyHostToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(i);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyHostToDevice(*devBuf, hostIn.data(),
                                             /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToHost) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(2 * i + 1);

  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(devBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(builder->addCopyDeviceToHost(hostOut.data(), *devBuf,
                                             /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
    EXPECT_FLOAT_EQ(hostOut[i], hostIn[i]) << "at index " << i;
}

TEST(DeviceGraphBuilderTest, AddCopyDeviceToDevice) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const uint64_t length = 1024;
  const size_t byteSize = length * sizeof(float);

  llvm::SmallVector<float> hostIn(length), hostOut(length, 0.0f);
  for (uint64_t i : llvm::seq(length))
    hostIn[i] = static_cast<float>(i * i);

  ASSIGN_NO_ERR(DeviceBufferRef, srcBuf, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, dstBuf, ctx->enqueueCreateBuffer(byteSize));
  EXPECT_NO_ERR(srcBuf->copyFrom(hostIn.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  EXPECT_NO_ERR(
      builder->addCopyDeviceToDevice(*dstBuf, *srcBuf, /*dependencies=*/{}));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(dstBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (uint64_t i : llvm::seq(length))
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
  // overwrites the previous one, so the four nodes are chained via explicit
  // dependencies and verification is against the final state.
  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceGraphNode, n0,
                builder->addSetMemory(*devBuf, /*val=*/123, /*valSize=*/1,
                                      /*dependencies=*/{}));
  ASSIGN_NO_ERR(DeviceGraphNode, n1,
                builder->addSetMemory(*devBuf, /*val=*/0xbeef, /*valSize=*/2,
                                      llvm::ArrayRef<DeviceGraphNode>(n0)));
  ASSIGN_NO_ERR(DeviceGraphNode, n2,
                builder->addSetMemory(*devBuf, /*val=*/0xdeadbeefu,
                                      /*valSize=*/4,
                                      llvm::ArrayRef<DeviceGraphNode>(n1)));
  EXPECT_NO_ERR(builder->addSetMemory(*devBuf, /*val=*/0x0101010101010101ull,
                                      /*valSize=*/8,
                                      llvm::ArrayRef<DeviceGraphNode>(n2)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostBuf.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // Final state is the symmetric 8-byte pattern (0x01 in every byte).
  for (size_t i : llvm::seq(byteSize))
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
                                       /*valSize=*/8, /*dependencies=*/{}),
                 "single graph node");
}

// Two graph operations that mutate the same buffer with different values.
// Without an explicit dependency edge from the memset to the copy, the two
// writes are graph roots and may execute concurrently — the final state
// would be a race. Adding the copy with the memset as its explicit
// predecessor forces the copy to run after the memset and the buffer must
// hold the copy's pattern.
TEST(DeviceGraphBuilderTest, ExplicitDepSequencesOpsOnSameBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 4096;
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx->enqueueCreateBuffer(byteSize));

  llvm::SmallVector<uint8_t> hostIn(byteSize, 0xBBu);
  llvm::SmallVector<uint8_t> hostOut(byteSize, 0);

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());
  ASSIGN_NO_ERR(DeviceGraphNode, memsetNode,
                builder->addSetMemory(*devBuf, /*val=*/0xAAu, /*valSize=*/1,
                                      /*dependencies=*/{}));
  EXPECT_NO_ERR(builder->addCopyHostToDevice(
      *devBuf, hostIn.data(), llvm::ArrayRef<DeviceGraphNode>(memsetNode)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  EXPECT_NO_ERR(devBuf->copyTo(hostOut.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  // The copy must observe a fully-written memset and overwrite every byte.
  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostOut[i], 0xBBu) << "at index " << i;
}

// Build two independent sequences of memsets, each on its own buffer. Each
// sequence's first node is rooted with an empty dependency list; subsequent
// nodes within a sequence carry an explicit single-element dependency on
// their immediate predecessor. The final state of each buffer must reflect
// the LAST memset of its own sequence — the two sequences are unordered
// with respect to each other but each is internally ordered.
TEST(DeviceGraphBuilderTest, IndependentMemsetSequences) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, /*id=*/0));

  const size_t byteSize = 1024;
  ASSIGN_NO_ERR(DeviceBufferRef, bufA, ctx->enqueueCreateBuffer(byteSize));
  ASSIGN_NO_ERR(DeviceBufferRef, bufB, ctx->enqueueCreateBuffer(byteSize));

  ASSIGN_NO_ERR(DeviceGraphBuilderRef, builder, ctx->createGraphBuilder());

  // Empty ArrayRef = explicit "no dependencies" (graph root).
  llvm::ArrayRef<DeviceGraphNode> rootDeps;

  // Sequence A: 0x11 -> 0x22 -> 0x33 on bufA.
  ASSIGN_NO_ERR(
      DeviceGraphNode, a0,
      builder->addSetMemory(*bufA, /*val=*/0x11u, /*valSize=*/1, rootDeps));
  ASSIGN_NO_ERR(DeviceGraphNode, a1,
                builder->addSetMemory(*bufA, /*val=*/0x22u, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(a0)));
  EXPECT_NO_ERR(builder->addSetMemory(*bufA, /*val=*/0x33u, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(a1)));

  // Sequence B: 0xAA -> 0xBB -> 0xCC on bufB. Independent of sequence A.
  ASSIGN_NO_ERR(
      DeviceGraphNode, b0,
      builder->addSetMemory(*bufB, /*val=*/0xAAu, /*valSize=*/1, rootDeps));
  ASSIGN_NO_ERR(DeviceGraphNode, b1,
                builder->addSetMemory(*bufB, /*val=*/0xBBu, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(b0)));
  EXPECT_NO_ERR(builder->addSetMemory(*bufB, /*val=*/0xCCu, /*valSize=*/1,
                                      llvm::ArrayRef<DeviceGraphNode>(b1)));

  ASSIGN_NO_ERR(DeviceGraphRef, graph, builder->instantiate());
  EXPECT_NO_ERR(graph->replay());
  EXPECT_NO_ERR(ctx->synchronize());

  llvm::SmallVector<uint8_t> hostA(byteSize, 0);
  llvm::SmallVector<uint8_t> hostB(byteSize, 0);
  EXPECT_NO_ERR(bufA->copyTo(hostA.data()));
  EXPECT_NO_ERR(bufB->copyTo(hostB.data()));
  EXPECT_NO_ERR(ctx->synchronize());

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostA[i], 0x33u) << "bufA at index " << i;

  for (size_t i : llvm::seq(byteSize))
    EXPECT_EQ(hostB[i], 0xCCu) << "bufB at index " << i;
}

} // namespace
