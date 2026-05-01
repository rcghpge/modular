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

} // namespace
