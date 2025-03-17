//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

static const char *vecAddPtx = R"PTX(
// From Kernels/test/gpu/vec_add.ptx

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

class CUDADeviceContextTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Allocate the GPU device for this platform. Then check that we are running
    // on a CUDA enabled platform, if not skip the CUDA specific tests in this
    // fixture.

    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      // Have the tests deal with device errors.
      return;

    if ((*device)->api() != Device::cudaAPI) {
      GTEST_SKIP() << "Skipping CUDA tests for GPU \"" << (*device)->vendor()
                   << ":" << (*device)->api() << "\"";
    }
  }
};

TEST_F(CUDADeviceContextTest, DeviceExec) {
  // Disable the buffer cache to avoid reference counts from the memory blocks
  // confusing our accounting here.
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /* id = */ 0,
                                      /* bufferCacheSize = */ 0));
  ASSERT_TRUE(ctx);

  // Check reference count.
  EXPECT_EQ(ctx->getNumReferences(), 1);

  {
    std::cerr << "Compiling vecAddPtx...\n";
    DeviceFunction::SignatureBuilder sig = DeviceFunction::SignatureBuilder()
                                               .addBuffer()
                                               .addBuffer()
                                               .addBuffer()
                                               .addUInt64();
    ASSIGN_NO_ERR(
        DeviceFunctionRef, deviceFunction,
        ctx->loadFunction("vec_add_module", "vec_add", sig, vecAddPtx));

    // One reference count from ctx and one from deviceFunction.
    EXPECT_EQ(ctx->getNumReferences(), 2);
    EXPECT_EQ(deviceFunction->getNumReferences(), 1);

    constexpr size_t length = 1024;
    constexpr size_t elemSize = sizeof(float);

    float *in0Host = reinterpret_cast<float *>(malloc(length * elemSize));
    float *in1Host = reinterpret_cast<float *>(malloc(length * elemSize));
    float *outHost = reinterpret_cast<float *>(malloc(length * elemSize));

    for (size_t i = 0; i < length; i++) {
      in0Host[i] = static_cast<float>(i);
      in1Host[i] = 2.0;
      outHost[i] =
          static_cast<float>(i) * 2.0; // Write a value that is not expected.
    }

    ASSIGN_NO_ERR(DeviceBufferRef, in0Device,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(in0Device);

    ASSIGN_NO_ERR(DeviceBufferRef, in1Device,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(in1Device);

    ASSIGN_NO_ERR(DeviceBufferRef, outDevice,
                  ctx->enqueueCreateBuffer(length * elemSize));
    EXPECT_TRUE(outDevice);

    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*in0Device, in0Host));
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*in1Device, in1Host));
    // This copy is not strictly necessary, but fill outDevice with known bad
    // values.
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*outDevice, outHost));

    constexpr uint32_t blockDim = 32;

    EXPECT_NO_ERR(ctx->enqueueFunctionExec(
        *deviceFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0,
        *deviceFunction->buildArguments(*outDevice, *in0Device, *in1Device,
                                        length)));

    EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(outHost, *outDevice));

    EXPECT_NO_ERR(ctx->synchronize());

    for (size_t i = 0; i < length; i++) {
      EXPECT_EQ(outHost[i], static_cast<float>(i) + 2.0);
    }

    // Check cache hit.
    {
      ASSIGN_NO_ERR(
          DeviceFunctionRef, deviceFunction2,
          ctx->loadFunction("vec_add_module", "vec_add", sig, vecAddPtx));
      EXPECT_TRUE(deviceFunction2);

      // A cache hit will point to the same backing store for the function (e.g.
      // name).
      EXPECT_EQ(&deviceFunction->name(), &deviceFunction2->name());
      // Even with a cache hit, we do get different DeviceFunction objects.
      EXPECT_EQ(deviceFunction->getNumReferences(), 1);
      EXPECT_EQ(deviceFunction2->getNumReferences(), 1);
    }

    // Check error cases of mismatched arguments being passed.
    EXPECT_ERR_MSG(
        ctx->enqueueFunctionExec(*deviceFunction, length / blockDim, 1, 1,
                                 blockDim, 1, 1, 0,
                                 *deviceFunction->buildArguments(
                                     *outDevice, *in0Device, 12, length)),
        "Argument mismatch at 3th position: "
        "Expected DeviceBuffer, got uint64.");

    EXPECT_ERR_MSG(ctx->enqueueFunctionExec(
                       *deviceFunction, length / blockDim, 1, 1, blockDim, 1, 1,
                       0,
                       *deviceFunction->buildArguments(*outDevice, *in0Device,
                                                       *in1Device, *in1Device)),
                   "Argument mismatch at 4th position: "
                   "Expected uint64, got DeviceBuffer.");

    EXPECT_ERR_MSG(
        ctx->enqueueFunctionExec(
            *deviceFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0,
            *deviceFunction->buildArguments(*outDevice, *in0Device, *in1Device,
                                            length, length)),
        "Argument count mismatch at 5th position: Expected 4 arguments.");

    free(outHost);
    free(in1Host);
    free(in0Host);
  }

  // Check cached functions do not keep reference count on context.
  EXPECT_EQ(ctx->getNumReferences(), 1);
}

TEST_F(CUDADeviceContextTest, MultiGPUKernels) {
  const char *p2pCopyPtx = R"PTX(
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-32688072
// Cuda compilation tools, release 12.1, V12.1.105
// Based on NVVM 7.0.1
//
// Compiled by nvcc -ptx
// Simple copy kernel
// __global__ void p2pMemcpyKernel(float* dst, float* src, size_t numElements) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < numElements) {
//         dst[idx] = src[idx];
//     }
// }

.version 8.1
.target sm_52
.address_size 64

	// .globl	_Z15p2pMemcpyKernelPfS_m

.visible .entry _Z15p2pMemcpyKernelPfS_m(
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemcpyKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemcpyKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemcpyKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemcpyKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	ld.global.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  std::cerr << "Number of CUDA devices: " << numDevices << "\n";

  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::cudaAPI, 0));
  ASSERT_TRUE(srcCtx);

  for (int curDeviceId = -1; curDeviceId < numDevices; curDeviceId++) {
    DeviceContextRef dstCtx;
    if (curDeviceId < 0) {
      std::cerr << "Copying within the same device context.\n";
      dstCtx = DeviceContextRef::copy(srcCtx.getPointer());
    } else {
      std::cerr << "Copying from context on GPU[0] to context on GPU["
                << curDeviceId << "].\n";

      ErrorOr<DeviceContextRef> errCtx =
          DeviceContext::create(Device::cudaAPI, curDeviceId);
      ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
      dstCtx = std::move(errCtx.get());
    }
    ASSERT_TRUE(dstCtx);

    if (srcCtx->device().id() != dstCtx->device().id()) {
      ASSIGN_NO_ERR(bool, accessible,
                    srcCtx->device().canAccess(dstCtx->device()));
      if (!accessible) {
        // TODO(SDLC-1831): Remove this once drivers are updated.
        GTEST_SKIP() << "Skipping MultiGPU kernel test because GPU["
                     << curDeviceId << "] is not accessible from GPU[0].";
      }

      EXPECT_NO_ERR(srcCtx->enablePeerAccess(*dstCtx));
      EXPECT_NO_ERR(dstCtx->enablePeerAccess(*srcCtx));
    }

    const size_t numElems = 256;
    const size_t allocLen = 4 * numElems;

    char *checkData = new char[allocLen];
    // Initialize checkData with a pattern specific for this run.
    for (size_t i = 0; i < allocLen; i++) {
      checkData[i] = curDeviceId + i;
    }

    ASSIGN_NO_ERR(void *, srcPtr, srcCtx->mallocHost(allocLen));
    EXPECT_NE(srcPtr, nullptr);

    ASSIGN_NO_ERR(DeviceBufferRef, srcBuf, srcCtx->createBuffer(allocLen));
    EXPECT_TRUE(srcBuf);

    ASSIGN_NO_ERR(DeviceBufferRef, dstBuf, dstCtx->createBuffer(allocLen));
    EXPECT_TRUE(dstBuf);

    char *dstPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dstPtr, nullptr);
    // Initialize with known bad values.
    for (size_t i = 0; i < allocLen; i++) {
      // Use a different factor for i than the checkData, so that we can't ever
      // overlap.
      dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
    }

    // Copy between different buffers.
    memcpy(srcPtr, checkData, allocLen);
    EXPECT_NO_ERR(srcCtx->copyToDevice(*srcBuf, srcPtr));

    DeviceFunction::SignatureBuilder sig =
        DeviceFunction::SignatureBuilder().addBuffer().addBuffer().addUInt64();
    ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunction,
                  srcCtx->loadFunction("my_module", "_Z15p2pMemcpyKernelPfS_m",
                                       sig, p2pCopyPtx));
    EXPECT_TRUE(deviceFunction);

    constexpr uint32_t blockDim = 256;

    EXPECT_NO_ERR(srcCtx->enqueueFunctionExec(
        *deviceFunction, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        *deviceFunction->buildArguments(*dstBuf, *srcBuf, numElems)));

    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_NO_ERR(dstCtx->enqueueCopyFromDevice(dstPtr, *dstBuf));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(dstCtx->synchronize());
    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);

    delete[] checkData;
    free(dstPtr);
    EXPECT_NO_ERR(srcCtx->freeHost(srcPtr));
  }
}

} // namespace
