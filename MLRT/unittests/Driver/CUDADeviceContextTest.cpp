//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CUDA/NVML.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace M::Driver;
using M::ErrorOr;
using M::ErrorOrSuccess;

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
  static void SetUpTestSuite() {
    // Print information about the device to make it easier to debug the
    // environment.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      // Have the tests deal with device errors.
      return;

    std::cerr << "##### CUDADeviceContextTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }
};

TEST_F(CUDADeviceContextTest, DeviceExec) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::cudaAPI, /* id = */ 0));
  ASSERT_TRUE(ctx);

  // Check that the API and driver versions are available.
  // The versions may vary depending on the machine, so we don't check the
  // values.
  EXPECT_NO_ERR(ctx->device().getApiVersion());
  EXPECT_NO_ERR(ctx->device().getDriverVersion());

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

    for (size_t i = 0; i < length; ++i) {
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

    void *outPtr = outDevice->ptr();
    void *in0Ptr = in0Device->ptr();
    void *in1Ptr = in1Device->ptr();
    void *args[] = {&outPtr, &in0Ptr, &in1Ptr, (void *)&length};
    size_t argSizes[] = {sizeof(outPtr), sizeof(in0Ptr), sizeof(in1Ptr),
                         sizeof(length)};

    EXPECT_NO_ERR(ctx->enqueueFunctionExecDirect(
        *deviceFunction, length / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args, argSizes));

    EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(outHost, *outDevice));

    EXPECT_NO_ERR(ctx->synchronize());

    for (size_t i = 0; i < length; ++i) {
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

    free(outHost);
    free(in1Host);
    free(in0Host);
  }

  // Check that cached functions do not keep reference count on context. After
  // ensuring that all pending operations have terminated.
  EXPECT_NO_ERR(ctx->synchronize());
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

  for (int curDeviceId = -1; curDeviceId < numDevices; ++curDeviceId) {
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
      ASSERT_TRUE(accessible);
      EXPECT_NO_ERR(srcCtx->device().enablePeerAccess(dstCtx->device()));
      EXPECT_NO_ERR(dstCtx->device().enablePeerAccess(srcCtx->device()));
    }

    const size_t numElems = 256;
    const size_t allocLen = 4 * numElems;

    char *checkData = new char[allocLen];
    // Initialize checkData with a pattern specific for this run.
    for (size_t i = 0; i < allocLen; ++i) {
      checkData[i] = curDeviceId + i;
    }

    ASSIGN_NO_ERR(DeviceBufferRef, srcPtr, srcCtx->createHostBuffer(allocLen));
    EXPECT_TRUE(srcPtr);

    ASSIGN_NO_ERR(DeviceBufferRef, srcBuf,
                  srcCtx->enqueueCreateBuffer(allocLen));
    EXPECT_NO_ERR(srcCtx->synchronize());
    EXPECT_TRUE(srcBuf);

    ASSIGN_NO_ERR(DeviceBufferRef, dstBuf,
                  dstCtx->enqueueCreateBuffer(allocLen));
    EXPECT_NO_ERR(dstCtx->synchronize());
    EXPECT_TRUE(dstBuf);

    char *dstPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dstPtr, nullptr);
    // Initialize with known bad values.
    for (size_t i = 0; i < allocLen; ++i) {
      // Use a different factor for i than the checkData, so that we can't ever
      // overlap.
      dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(srcPtr->copyFrom(checkData));
    EXPECT_NO_ERR(srcCtx->enqueueCopyToDevice(*srcBuf, srcPtr->ptr()));

    DeviceFunction::SignatureBuilder sig =
        DeviceFunction::SignatureBuilder().addBuffer().addBuffer().addUInt64();
    ASSIGN_NO_ERR(DeviceFunctionRef, deviceFunction,
                  srcCtx->loadFunction("my_module", "_Z15p2pMemcpyKernelPfS_m",
                                       sig, p2pCopyPtx));
    EXPECT_TRUE(deviceFunction);

    constexpr uint32_t blockDim = 256;

    void *dstBufPtr = dstBuf->ptr();
    void *srcBufPtr = srcBuf->ptr();
    void *args[] = {&dstBufPtr, &srcBufPtr, (void *)&numElems};
    size_t argSizes[] = {sizeof(dstBufPtr), sizeof(srcBufPtr),
                         sizeof(numElems)};

    EXPECT_NO_ERR(srcCtx->enqueueFunctionExecDirect(
        *deviceFunction, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, args, argSizes));

    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_NO_ERR(dstCtx->enqueueCopyFromDevice(dstPtr, *dstBuf));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(dstCtx->synchronize());
    EXPECT_NO_ERR(srcCtx->synchronize());

    EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);

    delete[] checkData;
    free(dstPtr);
  }
}

// TODO(PLATENG-312): Re-enable after fixing CUDA_ERROR_ILLEGAL_ADDRESS on B200.
TEST_F(CUDADeviceContextTest, DISABLED_MultiMapMemory) {
  const char *copyPtx = R"PTX(
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

  const char *splatPtx = R"PTX(
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
.target sm_90
.address_size 64

	// .globl	_Z15p2pMemSplatKernelPfS_m

.visible .entry _Z15p2pMemSplatKernelPfS_m(
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemSplatKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemSplatKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemSplatKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemSplatKernelPfS_m_param_2];
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
	multimem.st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  const char *mapPtx = R"PTX(
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
.target sm_90
.address_size 64

	// .globl	_Z15p2pMemMapKernelPfS_m

.visible .entry _Z15p2pMemMapKernelPfS_m(
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_0,
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_1,
	.param .u64 _Z15p2pMemMapKernelPfS_m_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<2>;
	.reg .b32 	%r<5>;
	.reg .b64 	%rd<10>;


	ld.param.u64 	%rd2, [_Z15p2pMemMapKernelPfS_m_param_0];
	ld.param.u64 	%rd3, [_Z15p2pMemMapKernelPfS_m_param_1];
	ld.param.u64 	%rd4, [_Z15p2pMemMapKernelPfS_m_param_2];
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	cvt.s64.s32 	%rd1, %r4;
	setp.ge.u64 	%p1, %rd1, %rd4;
	@%p1 bra 	$L__BB0_2;

  // Did not change anything: fence.proxy.alias;
	cvta.to.global.u64 	%rd5, %rd3;
	shl.b64 	%rd6, %rd1, 2;
	add.s64 	%rd7, %rd5, %rd6;
	multimem.ld_reduce.global.sys.add.f32 	%f1, [%rd7];
	cvta.to.global.u64 	%rd8, %rd2;
	add.s64 	%rd9, %rd8, %rd6;
	st.global.f32 	[%rd9], %f1;

$L__BB0_2:
	ret;

}
)PTX";

  int numDevices = Device::numberOfDevices(Device::cudaAPI);
  std::cerr << "Number of CUDA devices: " << numDevices << "\n";
  ASSERT_GT(numDevices, 0);

  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numDevices; ++i) {
    ErrorOr<DeviceContextRef> errCtx =
        DeviceContext::create(Device::cudaAPI, i);
    ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
    DeviceContextRef ctx = std::move(errCtx.get());
    ASSERT_TRUE(ctx);
    contexts.push_back(std::move(ctx));
  }

  const DeviceContext &ctx0 = *contexts[0];

  ASSIGN_NO_ERR(DeviceFunctionRef, splatFun,
                ctx0.loadFunction("my_module", "_Z15p2pMemSplatKernelPfS_m",
                                  DeviceFunction::SignatureBuilder()
                                      .addBuffer()
                                      .addBuffer()
                                      .addUInt64(),
                                  splatPtx));

  ASSIGN_NO_ERR(DeviceFunctionRef, mapFun,
                ctx0.loadFunction("my_module", "_Z15p2pMemMapKernelPfS_m",
                                  DeviceFunction::SignatureBuilder()
                                      .addBuffer()
                                      .addBuffer()
                                      .addUInt64(),
                                  mapPtx));

  constexpr uint32_t blockDim = 16;
  const size_t numElems = 256;
  const size_t allocLen = 4 * numElems;

  ASSIGN_NO_ERR(bool, multicastSupported, ctx0.device().supportsMulticast());
  if (!multicastSupported) {
    ErrorOr<DeviceMulticastBufferRef> err =
        DeviceMulticastBuffer::allocate(contexts, allocLen);
    EXPECT_TRUE(err.isError());
    EXPECT_THAT(err.getError(),
                testing::HasSubstr("Multicast buffer not supported"));
    return;
  }

  if (numDevices < 2) {
    ErrorOr<DeviceMulticastBufferRef> err =
        DeviceMulticastBuffer::allocate(contexts, allocLen);
    EXPECT_TRUE(err.isError());
    EXPECT_THAT(err.getError(),
                testing::HasSubstr("At least two devices necessary"));
    return;
  }

  ASSIGN_NO_ERR(DeviceMulticastBufferRef, multiBuf,
                DeviceMulticastBuffer::allocate(contexts, allocLen));
  {
    std::cerr << "Using unicast buffers on each device.\n";
    for (int i = 0; i < numDevices; ++i) {
      const DeviceContext &ctx = *contexts[i];
      ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx.createHostBuffer(allocLen));
      float *hostPtr = static_cast<float *>(hostBuf->ptr());
      EXPECT_NE(hostPtr, nullptr);
      for (size_t j = 0; j < numElems; ++j) {
        hostPtr[j] = j + (3.0 * (ctx.device().id() + 1));
      }
      ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx.enqueueCreateBuffer(allocLen));
      EXPECT_NO_ERR(hostBuf->copyTo(*devBuf));
      ASSIGN_NO_ERR(DeviceBufferRef, mapBuf, multiBuf->unicastBufferFor(ctx));

      ASSIGN_NO_ERR(DeviceFunctionRef, copyFun,
                    ctx.loadFunction("my_module", "_Z15p2pMemcpyKernelPfS_m",
                                     DeviceFunction::SignatureBuilder()
                                         .addBuffer()
                                         .addBuffer()
                                         .addUInt64(),
                                     copyPtx));

      void *mapPtr = mapBuf->ptr();
      void *devPtr = devBuf->ptr();
      void *copyArgs[] = {&mapPtr, &devPtr, (void *)&numElems};
      size_t copyArgSizes[] = {sizeof(mapPtr), sizeof(devPtr),
                               sizeof(numElems)};

      EXPECT_NO_ERR(ctx.enqueueFunctionExecDirect(
          *copyFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
          /*attrs=*/nullptr, /*numAttrs=*/0, copyArgs, copyArgSizes));

      bzero(hostPtr, allocLen);
      EXPECT_NO_ERR(mapBuf->copyTo(*hostBuf));

      EXPECT_NO_ERR(ctx.synchronize());
      for (size_t j = 0; j < numElems; ++j) {
        if (j < 5) {
          std::cerr << "unicast buf[" << j << "/device(" << ctx.device().id()
                    << ")]: " << hostPtr[j] << "\n";
        }
        EXPECT_EQ(hostPtr[j], j + (3.0 * (ctx.device().id() + 1)));
      }
      std::cerr << "------------\n";
    }
  }

  // Setup multimem based tests.
  ASSIGN_NO_ERR(DeviceBufferRef, mapBuf, multiBuf->multicastBuffer(ctx0));
  ASSIGN_NO_ERR(DeviceBufferRef, devBuf, ctx0.enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx0.createHostBuffer(allocLen));
  float *hostPtr = static_cast<float *>(hostBuf->ptr());
  {
    // No synchronization with producing streams on other devices necessary
    // since the verification code above did synchronization.
    std::cerr << "Using multicast to map from all devices.\n";
    void *devPtr = devBuf->ptr();
    void *mapPtr = mapBuf->ptr();
    void *mapArgs[] = {&devPtr, &mapPtr, (void *)&numElems};
    size_t mapArgSizes[] = {sizeof(devPtr), sizeof(mapPtr), sizeof(numElems)};

    EXPECT_NO_ERR(ctx0.enqueueFunctionExecDirect(
        *mapFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, mapArgs, mapArgSizes));

    bzero(hostPtr, allocLen); // Reset the memory used to validate.
    EXPECT_NO_ERR(devBuf->copyTo(*hostBuf));
    EXPECT_NO_ERR(ctx0.synchronize());
    for (size_t j = 0; j < numElems; ++j) {
      if (j < 5) {
        std::cerr << "buf[" << j << "]: " << hostPtr[j] << "\n";
      }
      // TODO(iposva): Figure out why multimem reduction instructions do not
      // appear to work.
    }
    std::cerr << "------------\n";
  }

  {
    std::cerr << "Using multicast to splat across all devices.\n";
    for (size_t j = 0; j < numElems; ++j) {
      hostPtr[j] = j + (5.0 * j);
    }
    EXPECT_NO_ERR(hostBuf->copyTo(*devBuf));

    void *splatMapPtr = mapBuf->ptr();
    void *splatDevPtr = devBuf->ptr();
    void *splatArgs[] = {&splatMapPtr, &splatDevPtr, (void *)&numElems};
    size_t splatArgSizes[] = {sizeof(splatMapPtr), sizeof(splatDevPtr),
                              sizeof(numElems)};

    EXPECT_NO_ERR(ctx0.enqueueFunctionExecDirect(
        *splatFun, numElems / blockDim, 1, 1, blockDim, 1, 1, 0,
        /*attrs=*/nullptr, /*numAttrs=*/0, splatArgs, splatArgSizes));

    EXPECT_NO_ERR(ctx0.synchronize());
    for (int i = 0; i < numDevices; ++i) {
      const DeviceContext &ctx = *contexts[i];
      bzero(hostPtr, allocLen); // Reset the memory used to validate.
      ASSIGN_NO_ERR(DeviceBufferRef, uniBuf, multiBuf->unicastBufferFor(ctx));
      EXPECT_NO_ERR(uniBuf->copyTo(*hostBuf));
      EXPECT_NO_ERR(ctx.synchronize());
      for (size_t j = 0; j < numElems; ++j) {
        if (j < 5) {
          std::cerr << "unicast buf[" << j << "/device(" << ctx.device().id()
                    << ")]: " << hostPtr[j] << "\n";
        }
        EXPECT_EQ(hostPtr[j], j + (5.0 * j));
      }
      std::cerr << "------------\n";
    }
  }
}

TEST_F(CUDADeviceContextTest, NVMLGetPciInfo) {
  // Get NVML instance
  ErrorOr<const NVML &> nvmlOr = NVML::get();
  ASSERT_FALSE(nvmlOr.isError()) << nvmlOr.getError();
  const NVML &nvml = *nvmlOr;

  // Get device handle for GPU 0
  nvmlDevice_t device;
  ASSERT_EQ(nvml.nvmlDeviceGetHandleByIndex(0, &device), NVML::SUCCESS);

  // Retrieve PCI info
  nvmlPciInfo_t pciInfo;
  ASSERT_EQ(nvml.nvmlDeviceGetPciInfo_v3(device, &pciInfo), NVML::SUCCESS);

  // Validate PCI info fields
  EXPECT_GT(strlen(pciInfo.busId), 0u);
  EXPECT_LE(pciInfo.bus, 0xffu);
  EXPECT_LE(pciInfo.device, 31u);
}

} // namespace
