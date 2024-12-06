//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/include/AsyncRT/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

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
  const char *vec_add_ptx = R"PTX(
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

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create(Device::cudaAPI);
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  // Check reference count.
  EXPECT_EQ(ctx->getNumReferences(), 1);

  {
    std::cerr << "Compiling vec_add_ptx...\n";
    DeviceFunction::SignatureBuilder sig = DeviceFunction::SignatureBuilder()
                                               .addBuffer()
                                               .addBuffer()
                                               .addBuffer()
                                               .addUInt64();
    ErrorOr<DeviceFunctionRef> err_fun =
        ctx->loadFunction("vec_add_module", "vec_add", sig, vec_add_ptx);
    EXPECT_FALSE(err_fun) << err_fun.getError() << "\n";
    DeviceFunctionRef device_function = std::move(err_fun.get());
    EXPECT_TRUE(device_function);

    // One reference count from ctx and one from device_function.
    EXPECT_EQ(ctx->getNumReferences(), 2);
    EXPECT_EQ(device_function->getNumReferences(), 1);

    constexpr size_t length = 1024;
    constexpr size_t elem_size = sizeof(float);

    float *in0_host = reinterpret_cast<float *>(malloc(length * elem_size));
    float *in1_host = reinterpret_cast<float *>(malloc(length * elem_size));
    float *out_host = reinterpret_cast<float *>(malloc(length * elem_size));

    for (size_t i = 0; i < length; i++) {
      in0_host[i] = static_cast<float>(i);
      in1_host[i] = 2.0;
      out_host[i] =
          static_cast<float>(i) * 2.0; // Write a value that is not expected.
    }

    ErrorOr<DeviceBufferRef> err_buf =
        ctx->enqueueCreateBuffer(length * elem_size);
    EXPECT_FALSE(err_buf) << err_buf.getError() << "\n";
    DeviceBufferRef in0_device = std::move(err_buf.get());
    EXPECT_TRUE(in0_device);

    err_buf = ctx->enqueueCreateBuffer(length * elem_size);
    EXPECT_FALSE(err_buf) << err_buf.getError() << "\n";
    DeviceBufferRef in1_device = std::move(err_buf.get());
    EXPECT_TRUE(in1_device);

    err_buf = ctx->enqueueCreateBuffer(length * elem_size);
    EXPECT_FALSE(err_buf) << err_buf.getError() << "\n";
    DeviceBufferRef out_device = std::move(err_buf.get());
    EXPECT_TRUE(out_device);

    ErrorOrSuccess err = ctx->enqueueCopyToDevice(*in0_device, in0_host);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = ctx->enqueueCopyToDevice(*in1_device, in1_host);
    EXPECT_FALSE(err) << err.getError() << "\n";
    // This copy is not strictly necessary, but fill out_device with known bad
    // values.
    err = ctx->enqueueCopyToDevice(*out_device, out_host);
    EXPECT_FALSE(err) << err.getError() << "\n";

    constexpr uint32_t block_dim = 32;

    err = ctx->enqueueFunctionExec(
        *device_function, length / block_dim, 1, 1, block_dim, 1, 1, 0,
        *device_function->buildArguments(*out_device, *in0_device, *in1_device,
                                         length));
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = ctx->enqueueCopyFromDevice(out_host, *out_device);
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = ctx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    for (size_t i = 0; i < length; i++) {
      EXPECT_EQ(out_host[i], static_cast<float>(i) + 2.0);
    }

    // Check cache hit.
    {
      ErrorOr<DeviceFunctionRef> err_fun =
          ctx->loadFunction("vec_add_module", "vec_add", sig, vec_add_ptx);
      EXPECT_FALSE(err_fun) << err_fun.getError() << "\n";
      DeviceFunctionRef device_function2 = std::move(err_fun.get());
      EXPECT_TRUE(device_function2);

      // A cache hit will point to the same backing store for the function (e.g.
      // name).
      EXPECT_EQ(&device_function->name(), &device_function2->name());
      // Even with a cache hit, we do get different DeviceFunction objects.
      EXPECT_EQ(device_function->getNumReferences(), 1);
      EXPECT_EQ(device_function2->getNumReferences(), 1);
    }

    // Check error cases of mismatched arguments being passed.
    err = ctx->enqueueFunctionExec(
        *device_function, length / block_dim, 1, 1, block_dim, 1, 1, 0,
        *device_function->buildArguments(*out_device, *in0_device, 12, length));
    EXPECT_TRUE(err);
    EXPECT_EQ(strcmp(err.getError(), "Argument mismatch at 3th position: "
                                     "Expected DeviceBuffer, got uint64."),
              0)
        << err.getError();

    err = ctx->enqueueFunctionExec(
        *device_function, length / block_dim, 1, 1, block_dim, 1, 1, 0,
        *device_function->buildArguments(*out_device, *in0_device, *in1_device,
                                         *in1_device));
    EXPECT_TRUE(err);
    EXPECT_EQ(strcmp(err.getError(), "Argument mismatch at 4th position: "
                                     "Expected uint64, got DeviceBuffer."),
              0)
        << err.getError();

    err = ctx->enqueueFunctionExec(
        *device_function, length / block_dim, 1, 1, block_dim, 1, 1, 0,
        *device_function->buildArguments(*out_device, *in0_device, *in1_device,
                                         length, length));
    EXPECT_TRUE(err);
    EXPECT_EQ(
        strcmp(
            err.getError(),
            "Argument count mismatch at 5th position: Expected 4 arguments."),
        0)
        << err.getError();

    free(out_host);
    free(in1_host);
    free(in0_host);
  }

  // Check cached functions do not keep reference count on context.
  EXPECT_EQ(ctx->getNumReferences(), 1);
}

TEST_F(CUDADeviceContextTest, MultiGPUKernels) {
  const char *p2p_copy_ptx = R"PTX(
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

  int num_devices = Device::numberOfDevices(Device::cudaAPI);
  std::cerr << "Number of CUDA devices: " << num_devices << "\n";

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create(Device::cudaAPI, 0);
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef src_ctx = std::move(err_ctx.get());
  ASSERT_TRUE(src_ctx);

  for (int cur_device_id = -1; cur_device_id < num_devices; cur_device_id++) {
    DeviceContextRef dst_ctx;
    if (cur_device_id < 0) {
      std::cerr << "Copying within the same device context.\n";
      dst_ctx = DeviceContextRef::copy(src_ctx.getPointer());
    } else {
      std::cerr << "Copying from context on GPU[0] to context on GPU["
                << cur_device_id << "].\n";

      ErrorOr<DeviceContextRef> err_ctx =
          DeviceContext::create(Device::cudaAPI, cur_device_id);
      ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
      dst_ctx = std::move(err_ctx.get());
    }
    ASSERT_TRUE(dst_ctx);

    if (src_ctx->device().id() != dst_ctx->device().id()) {
      ErrorOr<bool> err_bool = src_ctx->device().canAccess(dst_ctx->device());
      ASSERT_FALSE(err_bool) << err_bool.getError() << "\n";
      ASSERT_TRUE(err_bool.get());

      ErrorOrSuccess err = src_ctx->enablePeerAccess(*dst_ctx);
      ASSERT_FALSE(err) << err.getError() << "\n";

      err = dst_ctx->enablePeerAccess(*src_ctx);
      ASSERT_FALSE(err) << err.getError() << "\n";
    }

    const size_t num_elems = 256;
    const size_t alloc_len = 4 * num_elems;

    char *check_data = new char[alloc_len];
    // Initialize check_data with a pattern specific for this run.
    for (size_t i = 0; i < alloc_len; i++) {
      check_data[i] = cur_device_id + i;
    }

    ErrorOr<void *> err_ptr = src_ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *src_ptr = err_ptr.get();
    EXPECT_NE(src_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref = src_ctx->createBuffer(alloc_len);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef src_buf = std::move(err_ref.get());
    EXPECT_TRUE(src_buf);

    err_ref = dst_ctx->createBuffer(alloc_len);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef dst_buf = std::move(err_ref.get());
    EXPECT_TRUE(dst_buf);

    char *dst_ptr = static_cast<char *>(malloc(alloc_len));
    EXPECT_NE(dst_ptr, nullptr);
    // Initialize with known bad values.
    for (size_t i = 0; i < alloc_len; i++) {
      // Use a different factor for i than the check_data, so that we can't ever
      // overlap.
      dst_ptr[i] = -static_cast<char>((7 + 2 * i) & 127);
    }

    // Copy between different buffers.
    memcpy(src_ptr, check_data, alloc_len);
    ErrorOrSuccess err = src_ctx->copyToDevice(*src_buf, src_ptr);
    EXPECT_FALSE(err) << err.getError() << "\n";

    DeviceFunction::SignatureBuilder sig =
        DeviceFunction::SignatureBuilder().addBuffer().addBuffer().addUInt64();
    ErrorOr<DeviceFunctionRef> err_fun = src_ctx->loadFunction(
        "my_module", "_Z15p2pMemcpyKernelPfS_m", sig, p2p_copy_ptx);
    EXPECT_FALSE(err_fun) << err_fun.getError() << "\n";
    DeviceFunctionRef device_function = std::move(err_fun.get());
    EXPECT_TRUE(device_function);

    constexpr uint32_t block_dim = 256;

    err = src_ctx->enqueueFunctionExec(
        *device_function, num_elems / block_dim, 1, 1, block_dim, 1, 1, 0,
        *device_function->buildArguments(*dst_buf, *src_buf, num_elems));
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = src_ctx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = dst_ctx->enqueueCopyFromDevice(dst_ptr, *dst_buf);
    EXPECT_FALSE(err) << err.getError() << "\n";

    // Ensure that all copy operations have finished.
    err = dst_ctx->synchronize();
    err = src_ctx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    EXPECT_EQ(memcmp(dst_ptr, check_data, alloc_len), 0);

    free(dst_ptr);
    err = src_ctx->free_host(src_ptr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }
}

} // namespace
