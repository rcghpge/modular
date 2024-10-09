//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

TEST(CUDADeviceContextTest, Smoke) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  std::string device_name = ctx->device().name();
  EXPECT_NE(device_name.find("NVIDIA"), std::string::npos) << device_name;
}

TEST(CUDADeviceContextTest, NumDevices) {
  ErrorOr<int> err_num = Device::numberOfDevices("cuda");
  ASSERT_FALSE(err_num) << err_num.getError() << "\n";
  EXPECT_GT(err_num.get(), 0);
  EXPECT_LT(err_num.get(), 16);
  std::cerr << "Number of CUDA devices: " << err_num.get() << "\n";
}

static void init_with_data(void *dst, size_t dst_len, const void *check_data,
                           size_t check_data_len) {
  char *cur = static_cast<char *>(dst);
  size_t remaining = dst_len;
  while (remaining > 0) {
    size_t to_copy = std::min(remaining, check_data_len);
    memcpy(cur, check_data, to_copy);
    remaining -= to_copy;
    cur += to_copy;
  }
}

static void compare_with_data(const void *dst, size_t dst_len,
                              const void *check_data, size_t check_data_len) {
  const char *cur = static_cast<const char *>(dst);
  size_t remaining = dst_len;
  while (remaining > 0) {
    size_t to_check = std::min(remaining, check_data_len);
    EXPECT_EQ(memcmp(cur, check_data, to_check), 0);
    remaining -= to_check;
    cur += to_check;
  }
}

TEST(CUDADeviceContextTest, DeviceBuffers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  char check_data[] = {
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '8', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '9', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', 'A', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'C', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
      'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
      'E',
  };

  const int copy_lengths[] = {256, 1024 * 1024};

  for (int i = 0; i < 4; i++) {
    bool sync_copies = i & 1;
    int alloc_len = copy_lengths[(i >> 1) & 1];

    std::cerr << "DeviceContext copies(len: " << alloc_len
              << ", sync: " << sync_copies << ")\n";

    ErrorOr<void *> err_ptr = ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *a_ptr = err_ptr.get();
    EXPECT_NE(a_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref = ctx->createBuffer(alloc_len, 1);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    err_ref = ctx->enqueueCreateBuffer(alloc_len / 16, 16);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef c_buf = std::move(err_ref.get());
    EXPECT_TRUE(c_buf);

    char *d_ptr = static_cast<char *>(malloc(alloc_len));
    EXPECT_NE(d_ptr, nullptr);

    // Initialize with known good and bad values.
    init_with_data(a_ptr, alloc_len, check_data, sizeof(check_data));

    for (int i = 0; i < alloc_len; i++) {
      d_ptr[i] = -static_cast<char>((7 + i) & 127);
    }

    ErrorOrSuccess err;
    if (sync_copies) {
      // Ensure that all setup has finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";

      // Copy between different buffers.
      err = ctx->copyToDevice(*b_buf, a_ptr);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->copyDeviceToDevice(*c_buf, *b_buf);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->copyFromDevice(d_ptr, *c_buf);
      EXPECT_FALSE(err) << err.getError() << "\n";
    } else {
      // Copy between different buffers.
      err = ctx->enqueueCopyToDevice(*b_buf, a_ptr);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->enqueueCopyDeviceToDevice(*c_buf, *b_buf);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->enqueueCopyFromDevice(d_ptr, *c_buf);
      EXPECT_FALSE(err) << err.getError() << "\n";

      // Ensure that all copy operations have finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";
    }
    compare_with_data(d_ptr, alloc_len, check_data, sizeof(check_data));

    free(d_ptr);
    err = ctx->free_host(a_ptr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }
}

TEST(CUDADeviceContextTest, DeviceExec) {
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

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  std::cerr << "Compiling vec_add_ptx...\n";
  DeviceFunction::SignatureBuilder sig = DeviceFunction::SignatureBuilder()
                                             .addBuffer()
                                             .addBuffer()
                                             .addBuffer()
                                             .addUInt64();
  ErrorOr<DeviceFunctionRef> err_fun =
      ctx->compileFunction("vec_add", sig, vec_add_ptx);
  EXPECT_FALSE(err_fun) << err_fun.getError() << "\n";
  DeviceFunctionRef device_function = std::move(err_fun.get());
  EXPECT_TRUE(device_function);

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
      ctx->enqueueCreateBuffer(length, elem_size);
  EXPECT_FALSE(err_buf) << err_buf.getError() << "\n";
  DeviceBufferRef in0_device = std::move(err_buf.get());
  EXPECT_TRUE(in0_device);

  err_buf = ctx->enqueueCreateBuffer(length, elem_size);
  EXPECT_FALSE(err_buf) << err_buf.getError() << "\n";
  DeviceBufferRef in1_device = std::move(err_buf.get());
  EXPECT_TRUE(in1_device);

  err_buf = ctx->enqueueCreateBuffer(length, elem_size);
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
  uint32_t ref_count = device_function->getNumReferences();
  // One refcount for device_function and one ref count for the cache in
  // CUDADeviceContext.
  EXPECT_EQ(ref_count, 2);
  {
    ErrorOr<DeviceFunctionRef> err_fun =
        ctx->compileFunction("vec_add", sig, vec_add_ptx);
    EXPECT_FALSE(err_fun) << err_fun.getError() << "\n";
    DeviceFunctionRef device_function2 = std::move(err_fun.get());
    EXPECT_TRUE(device_function2);

    EXPECT_EQ(device_function.getPointer(), device_function2.getPointer());
    // Additional refcount for device_function2.
    EXPECT_EQ(ref_count + 1, device_function->getNumReferences());
    EXPECT_EQ(ref_count + 1, device_function2->getNumReferences());
  }
  // Make sure the local reference for device_function2 has been released.
  EXPECT_EQ(ref_count, device_function->getNumReferences());

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
      strcmp(err.getError(),
             "Argument count mismatch at 5th position: Expected 4 arguments."),
      0)
      << err.getError();

  free(out_host);
  free(in1_host);
  free(in0_host);
}

TEST(CUDADeviceContextTest, Timers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  ErrorOr<DeviceTimerRef> err_timer = ctx->startTimer();
  EXPECT_FALSE(err_timer) << err_timer.getError() << "\n";
  DeviceTimerRef timer = std::move(err_timer.get());
  ASSERT_TRUE(timer);

  { // Run some memory copies within the timer.
    const size_t alloc_len = 1024 * 1024;

    ErrorOr<void *> err_ptr = ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *a_ptr = err_ptr.get();
    EXPECT_NE(a_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref = ctx->enqueueCreateBuffer(256, 1);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    ErrorOrSuccess err;
    for (int i = 0; i < 20; i++) {
      err = ctx->enqueueCopyToDevice(*b_buf, a_ptr);
      EXPECT_FALSE(err) << err.getError() << "\n";
    }
    err = ctx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = ctx->free_host(a_ptr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }

  ErrorOr<std::chrono::nanoseconds> err_nanos = ctx->stopTimer(*timer);
  EXPECT_FALSE(err_nanos) << err_nanos.getError() << "\n";
  int64_t elapsed_nanos = err_nanos.get().count();

  std::cerr << "Timer measured: " << elapsed_nanos << "ns\n";

  EXPECT_GT(elapsed_nanos, 0);
  EXPECT_LT(elapsed_nanos, 60 * 1000ll * 1000ll * 1000ll); // less than a minute
}

TEST(CUDADeviceContextTest, MultiContextCopies) {
  ErrorOr<int> err_num = Device::numberOfDevices("cuda");
  ASSERT_FALSE(err_num) << err_num.getError() << "\n";
  int num_devices = err_num.get();
  std::cerr << "Number of CUDA devices: " << num_devices << "\n";

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda", 0);
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
          DeviceContext::create("cuda", cur_device_id);
      ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
      dst_ctx = std::move(err_ctx.get());
    }
    ASSERT_TRUE(dst_ctx);

    char check_data[] = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
        'D', 'E', 'F', '0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'E', 'F', '0', '1', '3', '4', '5', '6', '7', '8', '9', 'A',
        'B', 'C', 'D', 'E', 'F', '0', '1', '2', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '5', '6', '7', '8',
        '9', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '6', '7',
        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4', '5',
        '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3', '4',
        '5', '6', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2', '3',
        '4', '5', '6', '7', '9', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1', '2',
        '3', '4', '5', '6', '7', '8', 'A', 'B', 'C', 'D', 'E', 'F', '0', '1',
        '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E', 'F', '0',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'C', 'D', 'E', 'F',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'E',
        'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
        'E', 'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
        'C', 'D', 'F', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A',
        'B', 'C', 'D', 'E',
    };

    const int alloc_len = 256;

    ErrorOr<void *> err_ptr = src_ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *src_ptr = err_ptr.get();
    EXPECT_NE(src_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref =
        src_ctx->enqueueCreateBuffer(alloc_len, 1);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef src_buf = std::move(err_ref.get());
    EXPECT_TRUE(src_buf);

    err_ref = dst_ctx->enqueueCreateBuffer(alloc_len, 1);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef dst_buf = std::move(err_ref.get());
    EXPECT_TRUE(dst_buf);

    char *dst_ptr = static_cast<char *>(malloc(alloc_len));
    EXPECT_NE(dst_ptr, nullptr);
    // Initialize with known bad values.
    for (int i = 0; i < alloc_len; i++) {
      dst_ptr[i] = -static_cast<char>((7 + i) & 127);
    }

    // Copy between different buffers.
    memcpy(src_ptr, check_data, alloc_len);
    ErrorOrSuccess err = src_ctx->enqueueCopyToDevice(*src_buf, src_ptr);
    EXPECT_FALSE(err) << err.getError() << "\n";
    // Make sure to use the dst_ctx here to properly enqueue future work on
    // dst_ctx within the stream.
    err = dst_ctx->enqueueCopyDeviceToDevice(*dst_buf, *src_buf);
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
