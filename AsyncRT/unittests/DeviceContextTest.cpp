//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

TEST(DeviceContextTest, Smoke) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  EXPECT_EQ(ctx->device().name(), "cpu");
}

TEST(DeviceContextTest, CreationFailure) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu", 1);
  ASSERT_TRUE(err_ctx);
  EXPECT_EQ(strcmp(err_ctx.getError(), "CPU device only supports id == 0."), 0)
      << err_ctx.getError();

  err_ctx = DeviceContext::create("pcu");
  ASSERT_TRUE(err_ctx);
  EXPECT_EQ(strcmp(err_ctx.getError(), "Unsupported device kind: pcu"), 0)
      << err_ctx.getError();
}

TEST(DeviceContextTest, NumDevices) {
  ErrorOr<int> err_num = Device::numberOfDevices("cpu");
  ASSERT_FALSE(err_num) << err_num.getError() << "\n";
  EXPECT_EQ(err_num.get(), 1);
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

TEST(DeviceContextTest, DeviceBuffers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
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
    DeviceContext::SyncMode sync_mode =
        ((i & 1) == 0) ? DeviceContext::SYNC : DeviceContext::ASYNC;
    int alloc_len = copy_lengths[(i >> 1) & 1];

    std::cerr << "DeviceContext copies(len: " << alloc_len << ", sync_mode: "
              << ((sync_mode == DeviceContext::SYNC) ? "SYNC" : "ASYNC")
              << ")\n";

    ErrorOr<void *> err_ptr = ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *a_ptr = err_ptr.get();
    EXPECT_NE(a_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref =
        ctx->createBuffer(alloc_len, 1, sync_mode);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    err_ref = ctx->createBuffer(alloc_len / 16, 16, sync_mode);
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
    // Copy between different buffers.
    err = ctx->copyToDevice(*b_buf, a_ptr, sync_mode);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = ctx->copyDeviceToDevice(*c_buf, *b_buf, sync_mode);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = ctx->copyFromDevice(d_ptr, *c_buf, sync_mode);
    EXPECT_FALSE(err) << err.getError() << "\n";

    if (sync_mode == DeviceContext::ASYNC) {
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

TEST(DeviceContextTest, FailDeviceBuffers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  const int copy_lengths[] = {256};

  for (int i = 0; i < 2; i++) {
    DeviceContext::SyncMode sync_mode =
        ((i & 1) == 0) ? DeviceContext::SYNC : DeviceContext::ASYNC;
    int alloc_len = copy_lengths[(i >> 1) & 1];

    std::cerr << "DeviceContext failing copies(len: " << alloc_len
              << ", sync_mode: "
              << ((sync_mode == DeviceContext::SYNC) ? "SYNC" : "ASYNC")
              << ")\n";

    char *a_ptr = static_cast<char *>(malloc(alloc_len));
    EXPECT_NE(a_ptr, nullptr);

    // Create a source buffer that is too small.
    ErrorOr<DeviceBufferRef> err_ref =
        ctx->createBuffer(alloc_len - 32, 1, sync_mode);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    err_ref = ctx->createBuffer(alloc_len / 16, 16, sync_mode);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef c_buf = std::move(err_ref.get());
    EXPECT_TRUE(c_buf);

    // Initialize with known good and bad values.
    for (int i = 0; i < alloc_len; i++) {
      a_ptr[i] = static_cast<char>(i & 127);
    }

    ErrorOrSuccess err;
    // Copy between different buffers.
    err = ctx->copyToDevice(*b_buf, a_ptr, sync_mode);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = ctx->copyDeviceToDevice(*c_buf, *b_buf, sync_mode);
    EXPECT_TRUE(err);
    EXPECT_THAT(err.getError(), testing::HasSubstr("not enough data in src"));

    if (sync_mode == DeviceContext::ASYNC) {
      // Ensure that all copy operations have finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";
    }

    free(a_ptr);
  }
}

TEST(CUDADeviceContextTest, SubBuffer) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  int alloc_len = 256;

  int sub_offset = 0;
  int sub_alloc_len = 3;

  std::cerr << "DeviceContext sub buffer copies(len: " << alloc_len << ")\n";

  char *a_ptr = static_cast<char *>(malloc(alloc_len));
  EXPECT_NE(a_ptr, nullptr);

  ErrorOr<DeviceBufferRef> err_ref = ctx->enqueueCreateBuffer(alloc_len, 1);
  EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
  DeviceBufferRef b_buf = std::move(err_ref.get());
  EXPECT_TRUE(b_buf);

  err_ref = b_buf->createSubBuffer(sub_offset, sub_alloc_len, 4);
  EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
  DeviceBufferRef c_buf = std::move(err_ref.get());
  EXPECT_TRUE(c_buf);

  char *d_ptr = static_cast<char *>(malloc(alloc_len));
  EXPECT_NE(d_ptr, nullptr);

  float *e_ptr = static_cast<float *>(malloc(sub_alloc_len * sizeof(float)));
  EXPECT_NE(e_ptr, nullptr);

  // Initialize with known good and bad values.
  for (int i = 0; i < alloc_len; i++) {
    a_ptr[i] = static_cast<char>(i & 127);
    d_ptr[i] = -static_cast<char>((7 + i) & 127);
  }

  for (int i = 0; i < sub_alloc_len; i++) {
    e_ptr[i] = static_cast<float>(i) * (((i & 1) == 0) ? -1.0 : 1.0);
  }

  // Copy base pattern into base buffer.
  ErrorOrSuccess err = ctx->enqueueCopyToDevice(*b_buf, a_ptr);
  EXPECT_FALSE(err) << err.getError() << "\n";

  // Copy overlay pattern into sub buffer.
  err = ctx->enqueueCopyToDevice(*c_buf, e_ptr);
  EXPECT_FALSE(err) << err.getError() << "\n";

  // Copy base buffer back to host memory.
  err = ctx->enqueueCopyFromDevice(d_ptr, *b_buf);
  EXPECT_FALSE(err) << err.getError() << "\n";

  uint32_t *d_int_ptr = reinterpret_cast<uint32_t *>(d_ptr);
  uint32_t expected_vals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (sub_alloc_len); i++) {
    uint32_t val = d_int_ptr[i];
    std::cerr << "SubBuffer: d[" << std::dec << i << "] = 0x" << std::hex << val
              << "\n";
    EXPECT_EQ(val, expected_vals[i]);
  }
  for (int i = sub_alloc_len * 4; i < alloc_len; i++) {
    int val = d_ptr[i];
    if (i < 20) {
      std::cerr << "Original: d[" << std::dec << i << "] = 0x" << std::hex
                << val << "\n";
    }
    EXPECT_EQ(val, i & 127);
  }

  free(e_ptr);
  free(d_ptr);
  free(a_ptr);
}

TEST(DeviceContextTest, DeviceExec) {
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

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
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
  // Expecting for an error to be set in the case of CPU device context.
  ASSERT_TRUE(err_fun) << err_fun.getError() << "\n";
}

TEST(DeviceContextTest, Timers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cpu");
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

} // namespace
