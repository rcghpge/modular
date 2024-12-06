//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

TEST(GPUDeviceContextTest, Smoke) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("gpu");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  std::string device_name = ctx->device().name();
  if (ctx->device().api() == "cuda") {
    EXPECT_NE(device_name.find("NVIDIA"), std::string::npos) << device_name;
    std::cerr << "Name of default CUDA device: " << device_name << "\n";
  } else {
    EXPECT_NE(device_name.find("AMD"), std::string::npos) << device_name;
    std::cerr << "Name of default HIP device: " << device_name << "\n";
  }
}

TEST(GPUDeviceContextTest, NumDevices) {
  ErrorOr<int> err_num = Device::numberOfDevices("gpu");
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

TEST(GPUDeviceContextTest, DeviceBuffers) {
  ErrorOr<DeviceContextRef> err_ctx =
      DeviceContext::create("gpu", 0, 100ull * 1024 * 1024);
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

    ErrorOr<DeviceBufferRef> err_ref = ctx->createBuffer(alloc_len);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    err_ref = ctx->enqueueCreateBuffer(alloc_len);
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
        ctx->createBuffer(alloc_len - 32, sync_mode);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef b_buf = std::move(err_ref.get());
    EXPECT_TRUE(b_buf);

    err_ref = ctx->createBuffer(alloc_len, sync_mode);
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

TEST(GPUDeviceContextTest, SubBuffer) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("gpu");
  ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  int alloc_len = 256;

  int sub_offset = 0;
  int sub_alloc_len = 3;

  std::cerr << "DeviceContext sub buffer copies(len: " << alloc_len << ")\n";

  char *a_ptr = static_cast<char *>(malloc(alloc_len));
  EXPECT_NE(a_ptr, nullptr);

  ErrorOr<DeviceBufferRef> err_ref = ctx->enqueueCreateBuffer(alloc_len);
  EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
  DeviceBufferRef b_buf = std::move(err_ref.get());
  EXPECT_TRUE(b_buf);

  err_ref = b_buf->createSubBuffer(sub_offset, sub_alloc_len * 4);
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
    std::cerr << "SubBuffer: d[" << i << "] = 0x" << std::hex << val << std::dec
              << "\n";
    EXPECT_EQ(val, expected_vals[i]);
  }
  for (int i = sub_alloc_len * 4; i < alloc_len; i++) {
    int val = d_ptr[i];
    if (i < 20) {
      std::cerr << "Original: d[" << std::dec << i << "] = 0x" << std::hex
                << val << std::dec << "\n";
    }
    EXPECT_EQ(val, i & 127);
  }
}

TEST(GPUDeviceContextTest, Timers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("gpu");
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

    ErrorOr<DeviceBufferRef> err_ref = ctx->enqueueCreateBuffer(256);
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

TEST(GPUDeviceContextTest, MultiContextCopies) {
  ErrorOr<int> err_num = Device::numberOfDevices("gpu");
  ASSERT_FALSE(err_num) << err_num.getError() << "\n";
  int num_devices = err_num.get();
  std::cerr << "Number of CUDA devices: " << num_devices << "\n";

  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("gpu", 0);
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
          DeviceContext::create("gpu", cur_device_id);
      ASSERT_FALSE(err_ctx) << err_ctx.getError() << "\n";
      dst_ctx = std::move(err_ctx.get());
    }
    ASSERT_TRUE(dst_ctx);

    const int alloc_len = 256;
    char *check_data = new char[alloc_len];
    // Initialize check_data with a pattern specific for this run.
    for (int i = 0; i < alloc_len; i++) {
      check_data[i] = cur_device_id + i;
    }

    ErrorOr<void *> err_ptr = src_ctx->malloc_host(alloc_len);
    EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
    void *src_ptr = err_ptr.get();
    EXPECT_NE(src_ptr, nullptr);

    ErrorOr<DeviceBufferRef> err_ref = src_ctx->enqueueCreateBuffer(alloc_len);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef src_buf = std::move(err_ref.get());
    EXPECT_TRUE(src_buf);

    err_ref = dst_ctx->enqueueCreateBuffer(alloc_len);
    EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
    DeviceBufferRef dst_buf = std::move(err_ref.get());
    EXPECT_TRUE(dst_buf);

    char *dst_ptr = static_cast<char *>(malloc(alloc_len));
    EXPECT_NE(dst_ptr, nullptr);
    // Initialize with known bad values.
    for (int i = 0; i < alloc_len; i++) {
      // Use a different factor for i than the check_data, so that we can't ever
      // overlap.
      dst_ptr[i] = -static_cast<char>((7 + 2 * i) & 127);
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
