//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"

#include "gtest/gtest.h"

#include <cstring>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

TEST(CUDADeviceContextTest, Smoke) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  EXPECT_FALSE(err_ctx) << err_ctx.getError() << "\n";
  DeviceContextRef ctx = std::move(err_ctx.get());
  ASSERT_TRUE(ctx);

  EXPECT_EQ(ctx->device().name(), "cuda");
}

TEST(CUDADeviceContextTest, DeviceBuffers) {
  ErrorOr<DeviceContextRef> err_ctx = DeviceContext::create("cuda");
  EXPECT_FALSE(err_ctx) << err_ctx.getError() << "\n";
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

  const size_t alloc_len = 256;

  ErrorOr<void *> err_ptr = ctx->malloc_host(alloc_len);
  EXPECT_FALSE(err_ptr) << err_ptr.getError() << "\n";
  void *a_ptr = err_ptr.get();
  EXPECT_NE(a_ptr, nullptr);

  ErrorOr<DeviceBufferRef> err_ref = ctx->createBuffer(256, 1);
  EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
  DeviceBufferRef b_buf = std::move(err_ref.get());
  EXPECT_TRUE(b_buf);

  err_ref = ctx->createBuffer(16, 16);
  EXPECT_FALSE(err_ref) << err_ref.getError() << "\n";
  DeviceBufferRef c_buf = std::move(err_ref.get());
  EXPECT_TRUE(c_buf);

  void *d_ptr = malloc(256);
  EXPECT_NE(d_ptr, nullptr);

  // Copy between different buffers.
  memcpy(a_ptr, check_data, 256);
  ErrorOrSuccess err = ctx->enqueueCopyToDevice(*b_buf, a_ptr);
  EXPECT_FALSE(err) << err.getError() << "\n";
  err = ctx->enqueueCopyDeviceToDevice(*c_buf, *b_buf);
  EXPECT_FALSE(err) << err.getError() << "\n";
  err = ctx->enqueueCopyFromDevice(d_ptr, *c_buf);
  EXPECT_FALSE(err) << err.getError() << "\n";

  // Ensure that all copy operations have finished.
  err = ctx->synchronize();
  EXPECT_FALSE(err) << err.getError() << "\n";

  EXPECT_EQ(memcmp(d_ptr, check_data, 256), 0);

  free(d_ptr);
  err = ctx->free_host(a_ptr);
  EXPECT_FALSE(err) << err.getError() << "\n";
}

} // namespace
