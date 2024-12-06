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
  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::gpuAPI);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef ctx = std::move(errCtx.get());
  ASSERT_TRUE(ctx);

  std::string deviceName = ctx->device().name();
  if (ctx->device().api() == Device::cudaAPI) {
    EXPECT_NE(deviceName.find("NVIDIA"), std::string::npos) << deviceName;
    std::cerr << "Name of default CUDA device: " << deviceName << "\n";
  } else {
    EXPECT_NE(deviceName.find("AMD"), std::string::npos) << deviceName;
    std::cerr << "Name of default HIP device: " << deviceName << "\n";
  }
}

TEST(GPUDeviceContextTest, NumDevices) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 0);
  EXPECT_LT(num, 16);
  std::cerr << "Number of GPU devices: " << num << "\n";
}

static void initWithData(void *dst, size_t dstLen, const void *checkData,
                         size_t checkDataLen) {
  char *cur = static_cast<char *>(dst);
  size_t remaining = dstLen;
  while (remaining > 0) {
    size_t toCopy = std::min(remaining, checkDataLen);
    memcpy(cur, checkData, toCopy);
    remaining -= toCopy;
    cur += toCopy;
  }
}

static void compareWithData(const void *dst, size_t dstLen,
                            const void *checkData, size_t checkDataLen) {
  const char *cur = static_cast<const char *>(dst);
  size_t remaining = dstLen;
  while (remaining > 0) {
    size_t toCheck = std::min(remaining, checkDataLen);
    EXPECT_EQ(memcmp(cur, checkData, toCheck), 0);
    remaining -= toCheck;
    cur += toCheck;
  }
}

TEST(GPUDeviceContextTest, DeviceBuffers) {
  ErrorOr<DeviceContextRef> errCtx =
      DeviceContext::create(Device::gpuAPI, 0, 100ull * 1024 * 1024);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef ctx = std::move(errCtx.get());
  ASSERT_TRUE(ctx);

  char checkData[] = {
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

  const int copyLengths[] = {256, 1024 * 1024};

  for (int i = 0; i < 4; i++) {
    bool doSyncCopies = i & 1;
    int allocLen = copyLengths[(i >> 1) & 1];

    ErrorOr<DeviceBufferRef> emptyBufErr = ctx->createBuffer(
        0, doSyncCopies ? DeviceContext::SYNC : DeviceContext::ASYNC);
    EXPECT_FALSE(emptyBufErr);
    EXPECT_EQ((*emptyBufErr)->ptr(), nullptr);
    EXPECT_EQ((*emptyBufErr)->byteSize(), 0);

    std::cerr << "DeviceContext copies(len: " << allocLen
              << ", sync: " << doSyncCopies << ")\n";

    ErrorOr<void *> errPtr = ctx->malloc_host(allocLen);
    EXPECT_FALSE(errPtr) << errPtr.getError() << "\n";
    void *aPtr = errPtr.get();
    EXPECT_NE(aPtr, nullptr);

    ErrorOr<DeviceBufferRef> errRef = ctx->createBuffer(allocLen);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef bBuf = std::move(errRef.get());
    EXPECT_TRUE(bBuf);

    errRef = ctx->enqueueCreateBuffer(allocLen);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef cBuf = std::move(errRef.get());
    EXPECT_TRUE(cBuf);

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aPtr, allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; i++) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    ErrorOrSuccess err;
    if (doSyncCopies) {
      // Ensure that all setup has finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";

      // Copy between different buffers.
      err = ctx->copyToDevice(*bBuf, aPtr);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->copyDeviceToDevice(*cBuf, *bBuf);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->copyFromDevice(dPtr, *cBuf);
      EXPECT_FALSE(err) << err.getError() << "\n";
    } else {
      // Copy between different buffers.
      err = ctx->enqueueCopyToDevice(*bBuf, aPtr);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf);
      EXPECT_FALSE(err) << err.getError() << "\n";
      err = ctx->enqueueCopyFromDevice(dPtr, *cBuf);
      EXPECT_FALSE(err) << err.getError() << "\n";

      // Ensure that all copy operations have finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";
    }
    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
    err = ctx->free_host(aPtr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }
}

TEST(DeviceContextTest, FailDeviceBuffers) {
  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::cpuAPI);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef ctx = std::move(errCtx.get());
  ASSERT_TRUE(ctx);

  const int copyLengths[] = {256};

  for (int i = 0; i < 2; i++) {
    DeviceContext::SyncMode syncMode =
        ((i & 1) == 0) ? DeviceContext::SYNC : DeviceContext::ASYNC;
    int allocLen = copyLengths[(i >> 1) & 1];

    std::cerr << "DeviceContext failing copies(len: " << allocLen
              << ", syncMode: "
              << ((syncMode == DeviceContext::SYNC) ? "SYNC" : "ASYNC")
              << ")\n";

    char *aPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(aPtr, nullptr);

    // Create a source buffer that is too small.
    ErrorOr<DeviceBufferRef> errRef =
        ctx->createBuffer(allocLen - 32, syncMode);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef bBuf = std::move(errRef.get());
    EXPECT_TRUE(bBuf);

    errRef = ctx->createBuffer(allocLen, syncMode);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef cBuf = std::move(errRef.get());
    EXPECT_TRUE(cBuf);

    // Initialize with known good and bad values.
    for (int i = 0; i < allocLen; i++) {
      aPtr[i] = static_cast<char>(i & 127);
    }

    ErrorOrSuccess err;
    // Copy between different buffers.
    err = ctx->copyToDevice(*bBuf, aPtr, syncMode);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = ctx->copyDeviceToDevice(*cBuf, *bBuf, syncMode);
    EXPECT_TRUE(err);
    EXPECT_THAT(err.getError(), testing::HasSubstr("not enough data in src"));

    if (syncMode == DeviceContext::ASYNC) {
      // Ensure that all copy operations have finished.
      err = ctx->synchronize();
      EXPECT_FALSE(err) << err.getError() << "\n";
    }

    free(aPtr);
  }
}

TEST(GPUDeviceContextTest, SubBuffer) {
  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::gpuAPI);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef ctx = std::move(errCtx.get());
  ASSERT_TRUE(ctx);

  int allocLen = 256;

  int subOffset = 0;
  int subAllocLen = 3;

  std::cerr << "DeviceContext sub buffer copies(len: " << allocLen << ")\n";

  char *aPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(aPtr, nullptr);

  ErrorOr<DeviceBufferRef> errRef = ctx->enqueueCreateBuffer(allocLen);
  EXPECT_FALSE(errRef) << errRef.getError() << "\n";
  DeviceBufferRef bBuf = std::move(errRef.get());
  EXPECT_TRUE(bBuf);

  errRef = bBuf->createSubBuffer(subOffset, subAllocLen * 4);
  EXPECT_FALSE(errRef) << errRef.getError() << "\n";
  DeviceBufferRef cBuf = std::move(errRef.get());
  EXPECT_TRUE(cBuf);

  char *dPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(dPtr, nullptr);

  float *ePtr = static_cast<float *>(malloc(subAllocLen * sizeof(float)));
  EXPECT_NE(ePtr, nullptr);

  // Initialize with known good and bad values.
  for (int i = 0; i < allocLen; i++) {
    aPtr[i] = static_cast<char>(i & 127);
    dPtr[i] = -static_cast<char>((7 + i) & 127);
  }

  for (int i = 0; i < subAllocLen; i++) {
    ePtr[i] = static_cast<float>(i) * (((i & 1) == 0) ? -1.0 : 1.0);
  }

  // Copy base pattern into base buffer.
  ErrorOrSuccess err = ctx->enqueueCopyToDevice(*bBuf, aPtr);
  EXPECT_FALSE(err) << err.getError() << "\n";

  // Copy overlay pattern into sub buffer.
  err = ctx->enqueueCopyToDevice(*cBuf, ePtr);
  EXPECT_FALSE(err) << err.getError() << "\n";

  // Copy base buffer back to host memory.
  err = ctx->enqueueCopyFromDevice(dPtr, *bBuf);
  EXPECT_FALSE(err) << err.getError() << "\n";

  uint32_t *dIntPtr = reinterpret_cast<uint32_t *>(dPtr);
  uint32_t expectedVals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (subAllocLen); i++) {
    uint32_t val = dIntPtr[i];
    std::cerr << "SubBuffer: d[" << i << "] = 0x" << std::hex << val << std::dec
              << "\n";
    EXPECT_EQ(val, expectedVals[i]);
  }
  for (int i = subAllocLen * 4; i < allocLen; i++) {
    int val = dPtr[i];
    if (i < 20) {
      std::cerr << "Original: d[" << std::dec << i << "] = 0x" << std::hex
                << val << std::dec << "\n";
    }
    EXPECT_EQ(val, i & 127);
  }
}

TEST(GPUDeviceContextTest, Timers) {
  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::gpuAPI);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef ctx = std::move(errCtx.get());
  ASSERT_TRUE(ctx);

  ErrorOr<DeviceTimerRef> errTimer = ctx->startTimer();
  EXPECT_FALSE(errTimer) << errTimer.getError() << "\n";
  DeviceTimerRef timer = std::move(errTimer.get());
  ASSERT_TRUE(timer);

  { // Run some memory copies within the timer.
    const size_t allocLen = 1024 * 1024;

    ErrorOr<void *> errPtr = ctx->malloc_host(allocLen);
    EXPECT_FALSE(errPtr) << errPtr.getError() << "\n";
    void *aPtr = errPtr.get();
    EXPECT_NE(aPtr, nullptr);

    ErrorOr<DeviceBufferRef> errRef = ctx->enqueueCreateBuffer(256);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef bBuf = std::move(errRef.get());
    EXPECT_TRUE(bBuf);

    ErrorOrSuccess err;
    for (int i = 0; i < 20; i++) {
      err = ctx->enqueueCopyToDevice(*bBuf, aPtr);
      EXPECT_FALSE(err) << err.getError() << "\n";
    }
    err = ctx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    err = ctx->free_host(aPtr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }

  ErrorOr<std::chrono::nanoseconds> errNanos = ctx->stopTimer(*timer);
  EXPECT_FALSE(errNanos) << errNanos.getError() << "\n";
  int64_t elapsedNanos = errNanos.get().count();

  std::cerr << "Timer measured: " << elapsedNanos << "ns\n";

  EXPECT_GT(elapsedNanos, 0);
  EXPECT_LT(elapsedNanos, 60 * 1000ll * 1000ll * 1000ll); // less than a minute
}

TEST(GPUDeviceContextTest, MultiContextCopies) {
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  std::cerr << "Number of GPU devices: " << numDevices << "\n";

  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::gpuAPI, 0);
  ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
  DeviceContextRef srcCtx = std::move(errCtx.get());
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
          DeviceContext::create(Device::gpuAPI, curDeviceId);
      ASSERT_FALSE(errCtx) << errCtx.getError() << "\n";
      dstCtx = std::move(errCtx.get());
    }
    ASSERT_TRUE(dstCtx);

    const int allocLen = 256;
    char *checkData = new char[allocLen];
    // Initialize checkData with a pattern specific for this run.
    for (int i = 0; i < allocLen; i++) {
      checkData[i] = curDeviceId + i;
    }

    ErrorOr<void *> errPtr = srcCtx->malloc_host(allocLen);
    EXPECT_FALSE(errPtr) << errPtr.getError() << "\n";
    void *srcPtr = errPtr.get();
    EXPECT_NE(srcPtr, nullptr);

    ErrorOr<DeviceBufferRef> errRef = srcCtx->enqueueCreateBuffer(allocLen);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef srcBuf = std::move(errRef.get());
    EXPECT_TRUE(srcBuf);

    errRef = dstCtx->enqueueCreateBuffer(allocLen);
    EXPECT_FALSE(errRef) << errRef.getError() << "\n";
    DeviceBufferRef dstBuf = std::move(errRef.get());
    EXPECT_TRUE(dstBuf);

    char *dstPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dstPtr, nullptr);
    // Initialize with known bad values.
    for (int i = 0; i < allocLen; i++) {
      // Use a different factor for i than the checkData, so that we can't ever
      // overlap.
      dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
    }

    // Copy between different buffers.
    memcpy(srcPtr, checkData, allocLen);
    ErrorOrSuccess err = srcCtx->enqueueCopyToDevice(*srcBuf, srcPtr);
    EXPECT_FALSE(err) << err.getError() << "\n";
    // Make sure to use the dstCtx here to properly enqueue future work on
    // dstCtx within the stream.
    err = dstCtx->enqueueCopyDeviceToDevice(*dstBuf, *srcBuf);
    EXPECT_FALSE(err) << err.getError() << "\n";
    err = dstCtx->enqueueCopyFromDevice(dstPtr, *dstBuf);
    EXPECT_FALSE(err) << err.getError() << "\n";

    // Ensure that all copy operations have finished.
    err = dstCtx->synchronize();
    err = srcCtx->synchronize();
    EXPECT_FALSE(err) << err.getError() << "\n";

    EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);

    free(dstPtr);
    err = srcCtx->free_host(srcPtr);
    EXPECT_FALSE(err) << err.getError() << "\n";
  }
}

} // namespace
