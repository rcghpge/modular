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

namespace {

TEST(GPUDeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

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

TEST(GPUDeviceContextTest, DefaultContext) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx0, DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::getOrCreate(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                DeviceContext::getOrCreate(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, ctx3, DeviceContext::create(Device::gpuAPI));
  EXPECT_NE(ctx0.getPointer(), ctx1.getPointer());
  EXPECT_NE(ctx0.getPointer(), ctx2.getPointer());
  EXPECT_NE(ctx0.getPointer(), ctx3.getPointer());
  EXPECT_EQ(ctx1.getPointer(), ctx2.getPointer());
  EXPECT_NE(ctx1.getPointer(), ctx3.getPointer());

  int num = Device::numberOfDevices(Device::gpuAPI);
  if (num > 1) {
    ASSIGN_NO_ERR(DeviceContextRef, ctx4,
                  DeviceContext::getOrCreate(Device::gpuAPI, 1));
    ASSIGN_NO_ERR(DeviceContextRef, ctx5,
                  DeviceContext::getOrCreate(Device::gpuAPI, 1));
    EXPECT_EQ(ctx4.getPointer(), ctx5.getPointer());
    EXPECT_NE(ctx1.getPointer(), ctx4.getPointer());
    EXPECT_NE(ctx0.getPointer(), ctx4.getPointer());
    EXPECT_NE(ctx3.getPointer(), ctx4.getPointer());
  }
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
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, 0, 100ull * 1024 * 1024));

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

  const int copyLengths[] = {0, 256, 1024 * 1024};

  for (int i = 0; i < 4; i++) {
    bool doSyncCopies = i & 1;
    int allocLen = copyLengths[(i >> 1) & 1];

    ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf,
                  (doSyncCopies) ? ctx->createBuffer(0)
                                 : ctx->enqueueCreateBuffer(0));
    EXPECT_EQ(emptyBuf->ptr(), nullptr);
    EXPECT_EQ(emptyBuf->byteSize(), 0);

    std::cerr << "DeviceContext copies(len: " << allocLen
              << ", sync: " << doSyncCopies << ")\n";

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->createBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aBuf->ptr(), allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; i++) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    if (doSyncCopies) {
      // Ensure that all setup has finished.
      EXPECT_NO_ERR(ctx->synchronize());

      // Copy between different buffers.
      EXPECT_NO_ERR(ctx->copyToDevice(*bBuf, aBuf->ptr()));
      EXPECT_NO_ERR(ctx->copyDeviceToDevice(*cBuf, *bBuf));
      EXPECT_NO_ERR(ctx->copyFromDevice(dPtr, *cBuf));
    } else {
      // Copy between different buffers.
      EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
      EXPECT_NO_ERR(ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf));
      EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(dPtr, *cBuf));

      // Ensure that all copy operations have finished.
      EXPECT_NO_ERR(ctx->synchronize());
    }
    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
  }
}

TEST(DeviceContextTest, FailDeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

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
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf,
                  (syncMode == DeviceContext::SYNC)
                      ? ctx->createBuffer(allocLen - 32)
                      : ctx->enqueueCreateBuffer(allocLen - 32));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf,
                  (syncMode == DeviceContext::SYNC)
                      ? ctx->createBuffer(allocLen)
                      : ctx->enqueueCreateBuffer(allocLen));

    // Initialize with known good and bad values.
    for (int i = 0; i < allocLen; i++) {
      aPtr[i] = static_cast<char>(i & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(ctx->copyToDevice(*bBuf, aPtr, syncMode));
    EXPECT_ERR_MSG(ctx->copyDeviceToDevice(*cBuf, *bBuf, syncMode),
                   "not enough data in src");

    if (syncMode == DeviceContext::ASYNC) {
      // Ensure that all copy operations have finished.
      EXPECT_NO_ERR(ctx->synchronize());
    }

    free(aPtr);
  }
}

TEST(GPUDeviceContextTest, SubBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;

  int subOffset = 0;
  int subAllocLen = 3;

  std::cerr << "DeviceContext sub buffer copies(len: " << allocLen << ")\n";

  char *aPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(aPtr, nullptr);

  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, cBuf,
                bBuf->createSubBuffer(subOffset, subAllocLen * 4));

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
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));
  // Copy overlay pattern into sub buffer.
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*cBuf, ePtr));
  // Copy base buffer back to host memory.
  EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(dPtr, *bBuf));

  EXPECT_NO_ERR(ctx->synchronize());
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
  free(ePtr);
  free(dPtr);
  free(aPtr);
}

TEST(DeviceContextTest, SetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));

  EXPECT_NO_ERR(ctx->setMemory(*buf, 123, 1, DeviceContext::ASYNC));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint8_t *ptr = static_cast<uint8_t *>(hostBuf->ptr());
    for (int i = 0; i < allocLen; i++) {
      EXPECT_EQ(ptr[i], 123);
    }
  }

  EXPECT_NO_ERR(ctx->setMemory(*buf, 12345, 2, DeviceContext::ASYNC));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint16_t *ptr = static_cast<uint16_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 2); i++) {
      EXPECT_EQ(ptr[i], 12345);
    }
  }

  EXPECT_NO_ERR(ctx->setMemory(*buf, 1234567890, 4, DeviceContext::ASYNC));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint32_t *ptr = static_cast<uint32_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 4); i++) {
      EXPECT_EQ(ptr[i], 1234567890);
    }
  }

  EXPECT_NO_ERR(
      ctx->setMemory(*buf, 0x0101010101010101ul, 8, DeviceContext::ASYNC));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); i++) {
      EXPECT_EQ(ptr[i], 0x0101010101010101ul);
    }
  }

  EXPECT_NO_ERR(
      ctx->setMemory(*buf, 1234567890123456789, 8, DeviceContext::ASYNC));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); i++) {
      EXPECT_EQ(ptr[i], 1234567890123456789);
    }
  }
}

TEST(GPUDeviceContextTest, Timers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx->startTimer());
  { // Run some memory copies within the timer.
    const size_t allocLen = 1024 * 1024;

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(256));

    for (int i = 0; i < 20; i++) {
      EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
    }
    EXPECT_NO_ERR(ctx->synchronize());
  }
  ASSIGN_NO_ERR(std::chrono::nanoseconds, nanos, ctx->stopTimer(*timer));
  int64_t elapsedNanos = nanos.count();

  std::cerr << "Timer measured: " << elapsedNanos << "ns\n";

  EXPECT_GT(elapsedNanos, 0);
  EXPECT_LT(elapsedNanos, 60 * 1000ll * 1000ll * 1000ll); // less than a minute
}

TEST(GPUDeviceContextTest, MultiContextCopies) {
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  std::cerr << "Number of GPU devices: " << numDevices << "\n";

  // Allocate a vector filled with all available devices prefixed with CPU and
  // GPU[0].
  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numDevices + 2; i++) {
    if (i == 0) {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::cpuAPI, 0));
      contexts.emplace_back(std::move(ctx));
    } else if (i == 1) {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::gpuAPI, 0));
      contexts.emplace_back(std::move(ctx));
    } else {
      ASSIGN_NO_ERR(DeviceContextRef, ctx,
                    DeviceContext::create(Device::gpuAPI, i - 2));
      contexts.emplace_back(std::move(ctx));
    }
  }

  for (int srcDeviceId = 0; srcDeviceId < numDevices + 2; srcDeviceId++) {
    const DeviceContext *srcCtx = contexts[srcDeviceId].getPointer();
    for (int dstDeviceId = 0; dstDeviceId < numDevices + 2; dstDeviceId++) {
      const DeviceContext *dstCtx = contexts[dstDeviceId].getPointer();

      std::cerr << "Copying from contexts[" << srcDeviceId << "] to contexts["
                << dstDeviceId << "].\n";

      const int allocLen = 256;
      char *checkData = new char[allocLen];
      // Initialize checkData with a pattern specific for this run.
      for (int i = 0; i < allocLen; i++) {
        checkData[i] = (srcDeviceId * 8) + dstDeviceId + i;
      }

      ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf,
                    srcCtx->createHostBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf,
                    srcCtx->enqueueCreateBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                    dstCtx->enqueueCreateBuffer(allocLen));
      ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf,
                    dstCtx->createHostBuffer(allocLen));

      char *dstHostPtr = static_cast<char *>(dstHostBuf->ptr());
      char *dstPtr = static_cast<char *>(malloc(allocLen));
      EXPECT_NE(dstPtr, nullptr);
      // Initialize with known bad values.
      for (int i = 0; i < allocLen; i++) {
        // Use a different factor for i than the checkData, so that we can't
        // ever overlap.
        dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
        dstHostPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
      }

      // Copy between different buffers.
      memcpy(srcHostBuf->ptr(), checkData, allocLen);
      EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));
      EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
      EXPECT_NO_ERR(dstCtx->enqueueCopyFromDevice(dstPtr, *dstDevBuf));
      EXPECT_NO_ERR(srcHostBuf->copyTo(*dstHostBuf));

      // Ensure that all copy operations have finished.
      EXPECT_NO_ERR(dstCtx->synchronize());
      EXPECT_NO_ERR(srcCtx->synchronize());

      EXPECT_EQ(memcmp(dstPtr, checkData, allocLen), 0);
      EXPECT_EQ(memcmp(dstHostPtr, checkData, allocLen), 0);

      free(dstPtr);
      delete[] checkData;
    }
  }
}

TEST(GPUDeviceContextTest, CrossContextEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, dstCtx,
                DeviceContext::create(Device::gpuAPI));

  const int allocLen = 12 * 1024 * 1024;
  char *checkData = new char[allocLen];
  // Initialize checkData with a pattern specific for this run.
  for (int i = 0; i < allocLen; i++) {
    checkData[i] = i;
  }

  ASSIGN_NO_ERR(DeviceBufferRef, srcHostBuf,
                srcCtx->createHostBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstHostBuf,
                dstCtx->createHostBuffer(allocLen));

  memcpy(srcHostBuf->ptr(), checkData, allocLen);

  ASSIGN_NO_ERR(DeviceBufferRef, srcDevBuf,
                srcCtx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, dstDevBuf,
                dstCtx->enqueueCreateBuffer(allocLen));

  EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));

  ASSIGN_NO_ERR(DeviceEventRef, copyDoneEvent, srcCtx->enqueueEvent());

  EXPECT_NO_ERR(dstCtx->enqueueWaitFor(*copyDoneEvent));
  EXPECT_NO_ERR(dstCtx->enqueueCopyDeviceToDevice(*dstDevBuf, *srcDevBuf));
  EXPECT_NO_ERR(dstDevBuf->copyTo(*dstHostBuf));

  // An event based sequence equivalent to: err = dstCtx->synchronize();
  ASSIGN_NO_ERR(DeviceEventRef, syncEvent, dstCtx->enqueueEvent());
  EXPECT_NO_ERR(syncEvent->synchronize());

  EXPECT_EQ(memcmp(dstHostBuf->ptr(), checkData, allocLen), 0);
  delete[] checkData;
}

} // namespace
