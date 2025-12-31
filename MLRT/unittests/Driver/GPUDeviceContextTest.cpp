//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace M::Driver {

// Test fixture is needed to enable testing of private methods in DeviceContext.
class GPUDeviceContextTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    // Print information about the device to make it easier to debug the
    // environment.
    ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);
    if (device.isError())
      // Have the tests deal with device errors.
      return;

    std::cerr << "##### GPUDeviceContextTest running on "
              << Device::numberOfDevices(Device::gpuAPI) << " x "
              << (*device)->name() << "\n";
  }

  void SetUp() override {
    // No common setup needed.
  }
};

TEST_F(GPUDeviceContextTest, Smoke) {
  ErrorOr<DeviceRef> device = Device::create(Device::gpuAPI);

  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  std::string deviceName = ctx->device().name();
  if (ctx->device().api() == Device::cudaAPI) {
    EXPECT_NE(deviceName.find("NVIDIA"), std::string::npos) << deviceName;
    std::cerr << "Name of default CUDA device: " << deviceName << "\n";
  } else if (ctx->device().api() == Device::metalAPI) {
    EXPECT_NE(deviceName.find("Apple"), std::string::npos) << deviceName;
    std::cerr << "Name of default Metal device: " << deviceName << "\n";
  } else {
    EXPECT_NE(deviceName.find("AMD"), std::string::npos) << deviceName;
    std::cerr << "Name of default HIP device: " << deviceName << "\n";
  }
}

TEST_F(GPUDeviceContextTest, NumDevices) {
  int num = Device::numberOfDevices(Device::gpuAPI);
  EXPECT_GT(num, 0);
  EXPECT_LT(num, 16);
  std::cerr << "Number of GPU devices: " << num << "\n";

  // Create devices outside of allowed range.
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, -7), "must be positive");
  EXPECT_ERR_MSG(Device::create(Device::gpuAPI, num), "not available");
}

TEST_F(GPUDeviceContextTest, ComputeCapability) {
  ASSIGN_NO_ERR(DeviceRef, dev, Device::create(Device::gpuAPI));

  ASSIGN_NO_ERR(int, cap, dev->computeCapability());
  if (dev->api() == Device::metalAPI) {
    // Metal returns 1,2,3,4 for compute capability (chip #)
    EXPECT_GE(cap, 0);
  } else {
    EXPECT_GE(cap, 75);
  }
  std::cerr << "Compute capability of GPU[0]: " << cap << "\n";
}

TEST_F(GPUDeviceContextTest, MAXCompatibility) {
  ASSIGN_NO_ERR(DeviceRef, dev, Device::create(Device::gpuAPI));
  EXPECT_NO_ERR(dev->isCompatible());
}

TEST_F(GPUDeviceContextTest, DefaultContext) {
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

TEST_F(GPUDeviceContextTest, DeviceContextScope) {

  // Create two different device contexts
  ASSIGN_NO_ERR(DeviceContextRef, ctx1,
                DeviceContext::create(Device::gpuAPI, 0));

  // Get the initial driver context
  const Device &dev = ctx1->device();

  // Skip test for Metal
  if (dev.api() == Device::metalAPI) {
    GTEST_SKIP() << "Metal not supported";
  }

  ASSIGN_NO_ERR(void *, initialContext, dev.getCurrentDriverContext());

  {
    // Create a scope for the first context
    DeviceContextScope scope(*ctx1);
    EXPECT_FALSE(scope.err().isError()) << scope.err().getError();

    // Verify the context is now active
    ASSIGN_NO_ERR(void *, currentContext, dev.getCurrentDriverContext());
    EXPECT_EQ(currentContext, ctx1->driverContext());

    // Create a buffer to verify the context is working
    ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx1->createBuffer(256));
    EXPECT_NE(buf->ptr(), nullptr);

    // Create a nested scope with the same context (should be a no-op)
    {
      DeviceContextScope nestedScope(*ctx1);
      EXPECT_FALSE(nestedScope.err().isError());

      ASSIGN_NO_ERR(void *, nestedContext, dev.getCurrentDriverContext());
      EXPECT_EQ(nestedContext, currentContext);
    }

    // Verify the context is still active after nested scope ends
    ASSIGN_NO_ERR(void *, afterNestedContext, dev.getCurrentDriverContext());
    EXPECT_EQ(afterNestedContext, currentContext);
  }

  // After the scope ends, verify the context is restored to initial state
  ASSIGN_NO_ERR(void *, restoredContext, dev.getCurrentDriverContext());
  EXPECT_EQ(restoredContext, initialContext);

  // Test with multiple devices if available
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  if (numDevices > 1) {
    std::cerr << "Testing DeviceContextScope with multiple devices\n";

    ASSIGN_NO_ERR(DeviceContextRef, ctx2,
                  DeviceContext::create(Device::gpuAPI, 1));

    // Switch to the second context
    {
      DeviceContextScope scope2(*ctx2);
      EXPECT_FALSE(scope2.err().isError()) << scope2.err().getError();

      // The current context should be from the second device
      ASSIGN_NO_ERR(void *, ctx2Current, dev.getCurrentDriverContext());
      EXPECT_EQ(ctx2Current, ctx2->driverContext());

      // Create a buffer to verify the second context works
      ASSIGN_NO_ERR(DeviceBufferRef, buf2, ctx2->createBuffer(256));
      EXPECT_NE(buf2->ptr(), nullptr);
    }

    // Verify it switches back to initial context
    ASSIGN_NO_ERR(void *, afterMultiContext, dev.getCurrentDriverContext());
    EXPECT_EQ(afterMultiContext, initialContext);
  }

  // Test error handling by creating an invalid device context scope
  // This part is tricky because we don't have a reliable way to create an
  // invalid context in a unit test, but we can check that the error propagation
  // works
  {
    DeviceContextScope invalidScope(*ctx1); // This should succeed
    EXPECT_FALSE(invalidScope.err().isError());

    // If we wanted to test failure, we'd need a way to create an invalid
    // context ErrorOrSuccess testErr = invalidScope.err();
    // EXPECT_TRUE(testErr.isError()); // This would check error propagation
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

TEST_F(GPUDeviceContextTest, DeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx,
                DeviceContext::create(Device::gpuAPI, 0));

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
  const int numCopyLengths = 3;

  for (int i = 0; i < numCopyLengths; ++i) {
    int allocLen = copyLengths[i];

    ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf, ctx->enqueueCreateBuffer(0));
    EXPECT_EQ(emptyBuf->ptr(), nullptr);
    EXPECT_EQ(emptyBuf->byteSize(), 0);

    std::cerr << "DeviceContext copies(len: " << allocLen << ")\n";

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->createBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aBuf->ptr(), allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; ++i) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
    EXPECT_NO_ERR(bBuf->copyTo(*cBuf));
    EXPECT_NO_ERR(cBuf->copyTo(dPtr));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(ctx->synchronize());
    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
  }
}

TEST_F(GPUDeviceContextTest, FailDeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;

  std::cerr << "DeviceContext failing copies(len: " << allocLen << ")\n";

  char *aPtr = static_cast<char *>(malloc(allocLen));
  EXPECT_NE(aPtr, nullptr);

  // Create a source buffer that is too small.
  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen - 32));
  ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

  // Initialize with known good and bad values.
  for (int i = 0; i < allocLen; ++i) {
    aPtr[i] = static_cast<char>(i & 127);
  }

  // Copy between different buffers.
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));
  EXPECT_ERR_MSG(ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf),
                 "not enough data in src");

  // Ensure that all copy operations have finished.
  EXPECT_NO_ERR(ctx->synchronize());

  // Create an *owned* DeviceBuffer which will be invalid when destructed.
  {
    DeviceBufferRef bad = ctx->createBufferOwning(aPtr, allocLen,
                                                  /* ownning = */ true);
  }
  // Trigger deferred error.
  EXPECT_ERR_MSG(ctx->synchronize(), "failed");

  free(aPtr);
}

TEST_F(GPUDeviceContextTest, SubBuffer) {
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
  for (int i = 0; i < allocLen; ++i) {
    aPtr[i] = static_cast<char>(i & 127);
    dPtr[i] = -static_cast<char>((7 + i) & 127);
  }

  for (int i = 0; i < subAllocLen; ++i) {
    ePtr[i] = static_cast<float>(i) * (((i & 1) == 0) ? -1.0 : 1.0);
  }

  // Copy base pattern into base buffer.
  EXPECT_NO_ERR(bBuf->copyFrom(aPtr));
  // Copy overlay pattern into sub buffer.
  EXPECT_NO_ERR(cBuf->copyFrom(ePtr));
  // Copy base buffer back to host memory.
  EXPECT_NO_ERR(bBuf->copyTo(dPtr));

  EXPECT_NO_ERR(ctx->synchronize());
  uint32_t *dIntPtr = reinterpret_cast<uint32_t *>(dPtr);
  uint32_t expectedVals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (subAllocLen); ++i) {
    uint32_t val = dIntPtr[i];
    std::cerr << "SubBuffer: d[" << i << "] = 0x" << std::hex << val << std::dec
              << "\n";
    EXPECT_EQ(val, expectedVals[i]);
  }
  for (int i = subAllocLen * 4; i < allocLen; ++i) {
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
TEST_F(GPUDeviceContextTest, OverlappingCopies) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen =
      127 * 1024 * 1024; // 127MB to fit into host memory manager chunks.
  ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, refBuf, ctx->createHostBuffer(allocLen));

  void *aHostPtr = nullptr;
  {
    ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));
    aHostPtr = hostBuf->ptr();
    memset(aHostPtr, 12, allocLen);
    EXPECT_NO_ERR(hostBuf->copyTo(*aBuf));
    // hostBuf is released here, but should not be reallocated from the host
    // memory manager until the copy above has finished.
  }

  // Allocate another host buffer of the same size and overwrite the last byte.
  // If the above hostBuf was released too early this write will race with the
  // copy.
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));
  char *bHostPtr = reinterpret_cast<char *>(hostBuf->ptr());
  bHostPtr[allocLen - 1] = 15;
  EXPECT_NO_ERR(hostBuf->copyTo(*bBuf));

  // Verify that the original hostBuf was not overwritten.
  EXPECT_NO_ERR(ctx->synchronize());
  EXPECT_NO_ERR(aBuf->copyTo(*refBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  char *refPtr = reinterpret_cast<char *>(refBuf->ptr());
  for (int i = 0; i < allocLen; ++i) {
    EXPECT_EQ(refPtr[i], 12) << "failure at index: " << i << "\n";
  }
}

TEST_F(GPUDeviceContextTest, RecycleBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, allocCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, freeCtx,
                DeviceContext::create(Device::gpuAPI));

  constexpr size_t kb = 1024;
  constexpr size_t mb = 1024 * kb;
  constexpr size_t sizes[8] = {3 * kb,  2 * kb,   1 * mb,  511 * kb,
                               32 * kb, 128 * mb, 32 * mb, 78 * mb};
  // When run with MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=true this will
  // move all free buffers from allocCtx to freeCtx unless we regularly flush
  // buffers from freeCtx to the parent.
  for (int i = 0; i < 1000; ++i) {
    size_t allocSize = sizes[i & 7];
    ASSIGN_NO_ERR(DeviceBufferRef, buf,
                  allocCtx->enqueueCreateBuffer(allocSize));
    EXPECT_NO_ERR(buf->reassignOwnershipTo(*freeCtx));
    // buf goes out of scope here and is freed by freeCtx.
  }
}

TEST_F(GPUDeviceContextTest, SetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  int allocLen = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 123, 1));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint8_t *ptr = static_cast<uint8_t *>(hostBuf->ptr());
    for (int i = 0; i < allocLen; ++i) {
      EXPECT_EQ(ptr[i], 123);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 12345, 2));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint16_t *ptr = static_cast<uint16_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 2); ++i) {
      EXPECT_EQ(ptr[i], 12345);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890, 4));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint32_t *ptr = static_cast<uint32_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 4); ++i) {
      EXPECT_EQ(ptr[i], 1234567890);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 0x0101010101010101ul, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); ++i) {
      EXPECT_EQ(ptr[i], 0x0101010101010101ul);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890123456789, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); ++i) {
      EXPECT_EQ(ptr[i], 1234567890123456789);
    }
  }
}

TEST_F(GPUDeviceContextTest, Timers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::gpuAPI));

  ASSIGN_NO_ERR(DeviceTimerRef, timer, ctx->startTimer());
  { // Run some memory copies within the timer.
    const size_t allocLen = 1024 * 1024;

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(256));

    for (int i = 0; i < 20; ++i) {
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

TEST_F(GPUDeviceContextTest, MultiContextCopies) {
  int numDevices = Device::numberOfDevices(Device::gpuAPI);
  std::cerr << "Number of GPU devices: " << numDevices << "\n";

  // Allocate a vector filled with all available devices prefixed with CPU and
  // GPU[0].
  std::vector<DeviceContextRef> contexts;
  for (int i = 0; i < numDevices + 2; ++i) {
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

  for (int srcDeviceId = 0; srcDeviceId < numDevices + 2; ++srcDeviceId) {
    const DeviceContext *srcCtx = contexts[srcDeviceId].getPointer();
    for (int dstDeviceId = 0; dstDeviceId < numDevices + 2; ++dstDeviceId) {
      const DeviceContext *dstCtx = contexts[dstDeviceId].getPointer();

      std::cerr << "Copying from contexts[" << srcDeviceId << "] to contexts["
                << dstDeviceId << "].\n";

      const int allocLen = 256;
      char *checkData = new char[allocLen];
      // Initialize checkData with a pattern specific for this run.
      for (int i = 0; i < allocLen; ++i) {
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
      for (int i = 0; i < allocLen; ++i) {
        // Use a different factor for i than the checkData, so that we can't
        // ever overlap.
        dstPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
        dstHostPtr[i] = -static_cast<char>((7 + 2 * i) & 127);
      }

      // Copy between different buffers.
      memcpy(srcHostBuf->ptr(), checkData, allocLen);
      EXPECT_NO_ERR(srcHostBuf->copyTo(*srcDevBuf));
      EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
      EXPECT_NO_ERR(dstDevBuf->copyTo(dstPtr));
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

TEST_F(GPUDeviceContextTest, CrossContextEvent) {
  ASSIGN_NO_ERR(DeviceContextRef, srcCtx,
                DeviceContext::create(Device::gpuAPI));
  ASSIGN_NO_ERR(DeviceContextRef, dstCtx,
                DeviceContext::create(Device::gpuAPI));

  const int allocLen = 12 * 1024 * 1024;
  char *checkData = new char[allocLen];
  // Initialize checkData with a pattern specific for this run.
  for (int i = 0; i < allocLen; ++i) {
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
  EXPECT_NO_ERR(srcDevBuf->copyTo(*dstDevBuf));
  EXPECT_NO_ERR(dstDevBuf->copyTo(*dstHostBuf));

  // An event based sequence equivalent to: err = dstCtx->synchronize();
  ASSIGN_NO_ERR(DeviceEventRef, syncEvent, dstCtx->enqueueEvent());
  EXPECT_NO_ERR(syncEvent->synchronize());

  EXPECT_EQ(memcmp(dstHostBuf->ptr(), checkData, allocLen), 0);
  delete[] checkData;
}

} // namespace M::Driver
