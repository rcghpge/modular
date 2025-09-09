//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "AsyncRT/lib/DeviceContext/CPU/CpuDeviceContext.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace M::AsyncRT;
using M::ErrorOr;

extern char **environ;

namespace {

// RAII wrapper for environment variable management.
class ScopedEnvVar {
public:
  ScopedEnvVar(llvm::StringRef name, llvm::StringRef value) : varName(name) {
    originalValue = nullptr;

    // Attempt to save the existing variable if it exists.
    for (char **it = environ; *it != nullptr; it++) {
      auto str = llvm::StringRef(*it);
      auto [prefix, _] = str.split('=');

      if (prefix == name) {
        originalValue = *it;
        break;
      }
    }

    // Set the new value
    newVar = std::string(name) + "=" + std::string(value);
    putenv(newVar.data());
  }

  ~ScopedEnvVar() {
    // Restore the original value or unset if it didn't exist
    if (originalValue) {
      putenv(originalValue);
    } else {
      unsetenv(varName.c_str());
    }
  }

  // Disable copy and move.
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;
  ScopedEnvVar(ScopedEnvVar &&) = delete;
  ScopedEnvVar &operator=(ScopedEnvVar &&) = delete;

private:
  std::string varName;
  std::string newVar;
  char *originalValue;
};

TEST(DeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  EXPECT_EQ(ctx->device().name(), Device::cpuAPI);
}

TEST(DeviceContextTest, CreationFailure) {
  EXPECT_ERR_MSG(DeviceContext::create(Device::cpuAPI, 1),
                 "Device(api: cpu, id: 1) is not available");
  EXPECT_ERR_MSG(DeviceContext::create("pcu"),
                 "Unsupported device api: \"pcu\"");
}

TEST(DeviceContextTest, NumDevices) {
  EXPECT_EQ(Device::numberOfDevices(Device::cpuAPI), 1);
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

TEST(DeviceContextTest, DeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

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
  const int numCopyLengths = 2;

  ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf, ctx->enqueueCreateBuffer(0));
  EXPECT_EQ(emptyBuf->ptr(), nullptr);
  EXPECT_EQ(emptyBuf->byteSize(), 0);

  for (int i = 0; i < numCopyLengths; i++) {
    int allocLen = copyLengths[i];

    std::cerr << "DeviceContext copies(len: " << allocLen << ")\n";

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aBuf->ptr(), allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; i++) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(aBuf->copyTo(*bBuf));
    EXPECT_NO_ERR(ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf));
    EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(dPtr, *cBuf));

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(ctx->synchronize());

    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
  }
}

TEST(DeviceContextTest, FailDeviceBuffers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  const int copyLengths[] = {256, 1024 * 1024};
  const int numCopyLengths = 2;

  for (int i = 0; i < numCopyLengths; i++) {
    int allocLen = copyLengths[i];

    std::cerr << "DeviceContext failing copies(len: " << allocLen << ")\n";

    char *aPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(aPtr, nullptr);

    // Create a source buffer that is too small.
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf,
                  ctx->enqueueCreateBuffer(allocLen - 32));
    ASSIGN_NO_ERR(DeviceBufferRef, cBuf, ctx->enqueueCreateBuffer(allocLen));

    // Initialize with known good and bad values.
    for (int i = 0; i < allocLen; i++) {
      aPtr[i] = static_cast<char>(i & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));
    EXPECT_ERR_MSG(ctx->enqueueCopyDeviceToDevice(*cBuf, *bBuf),
                   "not enough data in src");

    // Ensure that all copy operations have finished.
    EXPECT_NO_ERR(ctx->synchronize());

    free(aPtr);
  }
}

TEST(DeviceContextTest, SubBuffer) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

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
  EXPECT_NO_ERR(bBuf->copyFrom(aPtr));

  // Copy overlay pattern into sub buffer.
  EXPECT_NO_ERR(cBuf->copyFrom(ePtr));

  // Copy base buffer back to host memory.
  EXPECT_NO_ERR(bBuf->copyTo(dPtr));

  uint32_t *dIntPtr = reinterpret_cast<uint32_t *>(dPtr);
  uint32_t expectedVals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (subAllocLen); i++) {
    uint32_t val = dIntPtr[i];
    std::cerr << "SubBuffer: d[" << std::dec << i << "] = 0x" << std::hex << val
              << "\n";
    EXPECT_EQ(val, expectedVals[i]);
  }
  for (int i = subAllocLen * 4; i < allocLen; i++) {
    int val = dPtr[i];
    if (i < 20) {
      std::cerr << "Original: d[" << std::dec << i << "] = 0x" << std::hex
                << val << "\n";
    }
    EXPECT_EQ(val, i & 127);
  }

  free(ePtr);
  free(dPtr);
  free(aPtr);
}

TEST(DeviceContextTest, SetMemory) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  int allocLen = 256;
  ASSIGN_NO_ERR(DeviceBufferRef, buf, ctx->enqueueCreateBuffer(allocLen));
  ASSIGN_NO_ERR(DeviceBufferRef, hostBuf, ctx->createHostBuffer(allocLen));

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 123, 1));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint8_t *ptr = static_cast<uint8_t *>(hostBuf->ptr());
    for (int i = 0; i < allocLen; i++) {
      EXPECT_EQ(ptr[i], 123);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 12345, 2));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint16_t *ptr = static_cast<uint16_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 2); i++) {
      EXPECT_EQ(ptr[i], 12345);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890, 4));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint32_t *ptr = static_cast<uint32_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 4); i++) {
      EXPECT_EQ(ptr[i], 1234567890);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 0x0101010101010101ul, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); i++) {
      EXPECT_EQ(ptr[i], 0x0101010101010101ul);
    }
  }

  EXPECT_NO_ERR(ctx->enqueueSetMemory(*buf, 1234567890123456789, 8));
  EXPECT_NO_ERR(buf->copyTo(*hostBuf));
  EXPECT_NO_ERR(ctx->synchronize());
  {
    uint64_t *ptr = static_cast<uint64_t *>(hostBuf->ptr());
    for (int i = 0; i < (allocLen / 8); i++) {
      EXPECT_EQ(ptr[i], 1234567890123456789);
    }
  }
}

TEST(DeviceContextTest, DeviceExec) {
  const char *vecAddPtx = R"PTX(
// From open-source/max/max/kernels/test/gpu/vec_add.ptx

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

  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  std::cerr << "Compiling vecAddPtx...\n";
  DeviceFunction::SignatureBuilder sig = DeviceFunction::SignatureBuilder()
                                             .addBuffer()
                                             .addBuffer()
                                             .addBuffer()
                                             .addUInt64();
  // Expecting for an error to be set in the case of CPU device context.
  EXPECT_ERR(ctx->loadFunction("vec_add_module", "vec_add", sig, vecAddPtx));
}

TEST(DeviceContextTest, Timers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

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

TEST(DeviceContextTest, SyncMode) {
  {
    ScopedEnvVar env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "0");
    ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));
    EXPECT_FALSE(ctx->syncMode());
  }

  {
    ScopedEnvVar env("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "1");
    ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));
    EXPECT_TRUE(ctx->syncMode());
  }
}

TEST(DeviceContextTest, BufferCache) {
  constexpr size_t chunkSize = 1024;
  constexpr size_t maxCacheSize = 4 * chunkSize;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;  // Needs to be larger than 256 * KB.
  memInfo.total = 256 * maxCacheSize; // Needs to be larger than 256 * KB.

  ScopedEnvVar env[] = {
      // Ensure that maxCacheSize is all of memInfo.free.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE", "25"),
      // Ensure that chunkSize is a 1/4th of maxCacheSize.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_CHUNK_PERCENT", "25"),
      // Ensure that selfCheck() is enabled.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SELF_CHECK", "True"),
      // Ensure that logging is enabled.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_LOG", "True")};

  // Create a CPU device that is associated with a BufferCache by passing in a
  // non-empty memInfo.
  CpuDeviceRef device = CpuDeviceRef::create(memInfo);
  size_t allocAlignment = device->allocationAlignment();

  BufferCacheFreeList &bufCache = *device->bufferCache().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  ASSIGN_NO_ERR(void *, base, bufCache.allocate(1, mode));
  char *cacheBase = static_cast<char *>(base);
  EXPECT_NO_ERR(bufCache.free(cacheBase));

  // Allocation patterns to hit different kinds of allocation blocks and
  // merges. These have been verified to have full coverage by instrumenting
  // various branches.
  EXPECT_NO_ERR(bufCache.free(nullptr));
  ASSIGN_NO_ERR(void *, a, bufCache.allocate(15, mode));
  EXPECT_EQ(a, cacheBase);
  ASSIGN_NO_ERR(void *, b, bufCache.allocate(15, mode));
  EXPECT_EQ(b, cacheBase + allocAlignment);
  EXPECT_NO_ERR(bufCache.free(a));
  ASSIGN_NO_ERR(void *, c, bufCache.allocate(allocAlignment, mode));
  EXPECT_EQ(c, cacheBase);
  EXPECT_NO_ERR(bufCache.free(c));
  ASSIGN_NO_ERR(void *, d, bufCache.allocate(allocAlignment + 1, mode));
  EXPECT_EQ(d, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(void *, e, bufCache.allocate(1, mode));
  EXPECT_EQ(e, cacheBase);
  EXPECT_NO_ERR(bufCache.free(b));
  ASSIGN_NO_ERR(void *, f, bufCache.allocate(allocAlignment, mode));
  EXPECT_EQ(f, cacheBase + allocAlignment);
  EXPECT_NO_ERR(bufCache.free(f));
  EXPECT_NO_ERR(bufCache.free(e));
  ASSIGN_NO_ERR(void *, g, bufCache.allocate(2 * allocAlignment, mode));
  EXPECT_EQ(g, cacheBase);
  ASSIGN_NO_ERR(void *, h, bufCache.allocate(1, mode));
  EXPECT_EQ(h, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(void *, i, bufCache.allocate(150, mode));
  EXPECT_EQ(i, cacheBase + 5 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(g));
  EXPECT_NO_ERR(bufCache.free(h));
  ASSIGN_NO_ERR(void *, j, bufCache.allocate(2, mode));
  EXPECT_EQ(j, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(j));
  EXPECT_NO_ERR(bufCache.free(d));
  ASSIGN_NO_ERR(void *, k, bufCache.allocate(150, mode));
  EXPECT_EQ(k, cacheBase);
  EXPECT_NO_ERR(bufCache.free(i));
  EXPECT_NO_ERR(bufCache.free(k));
  /* ----- */
  ASSIGN_NO_ERR(void *, l, bufCache.allocate(chunkSize, mode));
  EXPECT_EQ(l, cacheBase);
  EXPECT_NO_ERR(bufCache.free(l));
  ASSIGN_NO_ERR(void *, m, bufCache.allocate(1, mode));
  EXPECT_EQ(m, cacheBase);
  ASSIGN_NO_ERR(void *, n, bufCache.allocate(1, mode));
  EXPECT_EQ(n, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(void *, o, bufCache.allocate(1, mode));
  EXPECT_EQ(o, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(void *, p, bufCache.allocate(1, mode));
  EXPECT_EQ(p, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(void *, q, bufCache.allocate(1, mode));
  EXPECT_EQ(q, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(o));
  EXPECT_NO_ERR(bufCache.free(p));
  EXPECT_NO_ERR(bufCache.free(n));
  EXPECT_NO_ERR(bufCache.free(m));
  EXPECT_NO_ERR(bufCache.free(q));
  /* ----- */
  ASSIGN_NO_ERR(void *, r, bufCache.allocate(allocAlignment, mode));
  EXPECT_EQ(r, cacheBase);
  ASSIGN_NO_ERR(void *, s, bufCache.allocate(allocAlignment, mode));
  EXPECT_EQ(s, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(void *, t, bufCache.allocate(2 * allocAlignment, mode));
  EXPECT_EQ(t, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(void *, u, bufCache.allocate(allocAlignment, mode));
  EXPECT_EQ(u, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(r));
  EXPECT_NO_ERR(bufCache.free(t));
  ASSIGN_NO_ERR(void *, v, bufCache.allocate(2 * allocAlignment, mode));
  EXPECT_NO_ERR(bufCache.free(v));
  EXPECT_NO_ERR(bufCache.free(u));
  EXPECT_NO_ERR(bufCache.free(s));
  /* ----- */
  ASSIGN_NO_ERR(void *, x, bufCache.allocate(1, mode));
  EXPECT_EQ(x, cacheBase);
  ASSIGN_NO_ERR(void *, y, bufCache.allocate(1, mode));
  EXPECT_EQ(y, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(void *, z, bufCache.allocate(1, mode));
  EXPECT_EQ(z, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(void *, aa, bufCache.allocate(1, mode));
  EXPECT_EQ(aa, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(void *, ab, bufCache.allocate(1, mode));
  EXPECT_EQ(ab, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(void *, ac, bufCache.allocate(1, mode));
  EXPECT_EQ(ac, cacheBase + 5 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(ac));
  EXPECT_NO_ERR(bufCache.free(aa));
  EXPECT_NO_ERR(bufCache.free(y));
  EXPECT_NO_ERR(bufCache.free(z));
  EXPECT_NO_ERR(bufCache.free(x));
  EXPECT_NO_ERR(bufCache.free(ab));
  /* ----- */
  ASSIGN_NO_ERR(void *, ad, bufCache.allocate(1, mode));
  EXPECT_EQ(ad, cacheBase);
  ASSIGN_NO_ERR(void *, ae, bufCache.allocate(1, mode));
  EXPECT_EQ(ae, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(void *, af, bufCache.allocate(1, mode));
  EXPECT_EQ(af, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(void *, ag, bufCache.allocate(1, mode));
  EXPECT_EQ(ag, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(void *, ah, bufCache.allocate(1, mode));
  EXPECT_EQ(ah, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(void *, ai, bufCache.allocate(1, mode));
  EXPECT_EQ(ai, cacheBase + 5 * allocAlignment);
  ASSIGN_NO_ERR(void *, aj, bufCache.allocate(1, mode));
  EXPECT_EQ(aj, cacheBase + 6 * allocAlignment);
  ASSIGN_NO_ERR(void *, ak, bufCache.allocate(1, mode));
  EXPECT_EQ(ak, cacheBase + 7 * allocAlignment);
  EXPECT_NO_ERR(bufCache.free(ak));
  EXPECT_NO_ERR(bufCache.free(ag));
  EXPECT_NO_ERR(bufCache.free(ai));
  EXPECT_NO_ERR(bufCache.free(ae));
  EXPECT_NO_ERR(bufCache.free(af));
  EXPECT_NO_ERR(bufCache.free(aj));
  EXPECT_NO_ERR(bufCache.free(ah));
  EXPECT_NO_ERR(bufCache.free(ad));

  ASSIGN_NO_ERR(void *, al, bufCache.allocate(chunkSize - 64, mode));
  EXPECT_EQ(al, cacheBase);
  ASSIGN_NO_ERR(void *, am, bufCache.allocate(chunkSize - 64, mode));
  EXPECT_NE(am, nullptr);
  ASSIGN_NO_ERR(void *, an, bufCache.allocate(chunkSize - 64, mode));
  EXPECT_NE(an, nullptr);
  ASSIGN_NO_ERR(void *, ao, bufCache.allocate(chunkSize - 64, mode));
  EXPECT_NE(ao, nullptr);
  ASSIGN_NO_ERR(void *, ap, bufCache.allocate(chunkSize - 64, mode));
  EXPECT_NE(ap, nullptr);
  EXPECT_NO_ERR(bufCache.free(al));
  EXPECT_NO_ERR(bufCache.free(am));
  EXPECT_NO_ERR(bufCache.free(an));
  EXPECT_NO_ERR(bufCache.free(ao));
  EXPECT_FALSE(bufCache.free(ap));

  size_t minBlock = allocAlignment;
  ASSIGN_NO_ERR(void *, ba, bufCache.allocate(minBlock, mode));
  EXPECT_EQ(ba, cacheBase);
  ASSIGN_NO_ERR(void *, bb, bufCache.allocate(4 * minBlock, mode));
  EXPECT_NE(bb, nullptr);
  ASSIGN_NO_ERR(void *, bc, bufCache.allocate(minBlock, mode));
  EXPECT_NE(bc, nullptr);
  ASSIGN_NO_ERR(void *, bd, bufCache.allocate(4 * minBlock, mode));
  EXPECT_NE(bd, nullptr);
  ASSIGN_NO_ERR(void *, be, bufCache.allocate(minBlock, mode));
  EXPECT_NE(be, nullptr);

  EXPECT_NO_ERR(bufCache.free(bd));
  EXPECT_NO_ERR(bufCache.free(bb));

  ASSIGN_NO_ERR(void *, bf, bufCache.allocate(2 * minBlock, mode));
  EXPECT_NE(bf, nullptr);
  ASSIGN_NO_ERR(void *, bg, bufCache.allocate(3 * minBlock, mode));
  EXPECT_NE(bg, nullptr);
  ASSIGN_NO_ERR(void *, bh, bufCache.allocate(4 * minBlock, mode));
  EXPECT_NE(bh, nullptr);

  EXPECT_NO_ERR(bufCache.free(ba));
  EXPECT_NO_ERR(bufCache.free(bc));
  EXPECT_NO_ERR(bufCache.free(be));
  EXPECT_NO_ERR(bufCache.free(bf));
  EXPECT_NO_ERR(bufCache.free(bg));
  EXPECT_NO_ERR(bufCache.free(bh));
}

TEST(DeviceContextTest, GetMemoryInfoWithoutBufferCache) {
  // Test that getMemoryInfo returns device memory when buffer cache is disabled
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  ASSIGN_NO_ERR(Device::MemInfo, memInfo, ctx->getMemoryInfo());
  ASSIGN_NO_ERR(Device::MemInfo, deviceMemInfo, ctx->device().getMemoryInfo());

  // Without buffer cache, context memory info should match device memory info
  EXPECT_EQ(memInfo.total, deviceMemInfo.total);
  EXPECT_EQ(memInfo.free, deviceMemInfo.free);
}

TEST(DeviceContextTest, GetMemoryInfoWithBufferCache) {
  // Test that getMemoryInfo returns buffer cache memory when buffer cache is
  // enabled
  constexpr size_t maxCacheSize = 4 * 1024;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;
  memInfo.total = 256 * maxCacheSize;

  ScopedEnvVar env[] = {
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE", "25"),
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_CHUNK_PERCENT", "25")};

  // Create a CPU device with buffer cache enabled
  CpuDeviceRef device = CpuDeviceRef::create(memInfo);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  ASSIGN_NO_ERR(Device::MemInfo, contextMemInfo, ctx->getMemoryInfo());
  ASSIGN_NO_ERR(Device::MemInfo, deviceMemInfo, ctx->device().getMemoryInfo());

  // With buffer cache enabled, context memory info should differ from device
  // memory info
  EXPECT_NE(contextMemInfo.total, deviceMemInfo.total);
  EXPECT_NE(contextMemInfo.free, deviceMemInfo.free);

  // Context should report buffer cache limits, not device limits
  size_t expectedCacheSize = (memInfo.total * 25) / 100;
  EXPECT_EQ(contextMemInfo.total, expectedCacheSize);
  EXPECT_EQ(contextMemInfo.free, expectedCacheSize);
}

TEST(DeviceContextTest, GetMemoryInfoDynamicUpdate) {
  // Test that getMemoryInfo reflects allocations/frees in buffer cache
  constexpr size_t maxCacheSize = 8 * 1024;
  constexpr size_t allocSize = 1024;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;
  memInfo.total = 256 * maxCacheSize;

  ScopedEnvVar env[] = {
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_SIZE", "25"),
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_BUFFER_CACHE_CHUNK_PERCENT", "25")};

  // Create a CPU device with buffer cache enabled
  CpuDeviceRef device = CpuDeviceRef::create(memInfo);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Get initial memory info
  ASSIGN_NO_ERR(Device::MemInfo, initialMemInfo, ctx->getMemoryInfo());

  // Allocate a buffer
  ASSIGN_NO_ERR(DeviceBufferRef, buffer, ctx->enqueueCreateBuffer(allocSize));

  // Memory info should show less free memory
  ASSIGN_NO_ERR(Device::MemInfo, afterAllocMemInfo, ctx->getMemoryInfo());
  EXPECT_LT(afterAllocMemInfo.free, initialMemInfo.free);
  EXPECT_EQ(afterAllocMemInfo.total, initialMemInfo.total);

  // Free the buffer (destructor releases it)
  buffer.reset();
  EXPECT_NO_ERR(ctx->synchronize());

  // Memory info should show more free memory again
  ASSIGN_NO_ERR(Device::MemInfo, afterFreeMemInfo, ctx->getMemoryInfo());
  EXPECT_GE(afterFreeMemInfo.free, afterAllocMemInfo.free);
  EXPECT_EQ(afterFreeMemInfo.total, initialMemInfo.total);
}

} // namespace
