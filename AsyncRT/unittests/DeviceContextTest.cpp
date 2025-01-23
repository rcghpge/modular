//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/DeviceContext/DeviceContext.h"
#include "AsyncRT/lib/DeviceContext/BufferCache.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstring>
#include <iostream>

using namespace M::AsyncRT;
using M::ErrorOr;
using M::ErrorOrSuccess;

namespace {

TEST(DeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  EXPECT_EQ(ctx->device().name(), Device::cpuAPI);
}

TEST(DeviceContextTest, CreationFailure) {
  ErrorOr<DeviceContextRef> errCtx = DeviceContext::create(Device::cpuAPI, 1);
  ASSERT_TRUE(errCtx.isError());
  EXPECT_EQ(strcmp(errCtx.getError(), "CPU device only supports id == 0."), 0)
      << errCtx.getError();

  errCtx = DeviceContext::create("pcu");
  ASSERT_TRUE(errCtx.isError());
  EXPECT_EQ(strcmp(errCtx.getError(), "Unsupported device api: \"pcu\""), 0)
      << errCtx.getError();
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

  for (int i = 0; i < 4; i++) {
    DeviceContext::SyncMode syncMode =
        ((i & 1) == 0) ? DeviceContext::SYNC : DeviceContext::ASYNC;
    int allocLen = copyLengths[(i >> 1) & 1];

    ASSIGN_NO_ERR(DeviceBufferRef, emptyBuf,
                  (syncMode == DeviceContext::SYNC)
                      ? ctx->createBuffer(0)
                      : ctx->enqueueCreateBuffer(0));
    EXPECT_EQ(emptyBuf->ptr(), nullptr);
    EXPECT_EQ(emptyBuf->byteSize(), 0);

    std::cerr << "DeviceContext copies(len: " << allocLen << ", syncMode: "
              << ((syncMode == DeviceContext::SYNC) ? "SYNC" : "ASYNC")
              << ")\n";

    ASSIGN_NO_ERR(void *, aPtr, ctx->mallocHost(allocLen));
    EXPECT_NE(aPtr, nullptr);

    ASSIGN_NO_ERR(DeviceBufferRef, bBuf,
                  (syncMode == DeviceContext::SYNC)
                      ? ctx->createBuffer(allocLen)
                      : ctx->enqueueCreateBuffer(allocLen));

    ASSIGN_NO_ERR(DeviceBufferRef, cBuf,
                  (syncMode == DeviceContext::SYNC)
                      ? ctx->createBuffer(allocLen)
                      : ctx->enqueueCreateBuffer(allocLen));

    char *dPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(dPtr, nullptr);

    // Initialize with known good and bad values.
    initWithData(aPtr, allocLen, checkData, sizeof(checkData));

    for (int i = 0; i < allocLen; i++) {
      dPtr[i] = -static_cast<char>((7 + i) & 127);
    }

    // Copy between different buffers.
    EXPECT_NO_ERR(ctx->copyToDevice(*bBuf, aPtr, syncMode));
    EXPECT_NO_ERR(ctx->copyDeviceToDevice(*cBuf, *bBuf, syncMode));
    EXPECT_NO_ERR(ctx->copyFromDevice(dPtr, *cBuf, syncMode));

    if (syncMode == DeviceContext::ASYNC) {
      // Ensure that all copy operations have finished.
      EXPECT_NO_ERR(ctx->synchronize());
    }

    compareWithData(dPtr, allocLen, checkData, sizeof(checkData));

    free(dPtr);
    EXPECT_NO_ERR(ctx->freeHost(aPtr));
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

    ErrorOrSuccess err;
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

TEST(CUDADeviceContextTest, SubBuffer) {
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
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));

  // Copy overlay pattern into sub buffer.
  EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*cBuf, ePtr));

  // Copy base buffer back to host memory.
  EXPECT_NO_ERR(ctx->enqueueCopyFromDevice(dPtr, *bBuf));

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

TEST(DeviceContextTest, DeviceExec) {
  const char *vecAddPtx = R"PTX(
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

    ASSIGN_NO_ERR(void *, aPtr, ctx->mallocHost(allocLen));
    EXPECT_NE(aPtr, nullptr);

    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(256));

    for (int i = 0; i < 20; i++) {
      EXPECT_NO_ERR(ctx->enqueueCopyToDevice(*bBuf, aPtr));
    }
    EXPECT_NO_ERR(ctx->synchronize());

    EXPECT_NO_ERR(ctx->freeHost(aPtr));
  }

  ASSIGN_NO_ERR(std::chrono::nanoseconds, nanos, ctx->stopTimer(*timer));
  int64_t elapsedNanos = nanos.count();

  std::cerr << "Timer measured: " << elapsedNanos << "ns\n";

  EXPECT_GT(elapsedNanos, 0);
  EXPECT_LT(elapsedNanos, 60 * 1000ll * 1000ll * 1000ll); // less than a minute
}

TEST(DeviceContextTest, BufferCache) {
  char *cacheBase = reinterpret_cast<char *>(1024);
  size_t cacheSize = 4096;
  size_t allocAlignment = 32;

  BufferCache bufCache(cacheBase, cacheSize, allocAlignment);

  // Allocation patterns to hit different kinds of allocation blocks and merges.
  // These have been verified to have full coverage by instrumenting various
  // branches.
  EXPECT_EQ(nullptr, bufCache.allocate(cacheSize + 1));
  EXPECT_FALSE(bufCache.free(nullptr));
  EXPECT_FALSE(bufCache.free(cacheBase + 1));
  void *a = bufCache.allocate(15);
  EXPECT_EQ(a, cacheBase);
  void *b = bufCache.allocate(15);
  EXPECT_EQ(b, cacheBase + allocAlignment);
  EXPECT_TRUE(bufCache.free(a));
  void *c = bufCache.allocate(32);
  EXPECT_EQ(c, cacheBase);
  EXPECT_TRUE(bufCache.free(c));
  void *d = bufCache.allocate(33);
  EXPECT_EQ(d, cacheBase + 2 * allocAlignment);
  void *e = bufCache.allocate(1);
  EXPECT_EQ(e, cacheBase);
  EXPECT_TRUE(bufCache.free(b));
  void *f = bufCache.allocate(32);
  EXPECT_EQ(f, cacheBase + allocAlignment);
  EXPECT_TRUE(bufCache.free(f));
  EXPECT_TRUE(bufCache.free(e));
  void *g = bufCache.allocate(64);
  EXPECT_EQ(g, cacheBase);
  void *h = bufCache.allocate(1);
  EXPECT_EQ(h, cacheBase + 4 * allocAlignment);
  void *i = bufCache.allocate(150);
  EXPECT_EQ(i, cacheBase + 5 * allocAlignment);
  EXPECT_TRUE(bufCache.free(g));
  EXPECT_TRUE(bufCache.free(h));
  void *j = bufCache.allocate(2);
  EXPECT_EQ(j, cacheBase + 4 * allocAlignment);
  EXPECT_TRUE(bufCache.free(j));
  EXPECT_TRUE(bufCache.free(d));
  void *k = bufCache.allocate(150);
  EXPECT_EQ(k, cacheBase);
  EXPECT_TRUE(bufCache.free(i));
  EXPECT_TRUE(bufCache.free(k));
  /* ----- */
  void *l = bufCache.allocate(cacheSize);
  EXPECT_EQ(l, cacheBase);
  EXPECT_TRUE(bufCache.free(l));
  void *m = bufCache.allocate(1);
  EXPECT_EQ(m, cacheBase);
  void *n = bufCache.allocate(1);
  EXPECT_EQ(n, cacheBase + allocAlignment);
  void *o = bufCache.allocate(1);
  EXPECT_EQ(o, cacheBase + 2 * allocAlignment);
  void *p = bufCache.allocate(1);
  EXPECT_EQ(p, cacheBase + 3 * allocAlignment);
  void *q = bufCache.allocate(1);
  EXPECT_EQ(q, cacheBase + 4 * allocAlignment);
  EXPECT_TRUE(bufCache.free(o));
  EXPECT_TRUE(bufCache.free(p));
  EXPECT_TRUE(bufCache.free(n));
  EXPECT_TRUE(bufCache.free(m));
  EXPECT_TRUE(bufCache.free(q));
  /* ----- */
  void *r = bufCache.allocate(allocAlignment);
  EXPECT_EQ(r, cacheBase);
  void *s = bufCache.allocate(allocAlignment);
  EXPECT_EQ(s, cacheBase + allocAlignment);
  void *t = bufCache.allocate(2 * allocAlignment);
  EXPECT_EQ(t, cacheBase + 2 * allocAlignment);
  void *u = bufCache.allocate(allocAlignment);
  EXPECT_EQ(u, cacheBase + 4 * allocAlignment);
  EXPECT_TRUE(bufCache.free(r));
  EXPECT_TRUE(bufCache.free(t));
  void *v = bufCache.allocate(2 * allocAlignment);
  EXPECT_TRUE(bufCache.free(v));
  EXPECT_TRUE(bufCache.free(u));
  EXPECT_TRUE(bufCache.free(s));
  /* ----- */
  void *x = bufCache.allocate(1);
  EXPECT_EQ(x, cacheBase);
  void *y = bufCache.allocate(1);
  EXPECT_EQ(y, cacheBase + allocAlignment);
  void *z = bufCache.allocate(1);
  EXPECT_EQ(z, cacheBase + 2 * allocAlignment);
  void *aa = bufCache.allocate(1);
  EXPECT_EQ(aa, cacheBase + 3 * allocAlignment);
  void *ab = bufCache.allocate(1);
  EXPECT_EQ(ab, cacheBase + 4 * allocAlignment);
  void *ac = bufCache.allocate(1);
  EXPECT_EQ(ac, cacheBase + 5 * allocAlignment);
  EXPECT_TRUE(bufCache.free(ac));
  EXPECT_TRUE(bufCache.free(aa));
  EXPECT_TRUE(bufCache.free(y));
  EXPECT_TRUE(bufCache.free(z));
  EXPECT_TRUE(bufCache.free(x));
  EXPECT_TRUE(bufCache.free(ab));
  /* ----- */
  void *ad = bufCache.allocate(1);
  EXPECT_EQ(ad, cacheBase);
  void *ae = bufCache.allocate(1);
  EXPECT_EQ(ae, cacheBase + allocAlignment);
  void *af = bufCache.allocate(1);
  EXPECT_EQ(af, cacheBase + 2 * allocAlignment);
  void *ag = bufCache.allocate(1);
  EXPECT_EQ(ag, cacheBase + 3 * allocAlignment);
  void *ah = bufCache.allocate(1);
  EXPECT_EQ(ah, cacheBase + 4 * allocAlignment);
  void *ai = bufCache.allocate(1);
  EXPECT_EQ(ai, cacheBase + 5 * allocAlignment);
  void *aj = bufCache.allocate(1);
  EXPECT_EQ(aj, cacheBase + 6 * allocAlignment);
  void *ak = bufCache.allocate(1);
  EXPECT_EQ(ak, cacheBase + 7 * allocAlignment);
  EXPECT_TRUE(bufCache.free(ak));
  EXPECT_TRUE(bufCache.free(ag));
  EXPECT_TRUE(bufCache.free(ai));
  EXPECT_TRUE(bufCache.free(ae));
  EXPECT_TRUE(bufCache.free(af));
  EXPECT_TRUE(bufCache.free(aj));
  EXPECT_TRUE(bufCache.free(ah));
  EXPECT_TRUE(bufCache.free(ad));

  bufCache.clear();
}

} // namespace
