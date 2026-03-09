//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "MLRT/Driver/DeviceContext/DeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/CPU/CpuDeviceContext.h"
#include "MLRT/lib/Driver/DeviceContext/MemoryManager.h"
#include "ScopedEnvVar.h"
#include "TestUtils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace M::Driver;
using M::ErrorOr;

namespace {

TEST(DeviceContextTest, Smoke) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  EXPECT_EQ(ctx->device().name(), Device::cpuAPI);
}

TEST(DeviceContextTest, CPUDeviceContextCreateSucceeds) {
  // CPU devices should pass isCompatible() (returns success()).
  // This verifies the isCompatible() check in DeviceContext::create()
  // doesn't break non-GPU devices.
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));
  EXPECT_NE(ctx.getPointer(), nullptr);
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

  for (int i = 0; i < numCopyLengths; ++i) {
    int allocLen = copyLengths[i];

    std::cerr << "DeviceContext copies(len: " << allocLen << ")\n";

    ASSIGN_NO_ERR(DeviceBufferRef, aBuf, ctx->createHostBuffer(allocLen));
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf, ctx->enqueueCreateBuffer(allocLen));
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

  for (int i = 0; i < numCopyLengths; ++i) {
    int allocLen = copyLengths[i];

    std::cerr << "DeviceContext failing copies(len: " << allocLen << ")\n";

    char *aPtr = static_cast<char *>(malloc(allocLen));
    EXPECT_NE(aPtr, nullptr);

    // Create a source buffer that is too small.
    ASSIGN_NO_ERR(DeviceBufferRef, bBuf,
                  ctx->enqueueCreateBuffer(allocLen - 32));
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

  uint32_t *dIntPtr = reinterpret_cast<uint32_t *>(dPtr);
  uint32_t expectedVals[] = {0x80000000, 0x3f800000, 0xc0000000};
  for (int i = 0; i < (subAllocLen); ++i) {
    uint32_t val = dIntPtr[i];
    std::cerr << "SubBuffer: d[" << std::dec << i << "] = 0x" << std::hex << val
              << "\n";
    EXPECT_EQ(val, expectedVals[i]);
  }
  for (int i = subAllocLen * 4; i < allocLen; ++i) {
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

TEST(DeviceContextTest, DeviceExec) {
  const char *vecAddPtx = R"PTX(
// From max/kernels/test/gpu/vec_add.ptx

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
  // Expecting for an error to be set in the case of CPU device context.
  EXPECT_ERR(ctx->loadFunction("vec_add_module", "vec_add", vecAddPtx));
}

TEST(DeviceContextTest, Timers) {
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

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

TEST(DeviceContextTest, MemoryManager) {
  constexpr size_t chunkSize = 1024;
  constexpr size_t maxCacheSize = 4 * chunkSize;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;  // Needs to be larger than 256 * KB.
  memInfo.total = 256 * maxCacheSize; // Needs to be larger than 256 * KB.

  ScopedEnvVar env[] = {
      // Ensure that maxCacheSize is all of memInfo.free.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "25"),
      // Ensure that chunkSize is a 1/4th of maxCacheSize.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT", "25"),
      // Ensure that selfCheck() is enabled.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SELF_CHECK", "True"),
      // Ensure that logging is enabled.
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_LOG", "True")};

  // Create a CPU device that is associated with a MemoryManager by passing in a
  // non-empty memInfo.
  CpuDeviceRef device = CpuDeviceRef::create(memInfo);
  size_t allocAlignment = device->allocationAlignment();

  MemoryManagerFreeList &memMgrFreeList = *device->memoryManager().freeList();

  const Device::AllocMode mode = Device::AllocMode::SYNC;
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, base,
                memMgrFreeList.allocate(1, mode));
  char *cacheBase = static_cast<char *>(base.ptr);
  EXPECT_NO_ERR(memMgrFreeList.free(base.ptr, base.mode));

  // Allocation patterns to hit different kinds of allocation blocks and
  // merges. These have been verified to have full coverage by instrumenting
  // various branches.
  EXPECT_NO_ERR(memMgrFreeList.free(nullptr, mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, a,
                memMgrFreeList.allocate(15, mode));
  EXPECT_EQ(a.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, b,
                memMgrFreeList.allocate(15, mode));
  EXPECT_EQ(b.ptr, cacheBase + allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(a.ptr, a.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, c,
                memMgrFreeList.allocate(allocAlignment, mode));
  EXPECT_EQ(c.ptr, cacheBase);
  EXPECT_NO_ERR(memMgrFreeList.free(c.ptr, c.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, d,
                memMgrFreeList.allocate(allocAlignment + 1, mode));
  EXPECT_EQ(d.ptr, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, e,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(e.ptr, cacheBase);
  EXPECT_NO_ERR(memMgrFreeList.free(b.ptr, b.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, f,
                memMgrFreeList.allocate(allocAlignment, mode));
  EXPECT_EQ(f.ptr, cacheBase + allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(f.ptr, f.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(e.ptr, e.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, g,
                memMgrFreeList.allocate(2 * allocAlignment, mode));
  EXPECT_EQ(g.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, h,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(h.ptr, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, i,
                memMgrFreeList.allocate(150, mode));
  EXPECT_EQ(i.ptr, cacheBase + 5 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(g.ptr, g.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(h.ptr, h.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, j,
                memMgrFreeList.allocate(2, mode));
  EXPECT_EQ(j.ptr, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(j.ptr, j.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(d.ptr, d.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, k,
                memMgrFreeList.allocate(150, mode));
  EXPECT_EQ(k.ptr, cacheBase);
  EXPECT_NO_ERR(memMgrFreeList.free(i.ptr, i.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(k.ptr, k.mode));
  /* ----- */
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, l,
                memMgrFreeList.allocate(chunkSize, mode));
  EXPECT_EQ(l.ptr, cacheBase);
  EXPECT_NO_ERR(memMgrFreeList.free(l.ptr, l.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, m,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(m.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, n,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(n.ptr, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, o,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(o.ptr, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, p,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(p.ptr, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, q,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(q.ptr, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(o.ptr, o.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(p.ptr, p.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(n.ptr, n.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(m.ptr, m.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(q.ptr, q.mode));
  /* ----- */
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, r,
                memMgrFreeList.allocate(allocAlignment, mode));
  EXPECT_EQ(r.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, s,
                memMgrFreeList.allocate(allocAlignment, mode));
  EXPECT_EQ(s.ptr, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, t,
                memMgrFreeList.allocate(2 * allocAlignment, mode));
  EXPECT_EQ(t.ptr, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, u,
                memMgrFreeList.allocate(allocAlignment, mode));
  EXPECT_EQ(u.ptr, cacheBase + 4 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(r.ptr, r.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(t.ptr, t.mode));
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, v,
                memMgrFreeList.allocate(2 * allocAlignment, mode));
  EXPECT_NO_ERR(memMgrFreeList.free(v.ptr, v.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(u.ptr, u.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(s.ptr, s.mode));
  /* ----- */
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, x,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(x.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, y,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(y.ptr, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, z,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(z.ptr, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, aa,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(aa.ptr, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ab,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ab.ptr, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ac,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ac.ptr, cacheBase + 5 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(ac.ptr, ac.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(aa.ptr, aa.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(y.ptr, y.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(z.ptr, z.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(x.ptr, x.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ab.ptr, ab.mode));
  /* ----- */
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ad,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ad.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ae,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ae.ptr, cacheBase + allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, af,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(af.ptr, cacheBase + 2 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ag,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ag.ptr, cacheBase + 3 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ah,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ah.ptr, cacheBase + 4 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ai,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ai.ptr, cacheBase + 5 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, aj,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(aj.ptr, cacheBase + 6 * allocAlignment);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ak,
                memMgrFreeList.allocate(1, mode));
  EXPECT_EQ(ak.ptr, cacheBase + 7 * allocAlignment);
  EXPECT_NO_ERR(memMgrFreeList.free(ak.ptr, ak.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ag.ptr, ag.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ai.ptr, ai.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ae.ptr, ae.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(af.ptr, af.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(aj.ptr, aj.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ah.ptr, ah.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ad.ptr, ad.mode));

  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, al,
                memMgrFreeList.allocate(chunkSize - 64, mode));
  EXPECT_EQ(al.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, am,
                memMgrFreeList.allocate(chunkSize - 64, mode));
  EXPECT_NE(am.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, an,
                memMgrFreeList.allocate(chunkSize - 64, mode));
  EXPECT_NE(an.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ao,
                memMgrFreeList.allocate(chunkSize - 64, mode));
  EXPECT_NE(ao.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ap,
                memMgrFreeList.allocate(chunkSize - 64, mode));
  EXPECT_NE(ap.ptr, nullptr);
  EXPECT_NO_ERR(memMgrFreeList.free(al.ptr, al.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(am.ptr, am.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(an.ptr, an.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(ao.ptr, ao.mode));
  EXPECT_FALSE(memMgrFreeList.free(ap.ptr, ap.mode));

  size_t minBlock = allocAlignment;
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, ba,
                memMgrFreeList.allocate(minBlock, mode));
  EXPECT_EQ(ba.ptr, cacheBase);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bb,
                memMgrFreeList.allocate(4 * minBlock, mode));
  EXPECT_NE(bb.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bc,
                memMgrFreeList.allocate(minBlock, mode));
  EXPECT_NE(bc.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bd,
                memMgrFreeList.allocate(4 * minBlock, mode));
  EXPECT_NE(bd.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, be,
                memMgrFreeList.allocate(minBlock, mode));
  EXPECT_NE(be.ptr, nullptr);

  EXPECT_NO_ERR(memMgrFreeList.free(bd.ptr, bd.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(bb.ptr, bb.mode));

  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bf,
                memMgrFreeList.allocate(2 * minBlock, mode));
  EXPECT_NE(bf.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bg,
                memMgrFreeList.allocate(3 * minBlock, mode));
  EXPECT_NE(bg.ptr, nullptr);
  ASSIGN_NO_ERR(MemoryManagerFreeList::Allocation, bh,
                memMgrFreeList.allocate(4 * minBlock, mode));
  EXPECT_NE(bh.ptr, nullptr);

  EXPECT_NO_ERR(memMgrFreeList.free(ba.ptr, ba.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(bc.ptr, bc.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(be.ptr, be.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(bf.ptr, bf.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(bg.ptr, bg.mode));
  EXPECT_NO_ERR(memMgrFreeList.free(bh.ptr, bh.mode));
}

TEST(DeviceContextTest, GetMemoryInfoWithoutMemoryManager) {
  // Test that getMemoryInfo returns device memory when memory manager is
  // disabled
  ASSIGN_NO_ERR(DeviceContextRef, ctx, DeviceContext::create(Device::cpuAPI));

  ASSIGN_NO_ERR(Device::MemInfo, memInfo, ctx->getMemoryInfo());
  ASSIGN_NO_ERR(Device::MemInfo, deviceMemInfo, ctx->device().getMemoryInfo());

  // Without memory manager, context memory info should match device memory info
  EXPECT_EQ(memInfo.total, deviceMemInfo.total);
  EXPECT_EQ(memInfo.free, deviceMemInfo.free);
}

TEST(DeviceContextTest, GetMemoryInfoWithMemoryManager) {
  // Test that getMemoryInfo returns memory manager memory when memory manager
  // is enabled
  constexpr size_t maxCacheSize = 4 * 1024;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;
  memInfo.total = 256 * maxCacheSize;

  ScopedEnvVar env[] = {
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "25"),
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
                   "25")};

  // Create a CPU device with memory manager enabled
  CpuDeviceRef device = CpuDeviceRef::create(memInfo);
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  ASSIGN_NO_ERR(Device::MemInfo, contextMemInfo, ctx->getMemoryInfo());
  ASSIGN_NO_ERR(Device::MemInfo, deviceMemInfo, ctx->device().getMemoryInfo());

  // With memory manager enabled, context memory info should differ from device
  // memory info
  EXPECT_NE(contextMemInfo.total, deviceMemInfo.total);
  EXPECT_NE(contextMemInfo.free, deviceMemInfo.free);

  // Context should report memory manager limits, not device limits
  size_t expectedCacheSize = (memInfo.total * 25) / 100;
  EXPECT_EQ(contextMemInfo.total, expectedCacheSize);
  EXPECT_EQ(contextMemInfo.free, expectedCacheSize);
}

TEST(DeviceContextTest, GetMemoryInfoDynamicUpdate) {
  // Test that getMemoryInfo reflects allocations/frees in memory manager
  constexpr size_t maxCacheSize = 8 * 1024;
  constexpr size_t allocSize = 1024;

  Device::MemInfo memInfo;
  memInfo.free = 128 * maxCacheSize;
  memInfo.total = 256 * maxCacheSize;

  ScopedEnvVar env[] = {
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE", "25"),
      ScopedEnvVar("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT",
                   "25")};

  // Create a CPU device with memory manager enabled
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

TEST(DeviceContextTest, VirtualDeviceModeDefaultDisabled) {
  // Test that virtual device mode is disabled by default
  EXPECT_FALSE(isVirtualDeviceMode());
  EXPECT_EQ(getVirtualDeviceCount(), 0);
}

TEST(DeviceContextTest, VirtualDeviceModeSetGet) {
  // Test setting and getting virtual device count
  EXPECT_FALSE(isVirtualDeviceMode());
  EXPECT_EQ(getVirtualDeviceCount(), 0);

  // Enable virtual device mode with 10 virtual devices
  setVirtualDeviceCount(10);
  EXPECT_TRUE(isVirtualDeviceMode());
  EXPECT_EQ(getVirtualDeviceCount(), 10);

  // Disable virtual device mode
  setVirtualDeviceCount(0);
  EXPECT_FALSE(isVirtualDeviceMode());
  EXPECT_EQ(getVirtualDeviceCount(), 0);
}

TEST(DeviceContextTest, VirtualDeviceModeCreatesVirtualDevice) {
  // Skip test if no GPU is available (virtual mode requires a real GPU)
  if (Device::numberOfDevices(Device::cudaAPI) == 0) {
    GTEST_SKIP() << "Skipping test: No GPU available for virtual device mode";
  }

  // Test that enabling virtual device mode creates VirtualDevice
  setVirtualDeviceCount(10);
  setVirtualDeviceAPI("cuda");
  setVirtualDeviceTargetArch("sm_90a");

  // Create a device - should be VirtualDevice
  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI, 0));

  // Verify it's a VirtualDevice by checking the name
  EXPECT_EQ(device->name(), "VirtualGPU-0");
  EXPECT_EQ(device->api(), Device::cudaAPI);
  EXPECT_EQ(device->id(), 0);

  // Clean up
  setVirtualDeviceCount(0);
}

TEST(DeviceContextTest, VirtualDeviceModeMultipleDevices) {
  // Skip test if no GPU is available (virtual mode requires a real GPU)
  if (Device::numberOfDevices(Device::cudaAPI) == 0) {
    GTEST_SKIP() << "Skipping test: No GPU available for virtual device mode";
  }

  // Test creating multiple virtual devices
  setVirtualDeviceCount(10);
  setVirtualDeviceAPI("cuda");
  setVirtualDeviceTargetArch("sm_90a");

  ASSIGN_NO_ERR(DeviceRef, device0, Device::create(Device::gpuAPI, 0));
  ASSIGN_NO_ERR(DeviceRef, device1, Device::create(Device::gpuAPI, 1));
  ASSIGN_NO_ERR(DeviceRef, device2, Device::create(Device::gpuAPI, 2));

  EXPECT_EQ(device0->name(), "VirtualGPU-0");
  EXPECT_EQ(device0->id(), 0);

  EXPECT_EQ(device1->name(), "VirtualGPU-1");
  EXPECT_EQ(device1->id(), 1);

  EXPECT_EQ(device2->name(), "VirtualGPU-2");
  EXPECT_EQ(device2->id(), 2);

  // Clean up
  setVirtualDeviceCount(0);
}

TEST(DeviceContextTest, VirtualDeviceModeCpuDeviceUnaffected) {
  // Test that CPU device creation is not affected by virtual device mode
  setVirtualDeviceCount(10);
  setVirtualDeviceAPI("cuda");
  setVirtualDeviceTargetArch("sm_90a");

  // Create a CPU device - should still be a real CPU device
  ASSIGN_NO_ERR(DeviceRef, cpuDevice, Device::create(Device::cpuAPI, 0));

  // Verify it's a real CPU device
  EXPECT_EQ(cpuDevice->api(), Device::cpuAPI);
  EXPECT_EQ(cpuDevice->id(), 0);

  // Clean up
  setVirtualDeviceCount(0);
}

TEST(DeviceContextTest, VirtualDeviceModeWithDeviceContext) {
  // Skip test if no GPU is available (virtual mode requires a real GPU)
  if (Device::numberOfDevices(Device::cudaAPI) == 0) {
    GTEST_SKIP() << "Skipping test: No GPU available for virtual device mode";
  }

  // Test that VirtualDevice can create a context
  setVirtualDeviceCount(10);
  setVirtualDeviceAPI("cuda");
  setVirtualDeviceTargetArch("sm_90a");

  ASSIGN_NO_ERR(DeviceRef, device, Device::create(Device::gpuAPI, 0));
  ASSIGN_NO_ERR(DeviceContextRef, ctx, device->createContext());

  // Verify the context's device is correct
  EXPECT_EQ(ctx->device().name(), "VirtualGPU-0");
  EXPECT_EQ(ctx->device().id(), 0);

  // Test synchronize (should be a no-op for VirtualDevice)
  EXPECT_NO_ERR(ctx->synchronize());

  // Clean up
  setVirtualDeviceCount(0);
}
} // namespace
