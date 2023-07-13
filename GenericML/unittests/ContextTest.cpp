//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/Context.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/TypeID.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::GML;

struct PayloadA {
  explicit PayloadA(int i) : i(i) { resource = malloc(1); }
  ~PayloadA() { free(resource); }
  int i;
  void *resource;
};

struct PayloadB {
  explicit PayloadB(char c) : c(c) {}
  char c;
};

static LLCL::Runtime *GetRuntime() {
  static LLCL::Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        LLCL::createProfilingAllocator(LLCL::createMallocAllocator());
    std::unique_ptr<LLCL::WorkQueue> workQueue =
        LLCL::createSingleThreadWorkQueue();
    return new LLCL::Runtime(std::move(allocator), std::move(workQueue));
  }();
  return runtime;
}

#ifndef NDEBUG
TEST(ContextTest, StateSlotDowncastFailTest) {
  Context ctx(/*numSlots=*/1);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ASSERT_DEATH_IF_SUPPORTED(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/0),
                            "mismatch between actual and expected type ids");
}
#endif

TEST(ContextTest, StateSlotTest) {
  Context ctx(/*numSlots=*/2);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ctx.setStateSlot(/*slotIndex=*/1, std::make_unique<PayloadB>('a'),
                   /*numBuffers=*/1);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadA>(/*slotIndex=*/0).i, 43);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/1).c, 'a');
}

TEST(ContextTest, getOrCreateBufferTest) {
  auto runtime = GetRuntime();
  {
    // Add scope to make sure that the allocator outlives the context.
    Context ctx(/*numSlots=*/1);
    ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);

    auto &stateSlot = ctx.getStateSlot(/*slotIndex=*/0);
    auto buffer = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, runtime, /*size=*/1, /*alignment=*/{});
    size_t size = buffer.getSize();
    EXPECT_EQ(buffer.getOwnership(), GML::BufferRef::kBorrowed);

    // getOrCreateBuffer call with the same slot and the same size returns a
    // cached buffer.
    auto buffer2 = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, runtime, /*size=*/1, /*alignment=*/{});
    EXPECT_EQ(buffer.getBuffer(), buffer2.getBuffer());

    // Call with a different size will allocate a new buffer.
    auto buffer3 = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, runtime, /*size=*/2, /*alignment=*/{});
    EXPECT_NE(buffer3.getSize(), size);
  }
}
