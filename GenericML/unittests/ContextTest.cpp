//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/StateContext.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/TypeID.h"

#include "gtest/gtest.h"

using namespace M;

struct PayloadA : public ReferenceCounted<PayloadA> {
  explicit PayloadA(int i) : i(i) { resource = malloc(1); }
  ~PayloadA() { free(resource); }
  int i;
  void *resource;
};

struct PayloadB : public ReferenceCounted<PayloadB> {
  explicit PayloadB(char c) : c(c) {}
  char c;
};

static std::unique_ptr<LLCL::Runtime> getRuntime() {
  return LLCL::createUniqueRuntime(LLCL::RuntimeOptions().forDebug());
}

#ifndef NDEBUG
TEST(ContextTest, StateSlotDowncastFailTest) {
  StateContext ctx(/*numSlots=*/1);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ASSERT_DEATH_IF_SUPPORTED(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/0),
                            "mismatch between actual and expected type ids");
}
#endif

TEST(ContextTest, StateSlotTest) {
  StateContext ctx(/*numSlots=*/2);
  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ctx.setStateSlot(/*slotIndex=*/1, RCRef<PayloadB>::create('a'),
                   /*numBuffers=*/1);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadA>(/*slotIndex=*/0).i, 43);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/1).c, 'a');
}

TEST(ContextTest, getOrCreateBufferTest) {
  auto runtime = getRuntime();
  {
    // Add scope to make sure that the allocator outlives the context.
    StateContext ctx(/*numSlots=*/1);
    ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);

    auto &stateSlot = ctx.getStateSlot(/*slotIndex=*/0);
    {
      auto buffer = stateSlot.getOrCreateBuffer(
          /*bufferIndex=*/0, *runtime, /*size=*/1, /*alignment=*/{});

      // getOrCreateBuffer call with the same slot and the same size returns a
      // cached buffer.
      auto buffer2 = stateSlot.getOrCreateBuffer(
          /*bufferIndex=*/0, *runtime, /*size=*/1, /*alignment=*/{});
      EXPECT_EQ(buffer.getBuffer(), buffer2.getBuffer());

      // We're done with the above buffers, so safe to reallocate.
    }

    // Call with a different size will allocate a new buffer.
    auto buffer3 = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, *runtime, /*size=*/2, /*alignment=*/{});
    EXPECT_NE(buffer3.getSize(), 1UL);
  }
}
