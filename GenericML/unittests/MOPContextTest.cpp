//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MOP/Context.h"
#include "LLCL/Runtime/Allocator.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::MOP;

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

#ifndef NDEBUG
TEST(ContextTest, StateSlotDowncastFailTest) {
  Context ctx(/*numSlots=*/1);
  StateSlot::registerType<PayloadA>();

  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ASSERT_DEATH_IF_SUPPORTED(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/0),
                            "(mismatched static and dynamic StateSlot types)");
}
#endif

TEST(ContextTest, StateSlotTest) {
  Context ctx(/*numSlots=*/2);
  StateSlot::registerType<PayloadA>();
  StateSlot::registerType<PayloadB>();

  ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);
  ctx.emplaceStateSlot<PayloadB>(/*slotIndex=*/1, /*numBuffers=*/1, 'a');

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadA>(/*slotIndex=*/0).i, 43);

  EXPECT_EQ(ctx.getStateSlotPayload<PayloadB>(/*slotIndex=*/1).c, 'a');
}

TEST(ContextTest, getOrCreateBufferTest) {
  auto allocator = LLCL::createMallocAllocator();
  {
    // Add scope to make sure that the allocator outlives the context.
    Context ctx(/*numSlots=*/1);
    StateSlot::registerType<PayloadA>();
    ctx.emplaceStateSlot<PayloadA>(/*slotIndex=*/0, /*numBuffers=*/1, 43);

    auto &stateSlot = ctx.getStateSlot(/*slotIndex=*/0);
    auto buffer = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, /*allocator=*/allocator.get(), /*size=*/1);
    size_t size = buffer.getSize();
    EXPECT_EQ(buffer.getStorageKind(), LLCL::BufferRef::kBorrowed);

    // getOrCreateBuffer call with the same slot and the same size returns a
    // cached buffer.
    auto buffer2 = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, /*allocator=*/allocator.get(), /*size=*/1);
    EXPECT_EQ(buffer.getData(), buffer2.getData());

    // Call with a different size will allocate a new buffer.
    auto buffer3 = stateSlot.getOrCreateBuffer(
        /*bufferIndex=*/0, /*allocator=*/allocator.get(), /*size=*/2);
    EXPECT_NE(buffer3.getSize(), size);
  }
}
