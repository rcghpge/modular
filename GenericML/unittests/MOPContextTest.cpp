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

struct PrimContextA {
  PrimContextA(int i) : i(i) { resource = malloc(1); }
  ~PrimContextA() { free(resource); }
  int i;
  void *resource;
};

struct PrimContextB {
  PrimContextB(char c) : c(c) {}
  char c;
};

#ifndef NDEBUG
TEST(ContextTest, FrameworkPrimContextDowncastFailTest) {
  Context ctx(2);
  AnyGenericPrimContext::registerFramework<PrimContextA>();

  ctx.emplacePrimContext<PrimContextA>(0, 1, 43);
  auto &primCtxA = ctx.getPrimContext(0);
  ASSERT_DEATH_IF_SUPPORTED(primCtxA.getFrameworkPrimContext<PrimContextB>(),
                            "(typeID == getTypeID<T>())");
}
#endif

TEST(ContextTest, FrameworkPrimContextTest) {
  Context ctx(2);
  AnyGenericPrimContext::registerFramework<PrimContextA>();
  AnyGenericPrimContext::registerFramework<PrimContextB>();

  ctx.emplacePrimContext<PrimContextA>(0, 1, 43);
  ctx.emplacePrimContext<PrimContextB>(1, 1, 'a');

  auto &primCtxA = ctx.getPrimContext(0);
  EXPECT_EQ(primCtxA.getTypeID(), 0);
  EXPECT_EQ(primCtxA.getFrameworkPrimContext<PrimContextA>().i, 43);

  auto &primCtxB = ctx.getPrimContext(1);
  EXPECT_EQ(primCtxB.getTypeID(), 1);
  EXPECT_EQ(primCtxB.getFrameworkPrimContext<PrimContextB>().c, 'a');
}

TEST(ContextTest, getOrCreateBufferTest) {
  auto allocator = LLCL::createMallocAllocator();
  {
    // Add scope to make sure that the allocator outlives the context.
    Context ctx(1);
    AnyGenericPrimContext::registerFramework<PrimContextA>();
    ctx.emplacePrimContext<PrimContextA>(0, 1, 43);

    auto &primCtx = ctx.getPrimContext(0);
    auto buffer = primCtx.getOrCreateBuffer(
        /*index=*/0, /*allocator=*/allocator.get(), /*size=*/1);
    void *data = buffer.getData();
    EXPECT_EQ(buffer.getStorageKind(), LLCL::BufferRef::kBorrowed);

    // getOrCreateBuffer call with the same slot and the same size returns a
    // cached buffer.
    auto buffer2 = primCtx.getOrCreateBuffer(
        /*index=*/0, /*allocator=*/allocator.get(), /*size=*/1);
    EXPECT_EQ(buffer.getData(), buffer2.getData());

    // Call with a different size will allocate a new memory.
    auto buffer3 = primCtx.getOrCreateBuffer(
        /*index=*/0, /*allocator=*/allocator.get(), /*size=*/2);
    EXPECT_NE(buffer3.getData(), data);
  }
}
