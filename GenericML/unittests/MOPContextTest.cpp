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
    auto buffer = primCtx.getOrCreateBuffer(0, allocator.get(), 1);
    // refcount is 2, one for `buffer` and the other for the output cache.
    EXPECT_EQ(buffer->getNumReferences(), 2);

    {
      // Get the buffer from the cache. refcount added by one.
      auto buffer2 = primCtx.getOrCreateBuffer(0, allocator.get(), 1);
      EXPECT_EQ(buffer->getNumReferences(), 3);
      EXPECT_EQ(buffer->getData(), buffer2->getData());
    }

    // Create a buffer with a different size, overwriting the cache slot 0.
    auto buffer3 = primCtx.getOrCreateBuffer(0, allocator.get(), 2);
    EXPECT_EQ(buffer3->getNumReferences(), 2);
    // Now the buffer backed by buffer is kicked out of cache.
    EXPECT_EQ(buffer->getNumReferences(), 1);
  }
}

TEST(ContextTes, registerBufferGetBufferTest) {
  auto allocator = LLCL::createMallocAllocator();
  {
    // Add scope to make sure that the allocator outlives the context.
    Context ctx(1);
    AnyGenericPrimContext::registerFramework<PrimContextA>();
    ctx.emplacePrimContext<PrimContextA>(0, 1, 43);

    auto &primCtx = ctx.getPrimContext(0);
    auto bufferOr = primCtx.getBuffer(0);
    EXPECT_TRUE(bufferOr.isError());
    EXPECT_EQ(bufferOr.getError(),
              "Buffer not cached in the given output index");

    auto buffer = LLCL::RCBuffer::createHeap(allocator.get(), 1);
    EXPECT_EQ(buffer->getNumReferences(), 1);

    // Register to cache bumps the refcount.
    primCtx.registerBuffer(0, buffer.copy());
    EXPECT_EQ(buffer->getNumReferences(), 2);

    bufferOr = primCtx.getBuffer(0);
    EXPECT_FALSE(bufferOr);
    EXPECT_EQ(buffer->getNumReferences(), 3);
    EXPECT_EQ(buffer->getData(), (*bufferOr)->getData());
  }
}
