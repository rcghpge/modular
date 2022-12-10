//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/Support/BufferRef.h"

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::GML;

static LLCL::Runtime *getRuntime() {
  static LLCL::Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        LLCL::createProfilingAllocator(LLCL::createMallocAllocator());
    std::unique_ptr<LLCL::WorkQueue> workQueue =
        LLCL::createSingleThreadWorkQueue();
    return new LLCL::Runtime(std::move(allocator), std::move(workQueue));
  }();
  return runtime;
}

TEST(BufferRef, Default) {
  auto runtime = getRuntime();
  BufferRef ref;

  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref.getSize(), 0U);
  ASSERT_EQ(ref.getAlignment(), llvm::None);
  ASSERT_FALSE(ref.getRuntime());
  ASSERT_FALSE(ref.isOwned());
  ASSERT_EQ(ref.getLifetime(), BufferRef::kLocal);

  ref = BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/{});
  ASSERT_EQ(ref.getSize(), sizeof(int));
}

TEST(BufferRef, Move) {
  auto runtime = getRuntime();
  BufferRef ref =
      BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
  const void *data = ref.getBuffer();

  BufferRef ref2 = std::move(ref);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref2.getBuffer(), data);
}

TEST(BufferRef, BorrowError) {
  auto runtime = getRuntime();

  BufferRef ref =
      BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
  BufferRef bor = ref.borrow(BufferRef::kExecute);
  ErrorOr<BufferRef> bor2OrErr = bor.borrowOrError(BufferRef::kConstant);
  ASSERT_TRUE(bor2OrErr.isError());
  ErrorOr<BufferRef> bor2OrErr2 = bor.borrowOrError(BufferRef::kLocal);
  ASSERT_FALSE(bor2OrErr2.isError());
}

TEST(BufferRef, Borrow) {
  auto runtime = getRuntime();

  BufferRef ref =
      BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
  BufferRef bor = ref.borrow(BufferRef::kExecute);
  ASSERT_EQ(bor.getBuffer(), ref.getBuffer());
  ASSERT_EQ(bor.getSize(), ref.getSize());
  ASSERT_EQ(bor.getAlignment(), ref.getAlignment());
  ASSERT_TRUE(ref.isOwned());
  ASSERT_FALSE(bor.getRuntime());
  ASSERT_FALSE(bor.isOwned());
  ASSERT_EQ(bor.getLifetime(), BufferRef::kExecute);
}

TEST(BufferRef, DeepCopy) {
  auto runtime = getRuntime();
  BufferRef ref =
      BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
  memset(ref.getMutableBuffer(), 42, sizeof(int));
  BufferRef bor = ref.borrow(BufferRef::kExecute);
  BufferRef copy = bor.deepCopy(runtime);
  ASSERT_NE(copy.getBuffer(), bor.getBuffer());
  ASSERT_EQ(memcmp(bor.getBuffer(), copy.getBuffer(), sizeof(int)), 0);
  ASSERT_EQ(copy.getSize(), bor.getSize());
  ASSERT_EQ(copy.getAlignment(), bor.getAlignment());
  ASSERT_TRUE(copy.isOwned());
  ASSERT_EQ(copy.getLifetime(), BufferRef::kLocal);
}

TEST(BufferRef, Destructor) {
  // We are relying on the use of the profiling Allocator in getRuntime.
  auto runtime = getRuntime();
  void *data = runtime->allocateBytes(sizeof(int), 16);
  {
    BufferRef ref = BufferRef::createBorrowed(
        data, sizeof(int), /*alignment=*/16, BufferRef::kLocal);
    BufferRef ref2 = std::move(ref);
    // No deallocation.
  }
  runtime->deallocateBytes(data, sizeof(int));

  {
    BufferRef ref =
        BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
    BufferRef ref2 = std::move(ref);
    // Will deallocate.
  }
}
