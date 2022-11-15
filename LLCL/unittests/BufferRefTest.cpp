//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Support/BufferRef.h"
#include "LLCL/Runtime/Allocator.h"

#include "gtest/gtest.h"

using namespace M::LLCL;

static void Increment(int *p) { *p += 1; }

TEST(BufferRefTest, brrowedTest) {
  auto allocator = createMallocAllocator();
  auto ref = BufferRef::createOwned(allocator.get(), sizeof(int));
  auto ref2 = ref.borrow();
  Increment((int *)ref.getData());
  ASSERT_EQ(*(int *)ref.getData(), *(int *)ref2.getData());
}

TEST(BufferRefTest, deepCopyTest) {
  auto allocator = createMallocAllocator();
  int *data = (int *)allocator->allocateBytes(sizeof(int),
                                              M::kPreferredMemoryAlignment);
  *data = 0;
  auto ref = BufferRef::createBorrowed(data, sizeof(int));
  auto ref2 = ref.deepCopy(allocator.get());
  Increment((int *)ref.getData());
  ASSERT_EQ(*(int *)ref.getData(), 1);
  ASSERT_EQ(*(int *)ref2.getData(), 0);
  // deepCopy returns owned reference.
  ASSERT_EQ(ref2.getStorageKind(), BufferRef::kOwned);
  allocator->deallocateBytes(data, sizeof(int));
}

TEST(BufferRefTest, releaseTest) {
  int *data = (int *)malloc(sizeof(int));
  auto ref = BufferRef::createBorrowed(data, sizeof(int));
  auto ref2 = std::move(ref);
  ASSERT_EQ(ref2.getStorageKind(), BufferRef::kBorrowed);
  free(data);

  auto allocator = createMallocAllocator();
  auto ref3 = BufferRef::createOwned(allocator.get(), sizeof(int));
  auto ref4 = std::move(ref3);
  ASSERT_EQ(ref4.getStorageKind(), BufferRef::kOwned);
}
