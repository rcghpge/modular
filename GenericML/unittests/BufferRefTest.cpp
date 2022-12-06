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

static void Increment(int *p) { *p += 1; }

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

TEST(BufferRefTest, brrowedTest) {
  auto runtime = GetRuntime();

  auto ref = BufferRef::createOwned(runtime, sizeof(int), {});
  auto ref2 = ref.borrow();
  Increment((int *)ref.getData());
  ASSERT_EQ(*(int *)ref.getData(), *(int *)ref2.getData());
}

TEST(BufferRefTest, deepCopyTest) {
  auto runtime = GetRuntime();
  int *data =
      (int *)runtime->allocateBytes(sizeof(int), M::kPreferredMemoryAlignment);
  *data = 0;
  auto ref = BufferRef::createBorrowed(data, sizeof(int), /*alignment=*/{});
  auto ref2 = ref.deepCopy(runtime);
  Increment((int *)ref.getData());
  ASSERT_EQ(*(int *)ref.getData(), 1);
  ASSERT_EQ(*(int *)ref2.getData(), 0);
  // deepCopy returns owned reference.
  ASSERT_EQ(ref2.getStorageKind(), BufferRef::kOwned);
  runtime->deallocateBytes(data, sizeof(int));
}

TEST(BufferRefTest, releaseTest) {
  auto runtime = GetRuntime();
  int *data = (int *)malloc(sizeof(int));
  auto ref = BufferRef::createBorrowed(data, sizeof(int), /*alignment=*/{});
  auto ref2 = std::move(ref);
  ASSERT_EQ(ref2.getStorageKind(), BufferRef::kBorrowed);
  free(data);

  auto ref3 = BufferRef::createOwned(runtime, sizeof(int), {});
  auto ref4 = std::move(ref3);
  ASSERT_EQ(ref4.getStorageKind(), BufferRef::kOwned);
}
