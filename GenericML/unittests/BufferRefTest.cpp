//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/BufferRef.h"

#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Location.h"
#include "LLCL/Support/UnknownLocationDecoder.h"

#include "gtest/gtest.h"

using namespace M;
using namespace M::GML;

static std::unique_ptr<LLCL::Runtime> getRuntime() {
  return LLCL::createUniqueRuntime(LLCL::RuntimeOptions().forDebug());
}

TEST(BufferRef, Default) {
  BufferRef ref;

  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref.getSize(), 0U);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(BufferRef, CreateOwned) {
  auto runtime = getRuntime();
  BufferRef ref = BufferRef::createOwned(*runtime, /*size=*/4,
                                         /*alignment=*/128);
  ASSERT_TRUE(llvm::isAddrAligned(llvm::Align(128), ref.getBuffer()));
  ASSERT_TRUE((bool)ref.getMemStorageHandle());
}

TEST(BufferRef, CreateEmptyOwned) {
  auto runtime = getRuntime();
  BufferRef ref = BufferRef::createOwned(*runtime, /*size=*/0,
                                         /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(BufferRef, CreateBorrowed) {
  uint8_t data[4] = {1, 2, 3, 4};
  BufferRef ref = BufferRef::createWithNonTrackedMemory(data, /*size=*/4,
                                                        /*alignment=*/{});
  ASSERT_EQ(ref.getSize(), 4u);
  ASSERT_EQ(ref.getBuffer(), data);
}

TEST(BufferRef, CreateEmptyBorrowed) {
  BufferRef ref = BufferRef::createWithNonTrackedMemory(
      /*buffer=*/(void *)nullptr, /*size=*/0,
      /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(BufferRef, Move) {
  auto runtime = getRuntime();
  BufferRef ref =
      BufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
  const void *data = ref.getBuffer();

  BufferRef ref2 = std::move(ref);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref2.getBuffer(), data);
}

TEST(BufferRef, Borrow) {
  auto runtime = getRuntime();
  BufferRef ref =
      BufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
  BufferRef bor = ref.borrow();
  ASSERT_EQ(bor.getBuffer(), ref.getBuffer());
  ASSERT_EQ(bor.getSize(), ref.getSize());
  ASSERT_EQ(bor.getAlignment(), ref.getAlignment());
}

TEST(BufferRef, Destructor) {
  // We are relying on the use of the profiling Allocator in getRuntime.
  auto runtime = getRuntime();
  void *data = runtime->getAllocator()->allocateBytes(sizeof(int), 16);
  {
    BufferRef ref = BufferRef::createWithNonTrackedMemory(data, sizeof(int),
                                                          /*alignment=*/16);
    BufferRef ref2 = std::move(ref);
    // No deallocation.
  }
  runtime->getAllocator()->deallocateBytes(data, sizeof(int));

  {
    BufferRef ref =
        BufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
    BufferRef ref2 = std::move(ref);
    // Will deallocate.
  }
}

TEST(BufferRef, GetMemStorageHandle) {
  auto runtime = getRuntime();
  BufferRef *ref = new BufferRef(BufferRef::createOwned(*runtime, 4,
                                                        /*alignment=*/{}));
  AnyAsyncValueRef memory = ref->getMemStorageHandle();
  ASSERT_TRUE(memory);

  OwnedMemoryBlock &m = memory.get<OwnedMemoryBlock>();

  ASSERT_EQ(ref->getBuffer(), m.data);

  // This should not delete the underlying memory.
  delete ref;
}
