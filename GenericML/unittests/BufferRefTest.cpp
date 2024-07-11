//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "ArraySupport/TensorBufferRef.h"

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Support/Location.h"
#include "AsyncRT/Support/UnknownLocationDecoder.h"

#include "gtest/gtest.h"

using namespace M;

static std::unique_ptr<AsyncRT::Runtime> getRuntime() {
  return AsyncRT::createUniqueRuntime(AsyncRT::RuntimeOptions().forDebug());
}

TEST(TensorBufferRef, Default) {
  TensorBufferRef ref;

  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref.getSize(), 0U);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateOwned) {
  auto runtime = getRuntime();
  TensorBufferRef ref = TensorBufferRef::createOwned(*runtime, /*size=*/4,
                                                     /*alignment=*/128);
  ASSERT_TRUE(llvm::isAddrAligned(llvm::Align(128), ref.getBuffer()));
  ASSERT_TRUE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateEmptyOwned) {
  auto runtime = getRuntime();
  TensorBufferRef ref = TensorBufferRef::createOwned(*runtime, /*size=*/0,
                                                     /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, CreateBorrowed) {
  uint8_t data[4] = {1, 2, 3, 4};
  TensorBufferRef ref =
      TensorBufferRef::createWithNonTrackedMemory(data, /*size=*/4,
                                                  /*alignment=*/{});
  ASSERT_EQ(ref.getSize(), 4u);
  ASSERT_EQ(ref.getBuffer(), data);
}

TEST(TensorBufferRef, CreateEmptyBorrowed) {
  TensorBufferRef ref = TensorBufferRef::createWithNonTrackedMemory(
      /*buffer=*/(void *)nullptr, /*size=*/0,
      /*alignment=*/128);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_FALSE((bool)ref.getMemStorageHandle());
}

TEST(TensorBufferRef, Move) {
  auto runtime = getRuntime();
  TensorBufferRef ref =
      TensorBufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
  const void *data = ref.getBuffer();

  TensorBufferRef ref2 = std::move(ref);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref2.getBuffer(), data);
}

TEST(TensorBufferRef, Borrow) {
  auto runtime = getRuntime();
  TensorBufferRef ref =
      TensorBufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
  TensorBufferRef bor = ref.borrow();
  ASSERT_EQ(bor.getBuffer(), ref.getBuffer());
  ASSERT_EQ(bor.getSize(), ref.getSize());
}

TEST(TensorBufferRef, Destructor) {
  // We are relying on the use of the profiling Allocator in getRuntime.
  auto runtime = getRuntime();
  void *data = runtime->getAllocator()->allocateBytes(sizeof(int), 16);
  {
    TensorBufferRef ref =
        TensorBufferRef::createWithNonTrackedMemory(data, sizeof(int),
                                                    /*alignment=*/16);
    TensorBufferRef ref2 = std::move(ref);
    // No deallocation.
  }
  runtime->getAllocator()->deallocateBytes(data, sizeof(int));

  {
    TensorBufferRef ref =
        TensorBufferRef::createOwned(*runtime, sizeof(int), /*alignment=*/16);
    TensorBufferRef ref2 = std::move(ref);
    // Will deallocate.
  }
}

TEST(TensorBufferRef, GetMemStorageHandle) {
  auto runtime = getRuntime();
  TensorBufferRef *ref =
      new TensorBufferRef(TensorBufferRef::createOwned(*runtime, 4,
                                                       /*alignment=*/{}));
  AnyAsyncValueRef memory = ref->getMemStorageHandle();
  ASSERT_TRUE(memory);

  OwnedMemoryBlock &m = memory.get<OwnedMemoryBlock>();

  ASSERT_EQ(ref->getBuffer(), m.data);

  // This should not delete the underlying memory.
  delete ref;
}
