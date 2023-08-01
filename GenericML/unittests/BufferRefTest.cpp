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
  BufferRef ref;

  ASSERT_FALSE(ref);
  ASSERT_EQ(ref.getBuffer(), nullptr);
  ASSERT_EQ(ref.getSize(), 0U);
  ASSERT_EQ(ref.getAlignment(), std::nullopt);
  ASSERT_FALSE(ref.getRuntime());
  ASSERT_EQ(ref.getOwnership(), BufferRef::kNull);
  ASSERT_EQ(ref.getLifetime(), BufferRef::kLocal);
}

TEST(BufferRef, CreateOwned) {
  auto runtime = getRuntime();
  uint8_t data[4] = {1, 2, 3, 4};
  BufferRef ref = BufferRef::createOwned(runtime, /*size=*/4,
                                         /*alignment=*/128, /*copyFrom=*/data);
  ASSERT_TRUE(ref);
  ASSERT_TRUE(llvm::isAddrAligned(llvm::Align(128), ref.getBuffer()));
  ASSERT_EQ(memcmp(data, ref.getBuffer(), 4), 0);
}

TEST(BufferRef, CreateEmptyOwned) {
  auto runtime = getRuntime();
  BufferRef ref = BufferRef::createOwned(runtime, /*size=*/0,
                                         /*alignment=*/128);
  ASSERT_TRUE(ref);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_NE(ref.getBuffer(), nullptr);
  ASSERT_NE(ref.getMutableBuffer(), nullptr);
};

TEST(BufferRef, CreateBorrowed) {
  uint8_t data[4] = {1, 2, 3, 4};
  BufferRef ref =
      BufferRef::createBorrowed(data, /*size=*/4,
                                /*alignment=*/{}, BufferRef::kConstant);
  ASSERT_EQ(ref.getSize(), 4u);
  ASSERT_EQ(ref.getBuffer(), data);
}

TEST(BufferRef, CreateEmptyBorrowed) {
  BufferRef ref =
      BufferRef::createBorrowed(/*buffer=*/(void *)nullptr, /*size=*/0,
                                /*alignment=*/128, BufferRef::kConstant);
  ASSERT_TRUE(ref);
  ASSERT_EQ(ref.getSize(), 0u);
  ASSERT_NE(ref.getBuffer(), nullptr);
  ASSERT_NE(ref.getMutableBuffer(), nullptr);
}

TEST(BufferRef, CreateSafeBorrowed) {
  auto runtime = getRuntime();
  BufferRef owned =
      BufferRef::createOwned(runtime, /*size=*/8, /*alignment=*/64);
  auto borrowe =
      AsyncValueRef<BufferRef>::createReady(runtime, std::move(owned));
  {
    BufferRef borrowed = BufferRef::createSafeBorrowedSlice(borrowe.copy());
    ASSERT_EQ(borrowed.getSize(), 8u);
    ASSERT_EQ(borrowed.getAlignment(), 64u);
    ASSERT_EQ(borrowed.getBuffer(), borrowe.get().getBuffer());
    ASSERT_EQ(borrowe.getPointer()->getRefCountForDebugging(), 2u);
  }
  ASSERT_EQ(borrowe.getPointer()->getRefCountForDebugging(), 1u);
}

TEST(BufferRef, CreateSafeBorrowedSlice) {
  auto runtime = getRuntime();
  BufferRef owned =
      BufferRef::createOwned(runtime, /*size=*/8, /*alignment=*/64);
  auto borrowe =
      AsyncValueRef<BufferRef>::createReady(runtime, std::move(owned));
  BufferRef borrowed = BufferRef::createSafeBorrowedSlice(
      borrowe.copy(), /*offset=*/2, /*size=*/4);
  ASSERT_EQ(borrowed.getSize(), 4u);
  ASSERT_EQ(borrowed.getAlignment(), std::nullopt);
  ASSERT_EQ(borrowed.getBuffer(), borrowe.get().getBuffer<uint8_t>() + 2);
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
  ASSERT_EQ(ref.getOwnership(), BufferRef::kOwned);
  ASSERT_FALSE(bor.getRuntime());
  ASSERT_EQ(bor.getOwnership(), BufferRef::kBorrowed);
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
  ASSERT_EQ(copy.getOwnership(), BufferRef::kOwned);
  ASSERT_EQ(copy.getLifetime(), BufferRef::kLocal);
}

TEST(BufferRef, Destructor) {
  // We are relying on the use of the profiling Allocator in getRuntime.
  auto runtime = getRuntime();
  void *data = runtime->getAllocator()->allocateBytes(sizeof(int), 16);
  {
    BufferRef ref = BufferRef::createBorrowed(
        data, sizeof(int), /*alignment=*/16, BufferRef::kLocal);
    BufferRef ref2 = std::move(ref);
    // No deallocation.
  }
  runtime->getAllocator()->deallocateBytes(data, sizeof(int));

  {
    BufferRef ref =
        BufferRef::createOwned(runtime, sizeof(int), /*alignment=*/16);
    BufferRef ref2 = std::move(ref);
    // Will deallocate.
  }
}

TEST(BufferRef, ReleaseBufferIfOwned) {
  auto runtime = getRuntime();
  uint8_t data[] = {1, 2, 3, 4};
  BufferRef ref = BufferRef::createOwned(runtime, 4,
                                         /*alignment=*/{}, data);
  const void *released = ref.releaseBufferIfOwned();
  ASSERT_TRUE(released != nullptr);
  ASSERT_FALSE(ref);
  ASSERT_EQ(memcmp(data, released, 4), 0);
  runtime->getAllocator()->deallocateBytes(const_cast<void *>(released), 4);
}

struct TestForeignObject {
  int32_t buffer[4] = {1, 2, 3, 4};
  bool deleted = false;
};

class TestForeignBufferInterface : public ForeignBufferInterface {
public:
  TestForeignBufferInterface()
      : ForeignBufferInterface("TestForeignBufferInterface") {}
  ~TestForeignBufferInterface() override = default;

  const void *getBuffer(const void *object) const override {
    return cast(object)->buffer;
  }
  void deleteObject(void *object) const override {
    assert(!cast(object)->deleted && "already deleted");
    cast(object)->deleted = true;
  }

private:
  static TestForeignObject *cast(void *object) {
    return static_cast<TestForeignObject *>(object);
  }
  static const TestForeignObject *cast(const void *object) {
    return static_cast<const TestForeignObject *>(object);
  }
};

TEST(BufferRef, CreateForeign) {
  ForeignBufferInterfaceID id =
      ForeignBufferInterfaceRegistry::registerInterface(
          std::make_unique<TestForeignBufferInterface>());
  ASSERT_EQ(ForeignBufferInterfaceRegistry::registerInterface(
                std::make_unique<TestForeignBufferInterface>()),
            id);
  TestForeignObject object;
  {
    BufferRef ref = BufferRef::createForeign(id, &object, 4, /*alignment=*/{});
    ASSERT_EQ(ref.getBuffer(), object.buffer);
    ASSERT_EQ(ref.getSize(), 4U);
    ASSERT_EQ(ref.getOwnership(), BufferRef::kForeign);
  }
  ASSERT_TRUE(object.deleted);
}

TEST(BufferRef, MaybeReleaseForeignObject) {
  ForeignBufferInterfaceID id =
      ForeignBufferInterfaceRegistry::registerInterface(
          std::make_unique<TestForeignBufferInterface>());
  TestForeignObject object;
  BufferRef ref = BufferRef::createForeign(id, &object, 4, /*alignment=*/{});
  void *released = ref.maybeReleaseForeignObject(id);
  ASSERT_FALSE(ref);
  ASSERT_EQ(released, &object);
}

#if MODULAR_PARANOID
TEST(BufferRef, TransitiveDataRace_ExpectDeath) {
  auto runtime = getRuntime();
  auto parent = AsyncValueRef<BufferRef>::createReady(
      runtime, BufferRef::createOwned(runtime, /*size=*/42, /*alignment=*/{}));
  auto childA = AsyncValueRef<BufferRef>::createReady(
      runtime, BufferRef::createSafeBorrowedSlice(parent.copy(), /*offset=*/8,
                                                  /*size=*/8));
  auto childB = AsyncValueRef<BufferRef>::createReady(
      runtime, BufferRef::createSafeBorrowedSlice(parent.copy(), /*offset=*/8,
                                                  /*size=*/8));
  SmallVector<LLCL::ResourceUse> childAUses;
  childA->collectAccessUses(childAUses, "childA", LLCL::kReadingResourceUse);
  EXPECT_DEATH_IF_SUPPORTED(
      childB->collectAccessUses(childAUses, "childB",
                                LLCL::kWritingResourceUse),
      "attempting to write to \\(section of\\) resource which is also being "
      "read");
}
#endif // MODULAR_PARANOID
