//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Some token unit tests for generic Tensor.
//===----------------------------------------------------------------------===//

#include "GenericML/Support/Tensor.h"
#include "LLCL/Runtime/Allocator.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/WorkQueue.h"

#include "gtest/gtest.h"

using namespace M;

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

constexpr size_t kN = 4;
constexpr size_t kM = 6;

static Tensor makeTensor() {
  M::TensorSpec spec({kN, kM}, DType::getInt(8, /*isSigned=*/false).value());
  M::Tensor result =
      Tensor::createOwned(std::move(spec), getRuntime(), /*alignment=*/{});
  for (size_t i = 0, e = result.getSizeInBytes(); i < e; ++i)
    static_cast<uint8_t *>(result.getPointer())[i] = static_cast<uint8_t>(i);
  return result;
}

static void incrementContents(Tensor &tensor) {
  for (size_t i = 0; i < tensor.getSizeInBytes(); ++i)
    ++static_cast<uint8_t *>(tensor.getPointer())[i];
}

static bool haveSameContents(Tensor &lhs, Tensor &rhs) {
  if (lhs.getSizeInBytes() != rhs.getSizeInBytes())
    return false;
  return std::memcmp(lhs.getPointer(), rhs.getPointer(),
                     lhs.getSizeInBytes()) == 0;
}

TEST(Tensor, ReleaseBuffer) {
  auto runtime = getRuntime();
  M::Tensor tensor = makeTensor();
  M::Tensor tensor2 = tensor.deepCopy(runtime);
  GML::BufferRef ref = tensor.releaseBuffer();
  ASSERT_EQ(ref.getSize(), tensor2.getSizeInBytes());
  ASSERT_EQ(std::memcmp(ref.getData(), tensor2.getPointer(), ref.getSize()), 0);
}

TEST(Tensor, DeepCopy) {
  auto runtime = getRuntime();
  M::Tensor tensor = makeTensor();
  M::Tensor deepCopy = tensor.deepCopy(runtime);
  ASSERT_TRUE(haveSameContents(tensor, deepCopy));
  incrementContents(tensor);
  ASSERT_FALSE(haveSameContents(tensor, deepCopy));
}

TEST(Tensor, Borrow) {
  M::Tensor tensor = makeTensor();
  M::Tensor borrow = tensor.borrow(GML::BufferRef::kExecute);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
  incrementContents(tensor);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
}

TEST(Tensor, FromBuffer) {
  LLCL::Runtime *runtime = getRuntime();
  TensorSpec spec({kN, kM}, DType::getInt(8, /*isSigned=*/false).value());
  GML::BufferRef ref =
      GML::BufferRef::createOwned(runtime, 1, /*alignment=*/{});

  void *data = ref.getData();
  Tensor tensor = Tensor::create(std::move(ref), spec);
  // Tensor takes the ownership from the buffer ref.
  ASSERT_EQ(data, tensor.getPointer());
  ASSERT_EQ(ref.getData(), nullptr);
}
