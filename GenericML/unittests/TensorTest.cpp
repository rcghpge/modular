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

static GML::Tensor makeTensor() {
  M::TensorSpec spec({kN, kM}, *DType::getInt(8, /*isSigned=*/false));
  GML::Tensor result =
      GML::Tensor::createOwned(std::move(spec), getRuntime(), /*alignment=*/{});
  for (size_t i = 0, e = result.getSizeInBytes(); i < e; ++i)
    result.getMutableBuffer<uint8_t>()[i] = static_cast<uint8_t>(i);
  return result;
}

static void incrementContents(GML::Tensor &tensor) {
  for (size_t i = 0; i < tensor.getSizeInBytes(); ++i)
    ++tensor.getMutableBuffer<uint8_t>()[i];
}

static bool haveSameContents(GML::Tensor &lhs, GML::Tensor &rhs) {
  if (lhs.getSizeInBytes() != rhs.getSizeInBytes())
    return false;
  return std::memcmp(lhs.getBuffer(), rhs.getBuffer(), lhs.getSizeInBytes()) ==
         0;
}

TEST(Tensor, ReleaseBufferRef) {
  auto runtime = getRuntime();
  GML::Tensor tensor = makeTensor();
  GML::Tensor tensor2 = tensor.deepCopy(runtime);
  GML::BufferRef ref = tensor.releaseBufferRef();
  ASSERT_TRUE(tensor.getBuffer() == nullptr);
  ASSERT_EQ(ref.getSize(), tensor2.getSizeInBytes());
  ASSERT_EQ(std::memcmp(ref.getBuffer(), tensor2.getBuffer(), ref.getSize()),
            0);
}

TEST(Tensor, DeepCopy) {
  auto runtime = getRuntime();
  GML::Tensor tensor = makeTensor();
  GML::Tensor deepCopy = tensor.deepCopy(runtime);
  ASSERT_TRUE(haveSameContents(tensor, deepCopy));
  incrementContents(tensor);
  ASSERT_FALSE(haveSameContents(tensor, deepCopy));
}

TEST(Tensor, Borrow) {
  GML::Tensor tensor = makeTensor();
  GML::Tensor borrow = tensor.borrow(GML::BufferRef::kExecute);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
  incrementContents(tensor);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
}

TEST(Tensor, FromBuffer) {
  LLCL::Runtime *runtime = getRuntime();
  TensorSpec spec({kN, kM}, *DType::getInt(8, /*isSigned=*/false));
  GML::BufferRef ref =
      GML::BufferRef::createOwned(runtime, kN * kM, /*alignment=*/{});

  const void *data = ref.getBuffer();
  GML::Tensor tensor(std::move(ref), std::move(spec));
  // Tensor takes the ownership from the buffer ref.
  ASSERT_EQ(data, tensor.getBuffer());
  ASSERT_EQ(ref.getBuffer(), nullptr);
}
