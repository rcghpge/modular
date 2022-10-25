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

static LLCL::Runtime *GetRuntime() {
  static LLCL::Runtime *runtime = []() {
    std::unique_ptr<Allocator> allocator =
        LLCL::createProfilingAllocator(LLCL::createMallocAllocator());
    std::unique_ptr<LLCL::WorkQueue> workQueue =
        LLCL::createSingleThreadWorkQueue();
    return new LLCL::Runtime(std::move(allocator), std::move(workQueue), "");
  }();
  return runtime;
}

constexpr size_t kN = 4;
constexpr size_t kM = 6;

static Tensor MakeTensor() {
  M::TensorSpec spec({kN, kM}, DType::getInt(8, /*isSigned=*/false).value());
  M::Tensor result = Tensor::getUninit(std::move(spec), GetRuntime());
  for (size_t i = 0, e = result.getSizeInBytes(); i < e; ++i)
    static_cast<uint8_t *>(result.getPointer())[i] = static_cast<uint8_t>(i);
  return result;
}

static void IncrementContents(Tensor &tensor) {
  for (size_t i = 0; i < tensor.getSizeInBytes(); ++i)
    ++static_cast<uint8_t *>(tensor.getPointer())[i];
}

static bool HaveSameContents(Tensor &lhs, Tensor &rhs) {
  if (lhs.getSizeInBytes() != rhs.getSizeInBytes())
    return false;
  return std::memcmp(lhs.getPointer(), rhs.getPointer(),
                     lhs.getSizeInBytes()) == 0;
}

TEST(Tensor, ShallowCopy) {
  M::Tensor tensor = MakeTensor();
  M::Tensor shallowCopy = tensor.shallowCopy();
  ASSERT_TRUE(HaveSameContents(tensor, shallowCopy));
  IncrementContents(tensor);
  ASSERT_TRUE(HaveSameContents(tensor, shallowCopy));
}

TEST(Tensor, DeepCopy) {
  M::Tensor tensor = MakeTensor();
  M::Tensor deepCopy = tensor.deepCopy();
  ASSERT_TRUE(HaveSameContents(tensor, deepCopy));
  IncrementContents(tensor);
  ASSERT_FALSE(HaveSameContents(tensor, deepCopy));
}
