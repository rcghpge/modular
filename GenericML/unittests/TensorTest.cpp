//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
// Some token unit tests for generic Tensor.
//===----------------------------------------------------------------------===//

#include "ArraySupport/Tensor.h"
#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "AsyncRT/Runtime/WorkQueue.h"

#include "gtest/gtest.h"

using namespace M;

static std::unique_ptr<LLCL::Runtime> getRuntime() {
  return LLCL::createUniqueRuntime(LLCL::RuntimeOptions().forDebug());
}

constexpr size_t kN = 4;
constexpr size_t kM = 6;

static Tensor makeTensor(LLCL::Runtime &runtime) {
  M::TensorSpec spec({kN, kM}, *DType::getInt(8, /*isSigned=*/false));
  Tensor result =
      Tensor::createOwned(std::move(spec), runtime, /*alignment=*/{});
  for (size_t i = 0, e = result.getSizeInBytes(); i < e; ++i)
    result.getMutableBuffer<uint8_t>()[i] = static_cast<uint8_t>(i);
  return result;
}

static void incrementContents(Tensor &tensor) {
  for (size_t i = 0; i < tensor.getSizeInBytes(); ++i)
    ++tensor.getMutableBuffer<uint8_t>()[i];
}

static bool haveSameContents(Tensor &lhs, Tensor &rhs) {
  if (lhs.getSizeInBytes() != rhs.getSizeInBytes())
    return false;
  return std::memcmp(lhs.getBuffer(), rhs.getBuffer(), lhs.getSizeInBytes()) ==
         0;
}

TEST(Tensor, Borrow) {
  auto runtime = getRuntime();
  Tensor tensor = makeTensor(*runtime);
  Tensor borrow = tensor.borrow(TensorBufferRef::kExecute);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
  incrementContents(tensor);
  ASSERT_TRUE(haveSameContents(tensor, borrow));
}

TEST(Tensor, FromBuffer) {
  auto runtime = getRuntime();
  TensorSpec spec({kN, kM}, *DType::getInt(8, /*isSigned=*/false));
  TensorBufferRef ref = TensorBufferRef::createOwned(*runtime, kN * kM,
                                                     /*alignment=*/{});

  const void *data = ref.getBuffer();
  Tensor tensor(std::move(ref), std::move(spec));
  // Tensor takes the ownership from the buffer ref.
  ASSERT_EQ(data, tensor.getBuffer());
  ASSERT_EQ(ref.getBuffer(), nullptr);
}

// bitWidthT should be a type of the same bit-width as dtype
template <typename bitWidthT>
static void checkPrintTensor(LLCL::Runtime &runtime,
                             std::initializer_list<int64_t> shape,
                             const DType &dtype, SmallVector<bitWidthT> &&data,
                             const std::string &checkString) {
  // getWidthInBits() returns -1 if unk.
  ASSERT_GE(dtype.getWidthInBits(), 0);
  ASSERT_EQ(sizeof(bitWidthT) * CHAR_BIT,
            static_cast<size_t>(dtype.getWidthInBits()));

  M::TensorSpec spec(shape, dtype);
  Tensor tensor =
      Tensor::createOwned(std::move(spec), runtime, /*alignment=*/{});
  ASSERT_EQ(tensor.getSizeInBytes(), data.size() * sizeof(bitWidthT));
  memcpy((void *)tensor.getMutableBuffer(), (void *)data.data(),
         sizeof(bitWidthT) * data.size());

  testing::internal::CaptureStdout();
  ErrorOrSuccess err = tensor.print(ResultOutputStyle::kCompact);
  ASSERT_FALSE(err.isError()) << err.getError();

  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(output, checkString);
}

/// Test GML::printTensorCompact
TEST(Tensor, printTensor) {
  auto runtime = getRuntime();
  checkPrintTensor<uint8_t>(*runtime, {}, DType::ui8, {0},
                            "tensor([[0]], dtype=ui8, shape=[])\n");
  checkPrintTensor<uint8_t>(*runtime, {1}, DType::ui8, {0},
                            "tensor([[0]], dtype=ui8, shape=[1])\n");
  checkPrintTensor<uint8_t>(*runtime, {3}, DType::ui8, {0, 1, 2},
                            "tensor([[0, 1, 2]], dtype=ui8, shape=[3])\n");
  checkPrintTensor<uint8_t>(*runtime, {5, 2}, DType::ui8,
                            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                            "tensor([[0, 1],\n[2, 3],\n[4, 5],\n[6, 7],\n[8, "
                            "9]], dtype=ui8, shape=[5,2])\n");

  checkPrintTensor<int32_t>(*runtime, {}, DType::si32, {0},
                            "tensor([[0]], dtype=si32, shape=[])\n");
  checkPrintTensor<int32_t>(*runtime, {1}, DType::si32, {0},
                            "tensor([[0]], dtype=si32, shape=[1])\n");
  checkPrintTensor<int32_t>(*runtime, {3}, DType::si32, {0, 1, 2},
                            "tensor([[0, 1, 2]], dtype=si32, shape=[3])\n");
  checkPrintTensor<int32_t>(*runtime, {5, 2}, DType::si32,
                            {0, -1, 2, 3, 4, 5, 6, 7, 8, 9},
                            "tensor([[0, -1],\n[2, 3],\n[4, 5],\n[6, 7],\n[8, "
                            "9]], dtype=si32, shape=[5,2])\n");

  checkPrintTensor<float>(*runtime, {}, DType::f32, {0},
                          "tensor([[0.0000]], dtype=f32, shape=[])\n");
  checkPrintTensor<float>(*runtime, {1}, DType::f32, {0},
                          "tensor([[0.0000]], dtype=f32, shape=[1])\n");
  checkPrintTensor<float>(
      *runtime, {3}, DType::f32, {0, 1, 2},
      "tensor([[0.0000, 1.0000, 2.0000]], dtype=f32, shape=[3])\n");
  checkPrintTensor<float>(
      *runtime, {5, 2}, DType::f32, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9.123},
      "tensor([[0.0000, 1.0000],\n[2.0000, 3.0000],\n[4.0000, "
      "5.0000],\n[6.0000, 7.0000],\n[8.0000, "
      "9.1230]], dtype=f32, shape=[5,2])\n");

  // float16
  checkPrintTensor<uint16_t>(
      *runtime, {3}, DType::f16, {0x4248, 0xDD10, 0x4400},
      "tensor([[3.1406, -324.0000, 4.0000]], dtype=f16, shape=[3])\n");

  // bfloat16
  checkPrintTensor<uint16_t>(
      *runtime, {3}, DType::bf16, {0x4049, 0xC467, 0x4080},
      "tensor([[3.1406, -924.0000, 4.0000]], dtype=bf16, shape=[3])\n");
}
