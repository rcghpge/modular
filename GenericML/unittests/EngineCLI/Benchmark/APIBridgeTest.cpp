//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/APIBridge.h"
#include "Support/ML/TensorSpec.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "modular/cpp/aliases.h"
#include "gtest/gtest.h"

using namespace M;
using namespace M::BenchmarkCLI;

static constexpr auto kDynamic = mlir::ShapedType::kDynamic;

namespace {

struct DTypeMapping {
  DType m;
  M_Dtype capi;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DTypeMapping &mapping) {
  return os << "{" << mapping.m << ", " << mapping.capi << "}";
}

} // namespace

static const DTypeMapping kDTypeMapping[] = {
    {DType::invalid, M_UNKNOWN}, {DType::si8, M_INT8},
    {DType::si16, M_INT16},      {DType::si32, M_INT32},
    {DType::si64, M_INT64},      {DType::ui8, M_UINT8},
    {DType::ui16, M_UINT16},     {DType::ui32, M_UINT32},
    {DType::ui64, M_UINT64},     {DType::f16, M_FLOAT16},
    {DType::f32, M_FLOAT32},     {DType::f64, M_FLOAT64},
    {DType::kBool, M_BOOL},
};

TEST(APIBridge, dtypeToCAPI) {
  for (const auto &mapping : kDTypeMapping)
    EXPECT_EQ(mapping.capi, dtypeToCAPI(mapping.m)) << mapping;
  // In addition to known mappings, try unknown.
  EXPECT_EQ(M_UNKNOWN, dtypeToCAPI(DType::getComplex(DType::f32)));
}

TEST(APIBridge, capiToDType) {
  for (const auto &mapping : kDTypeMapping)
    EXPECT_EQ(mapping.m, capiToDType(mapping.capi)) << mapping;
  // In addition to known mappings, try unknown.
  EXPECT_EQ(DType(DType::invalid), capiToDType(static_cast<M_Dtype>(99999)));
}

TEST(APIBridge, tensorSpecToCAPIBasic) {
  TensorSpec spec({1, 2, 3}, DType::f32);
  TensorSpecPtr cSpec = tensorSpecToCAPI("example", spec);
  ASSERT_NE(cSpec, nullptr);
  EXPECT_STREQ("example", M_getName(cSpec.get()));
  EXPECT_EQ(M_FLOAT32, M_getDtype(cSpec.get()));
  EXPECT_EQ(size_t{3}, M_getRank(cSpec.get()));
  EXPECT_EQ(int64_t{1}, M_getDimAt(cSpec.get(), 0));
  EXPECT_EQ(int64_t{2}, M_getDimAt(cSpec.get(), 1));
  EXPECT_EQ(int64_t{3}, M_getDimAt(cSpec.get(), 2));
}

// Disabled because there is currently no documented way to create an
// M_TensorSpec containing undefined dimensions through the C API (#26301)
TEST(APIBridge, DISABLED_tensorSpecToCAPIUndefinedDims) {
  // According to TensorSpec docs, any negative value represents an undefined
  // dimension.
  TensorSpec spec({-1, 2, -3}, DType::f32);
  TensorSpecPtr cSpec = tensorSpecToCAPI("example", spec);
  ASSERT_NE(cSpec, nullptr);
  EXPECT_EQ(M_FLOAT32, M_getDtype(cSpec.get()));
  EXPECT_EQ(size_t{3}, M_getRank(cSpec.get()));
  // However, the C API has a more restrictive requirement that all undefined
  // dimensions are specifically -1.
  EXPECT_EQ(int64_t{-1}, M_getDimAt(cSpec.get(), 0));
  EXPECT_EQ(int64_t{2}, M_getDimAt(cSpec.get(), 1));
  EXPECT_EQ(int64_t{-1}, M_getDimAt(cSpec.get(), 2));
}

// TODO: Test undefined rank once C API supports that (#26162)

TEST(APIBridge, capiToTensorSpecBasic) {
  const int64_t cShape[]{1, 2, 3};
  TensorSpecPtr cSpec(M_newTensorSpec(cShape, 3, M_FLOAT32, "example"),
                      M_freeTensorSpec);
  ASSERT_NE(cSpec, nullptr);
  EXPECT_EQ(TensorSpec({1, 2, 3}, DType::f32), capiToTensorSpec(cSpec.get()));
}

TEST(APIBridge, capiToTensorSpecUndefinedDims) {
  const int64_t cShape[]{-1, 2, -1};
  TensorSpecPtr cSpec(M_newTensorSpec(cShape, 3, M_FLOAT32, "example"),
                      M_freeTensorSpec);
  EXPECT_EQ(TensorSpec({kDynamic, int64_t{2}, kDynamic}, DType::f32),
            capiToTensorSpec(cSpec.get()));
}

// TODO: Test undefined rank once C API supports that (#26162)
