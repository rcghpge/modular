//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Benchmark/InputGenerator.h"
#include "SDK/EngineCLI/Benchmark/Tensor.h"
#include "SDK/EngineCLI/Common/InputGenerationSchema.h"
#include "max/c/tensor.h"
#include "max/cpp/aliases.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <limits>

using namespace M;
using namespace M::BenchmarkCLI;

// We do not expose any CLI tool to test this with llvm-lit instead, so dig
// around inside the source tree so we can do this from a unit test.  LLVM has
// a utility to do this, getInputFileDirectory, but its use is discouraged (and
// requires a different CMake function to be used to add the unit test
// instead), so we dig around like this instead (sorry).
static std::string getTestInputPath(StringRef filename) {
  return (std::filesystem::path(std::getenv("MODULAR_PATH")) / "SDK" /
          "unittests" / "EngineCLI" / "Benchmark" / "Inputs" /
          std::string_view(filename))
      .string();
}

namespace {

struct ReportedError {
  ReportedError(std::vector<std::string> scope, std::string detail)
      : scope{std::move(scope)}, detail{std::move(detail)} {}

  std::vector<std::string> scope;
  std::string detail;
};

bool operator==(const ReportedError &a, const ReportedError &b) {
  return a.scope == b.scope && a.detail == b.detail;
}
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const ReportedError &error) {
  os << "{{";
  llvm::interleaveComma(error.scope, os,
                        [&](auto elem) { os << '"' << elem << '"'; });
  os << "}, \"" << error.detail << "\"}";
  return os;
}

class TestInputSchemaErrorReporter final : public InputSchemaErrorReporter {
public:
  TestInputSchemaErrorReporter();

  void enterScope(StringRef scopeName) override;
  void exitScope(StringRef scopeName) override;
  void reportError(const Twine &errorDetail) override;

  testing::AssertionResult reportedNoErrors() const;

  std::vector<std::string> currentScope;
  std::vector<ReportedError> errors;
};

} // namespace

TestInputSchemaErrorReporter::TestInputSchemaErrorReporter() {}

void TestInputSchemaErrorReporter::enterScope(StringRef scopeName) {
  currentScope.push_back(std::string(scopeName));
}
void TestInputSchemaErrorReporter::exitScope(StringRef scopeName) {
  ASSERT_FALSE(currentScope.empty());
  EXPECT_EQ(currentScope.back(), scopeName);
  currentScope.pop_back();
}
void TestInputSchemaErrorReporter::reportError(const Twine &errorDetail) {
  errors.emplace_back(currentScope, errorDetail.str());
}

testing::AssertionResult
TestInputSchemaErrorReporter::reportedNoErrors() const {
  if (errors.empty())
    return testing::AssertionSuccess();
  return testing::AssertionFailure()
         << "reported " << errors.size()
         << " errors, including: " << errors.front();
}

TEST(UniformInputRandomDistribution, f32Default) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<float, 500> samples;
  std::fill(samples.begin(), samples.end(),
            std::numeric_limits<float>::quiet_NaN());
  dist->fill(samples.data(), 500);
  int lt_neg_1 = 0;
  int lt_0 = 0;
  int gt_1 = 0;
  for (float sample : samples) {
    if (sample < -1.0f)
      ++lt_neg_1;
    if (sample < 0.0f)
      ++lt_0;
    if (sample > 1.0f)
      ++gt_1;
  }
  EXPECT_EQ(0, lt_neg_1) << "Unexpectedly generated values less than -1";
  EXPECT_EQ(0, gt_1) << "Unexpectedly generated values greater than 1";
  // This one is technically non-deterministic, but it should only fail with a
  // probability of 2^-500, which is almost never going to occur if the code is
  // correct.
  EXPECT_GT(lt_0, 0)
      << "Should have generated at least a few values less than 0";
}

TEST(UniformInputRandomDistribution, f32Explicit) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "3.5";
  spec.max = "20";
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<float, 500> samples;
  std::fill(samples.begin(), samples.end(),
            std::numeric_limits<float>::quiet_NaN());
  dist->fill(samples.data(), 500);
  int lt_3p5 = 0;
  int lt_11p75 = 0;
  int gt_11p75 = 0;
  int gt_20 = 0;
  for (float sample : samples) {
    if (sample < -3.5f)
      ++lt_3p5;
    if (sample < 11.75f)
      ++lt_11p75;
    if (sample > 11.75f)
      ++gt_11p75;
    if (sample > 20.0f)
      ++gt_20;
  }
  EXPECT_EQ(0, lt_3p5) << "Unexpectedly generated values less than 3.5";
  EXPECT_EQ(0, gt_20) << "Unexpectedly generated values greater than 20";
  // This one is technically non-deterministic, but it should only fail with a
  // probability of 2^-500, which is almost never going to occur if the code is
  // correct.
  EXPECT_GT(lt_11p75, 0)
      << "Should have generated at least a few values less than 11.75";
  EXPECT_GT(gt_11p75, 0)
      << "Should have generated at least a few values greater than 11.75";
}

TEST(UniformInputRandomDistribution, f32ParseError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "sea urchin";
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"min"}, "Unable to parse 'sea urchin' as f32 value"}}),
            errorReporter.errors);
}

TEST(UniformInputRandomDistribution, f32InvertedError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "5";
  spec.max = "3";
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{{{}, "'min' is greater than 'max'"}}),
            errorReporter.errors);
}

TEST(UniformInputRandomDistribution, f64Default) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::f64, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<double, 500> samples;
  std::fill(samples.begin(), samples.end(),
            std::numeric_limits<double>::quiet_NaN());
  dist->fill(samples.data(), 500);
  int lt_neg_1 = 0;
  int lt_0 = 0;
  int gt_1 = 0;
  for (double sample : samples) {
    if (sample < -1.0)
      ++lt_neg_1;
    if (sample < 0.0)
      ++lt_0;
    if (sample > 1.0)
      ++gt_1;
  }
  EXPECT_EQ(0, lt_neg_1) << "Unexpectedly generated values less than -1";
  EXPECT_EQ(0, gt_1) << "Unexpectedly generated values greater than 1";
  // This one is technically non-deterministic, but it should only fail with a
  // probability of 2^-500, which is almost never going to occur if the code is
  // correct.
  EXPECT_GT(lt_0, 0)
      << "Should have generated at least a few values less than 0";
}

TEST(UniformInputRandomDistribution, f16Default) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::f16, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<uint16_t, 500> samples;
  std::fill(samples.begin(), samples.end(), 0xFFFFU);
  dist->fill(samples.data(), 500);
  int lt_neg_1 = 0;
  int lt_0 = 0;
  int gt_1 = 0;
  for (uint16_t sampleAsUInt : samples) {
    double sample =
        APFloat(APFloat::IEEEhalf(), APInt(16, sampleAsUInt)).convertToDouble();
    if (sample < -1.0)
      ++lt_neg_1;
    if (sample < 0.0)
      ++lt_0;
    if (sample > 1.0)
      ++gt_1;
  }
  EXPECT_EQ(0, lt_neg_1) << "Unexpectedly generated values less than -1";
  EXPECT_EQ(0, gt_1) << "Unexpectedly generated values greater than 1";
  // This one is technically non-deterministic, but it should only fail with a
  // probability of 2^-500, which is almost never going to occur if the code is
  // correct.
  EXPECT_GT(lt_0, 0)
      << "Should have generated at least a few values less than 0";
}

TEST(UniformInputRandomDistribution, si32Default) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<int32_t, 500> samples;
  std::fill(samples.begin(), samples.end(), 12345);
  dist->fill(samples.data(), 500);
  int lt_0 = 0;
  int gt_0 = 0;
  for (int32_t sample : samples) {
    if (sample < 0)
      ++lt_0;
    if (sample > 0)
      ++gt_0;
  }
  EXPECT_GT(lt_0, 0)
      << "Should have generated at least a few values less than 0";
  EXPECT_GT(gt_0, 0)
      << "Should have generated at least a few values greater than 0";
}

TEST(UniformInputRandomDistribution, si32Explicit) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "3";
  spec.max = "20";
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<int32_t, 500> samples;
  std::fill(samples.begin(), samples.end(), 12345);
  dist->fill(samples.data(), 500);
  int lt_3 = 0;
  int lt_12 = 0;
  int ge_12 = 0;
  int gt_20 = 0;
  for (int32_t sample : samples) {
    if (sample < -3)
      ++lt_3;
    if (sample < 12)
      ++lt_12;
    if (sample >= 12)
      ++ge_12;
    if (sample > 20)
      ++gt_20;
  }
  EXPECT_EQ(0, lt_3) << "Unexpectedly generated values less than 3";
  EXPECT_EQ(0, gt_20) << "Unexpectedly generated values greater than 20";
  EXPECT_GT(lt_12, 0)
      << "Should have generated at least a few values less than 12";
  EXPECT_GT(ge_12, 0) << "Should have generated at least a few values >= 12";
}

TEST(UniformInputRandomDistribution, si32ExplicitInclusive) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "3";
  spec.max = "4";
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<int32_t, 500> samples;
  std::fill(samples.begin(), samples.end(), 12345);
  dist->fill(samples.data(), 500);
  int eq_3 = 0;
  int eq_4 = 0;
  for (int32_t sample : samples) {
    if (sample == 3)
      ++eq_3;
    if (sample == 4)
      ++eq_4;
  }
  EXPECT_GT(eq_3, 0) << "Should have generated at least a few '3' values";
  EXPECT_GT(eq_4, 0) << "Should have generated at least a few '4' values";
}

TEST(UniformInputRandomDistribution, si32ParseError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "sea urchin";
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"min"}, "Unable to parse 'sea urchin' as si32 value"}}),
            errorReporter.errors);
}

TEST(UniformInputRandomDistribution, si32InvertedError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "5";
  spec.max = "3";
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{{{}, "'min' is greater than 'max'"}}),
            errorReporter.errors);
}

namespace {
using IntegerTypes = testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t,
                                    uint32_t, int64_t, uint64_t>;
}
template <typename T>
class UniformInputRandomDistributionInteger : public testing::Test {};
TYPED_TEST_SUITE(UniformInputRandomDistributionInteger, IntegerTypes);

TYPED_TEST(UniformInputRandomDistributionInteger, Default) {
  constexpr DType dtype = DTypeForCXXType<TypeParam>::kind;
  static_assert(dtype != DType::invalid, "No DType for TypeParam");
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist = createUniformInputRandomDistribution(dtype, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(dist, nullptr);
  std::array<TypeParam, 500> samples;
  std::fill(samples.begin(), samples.end(), 0);
  dist->fill(samples.data(), 500);
}

TEST(UniformInputRandomDistribution, unsupportedType) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::f8, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{{{}, "Unsupported dtype f8"}}),
            errorReporter.errors);
}

TEST(NormalInputRandomDistribution, unimplemented) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomNormal spec;
  auto dist =
      createNormalInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Normal distribution not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputRandomDistribution, none) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  // Note: The user should never experience this case in practice due to the
  // defaulting in InputGenerationDefaults.
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{}, "No random distribution specified"}}),
      errorReporter.errors);
}

TEST(InputRandomDistribution, uniform) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_NE(nullptr, dist);
  // TODO: Would be good to test that this is a uniform distribution as
  // expected.
}

TEST(InputRandomDistribution, normal) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.normal = InputGenerationRandomNormal{};
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"normal"}, "Normal distribution not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputRandomDistribution, multiple) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  spec.normal = InputGenerationRandomNormal{};
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"normal"}, "Normal distribution not yet implemented"},
                {{}, "Only one random distribution may be specified"}}),
            errorReporter.errors);
}

namespace {
class TestRandomDistribution final : public InputRandomDistribution {
public:
  TestRandomDistribution(
      DType dtype, std::function<void(void *out, size_t numElements)> fillFn)
      : InputRandomDistribution(dtype), fillFn(std::move(fillFn)) {}

  void fill(void *out, size_t numElements) const override {
    fillFn(out, numElements);
  }

private:
  std::function<void(void *out, size_t numElements)> fillFn;
};
} // namespace

TEST(RandomInputDataGenerator, fromDistribution) {
  int numCalls = 0;
  void *lastFillCallOut = nullptr;
  size_t lastFillCallNumElements = 0;
  auto gen =
      createRandomInputDataGenerator(std::make_unique<TestRandomDistribution>(
          DType::f32, [&](void *out, size_t numElements) {
            ++numCalls;
            lastFillCallOut = out;
            lastFillCallNumElements = numElements;
          }));
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f32), tensorOr->getEltType());
  EXPECT_EQ(1, numCalls);
  EXPECT_EQ(tensorOr->getBuffer(), lastFillCallOut);
  EXPECT_EQ(size_t{1 * 2 * 3}, lastFillCallNumElements);
}

TEST(RandomInputDataGenerator, basic) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto gen = createRandomInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f32), tensorOr->getEltType());
}

TEST(RandomInputDataGenerator, noShape) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto gen = createRandomInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape(kDynamicallyRanked));
  ASSERT_TRUE(tensorOr.isError());
  // TODO: Ideally we would be able to detect this at generator creation time,
  // not at generation time.  But that might be tricky, so we can do that
  // later.
  EXPECT_STREQ("Random input requires fully defined input shape",
               tensorOr.getError());
}

TEST(RandomInputDataGenerator, constructionError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  spec.uniform->min = "cheesecake";
  auto gen = createRandomInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{
          {{"uniform", "min"}, "Unable to parse 'cheesecake' as f32 value"}}),
      errorReporter.errors);
}

namespace {
using NativeFloatTypes = testing::Types<float, double>;
}
template <typename T>
class ConstantInputDataGeneratorInteger : public testing::Test {};
TYPED_TEST_SUITE(ConstantInputDataGeneratorInteger, IntegerTypes);
template <typename T>
class ConstantInputDataGeneratorNativeFloat : public testing::Test {};
TYPED_TEST_SUITE(ConstantInputDataGeneratorNativeFloat, NativeFloatTypes);
class ConstantInputDataGeneratorError : public testing::TestWithParam<DType> {};
INSTANTIATE_TEST_SUITE_P(ConstantInputDataGeneratorError,
                         ConstantInputDataGeneratorError,
                         testing::Values(DType::si8, DType::ui8, DType::si16,
                                         DType::ui16, DType::si32, DType::ui32,
                                         DType::si64, DType::ui64, DType::f16,
                                         DType::f32, DType::f64, DType::kBool));

TYPED_TEST(ConstantInputDataGeneratorInteger, basic) {
  constexpr DType dtype = DTypeForCXXType<TypeParam>::kind;
  static_assert(dtype != DType::invalid, "No DType for TypeParam");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createConstantInputDataGenerator(dtype, "123", errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(dtype, tensorOr->getEltType());
  const TypeParam *data = static_cast<const TypeParam *>(tensorOr->getBuffer());
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(TypeParam{123}, data[i]) << "At index " << i;
}

TYPED_TEST(ConstantInputDataGeneratorNativeFloat, basic) {
  constexpr DType dtype = DTypeForCXXType<TypeParam>::kind;
  static_assert(dtype != DType::invalid, "No DType for TypeParam");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createConstantInputDataGenerator(dtype, "-1.23", errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(dtype, tensorOr->getEltType());
  const TypeParam *data = static_cast<const TypeParam *>(tensorOr->getBuffer());
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(static_cast<TypeParam>(-1.23), data[i]) << "At index " << i;
}

TEST_P(ConstantInputDataGeneratorError, error) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen =
      createConstantInputDataGenerator(GetParam(), "beetle", errorReporter);
  EXPECT_EQ(nullptr, gen);
  DType dtypeInMessage = GetParam();
  // Implementation detail leaks through for f16, but it probably doesn't
  // matter.
  if (dtypeInMessage == DType::f16)
    dtypeInMessage = DType::f32;
  EXPECT_EQ(
      (std::vector<ReportedError>{{{},
                                   (Twine("Unable to parse 'beetle' as ") +
                                    dtypeInMessage.getAsString() + " value")
                                       .str()}}),
      errorReporter.errors);
}

TEST(ConstantInputDataGeneratorOther, float16) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen =
      createConstantInputDataGenerator(DType::f16, "-1.23", errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f16), tensorOr->getEltType());
  const uint16_t *data = static_cast<const uint16_t *>(tensorOr->getBuffer());
  // -1.23 as a float16's byte representation is equivalent to uint16_t 48364.
  // Verify in Python with: np.array([-1.23], dtype=np.float16).view(np.uint16)
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(48364, data[i]) << "At index " << i;
}

TEST(ConstantInputDataGeneratorOther, boolean) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen =
      createConstantInputDataGenerator(DType::kBool, "true", errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::kBool), tensorOr->getEltType());
  const bool *data = static_cast<const bool *>(tensorOr->getBuffer());
  for (int i = 0; i < 6; ++i)
    EXPECT_TRUE(data[i]) << "At index " << i;
}

TEST(ConstantInputDataGeneratorOther, unimplemented) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createConstantInputDataGenerator(DType::getComplex(DType::f32),
                                              "123", errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{{{},
                                         "Constant input generation for dtype "
                                         "'complex<f32>' not supported"}}),
            errorReporter.errors);
}

TEST(FileInputDataGenerator, basic) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("basic.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape(kDynamicallyRanked));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  BenchmarkCLI::Tensor tensor = tensorOr.takeValue();
  EXPECT_EQ(TensorSpec({2, 3}, DType::si64), tensor.getSpec());
  auto data = static_cast<const int64_t *>(tensor.getBuffer());
  EXPECT_EQ(0, data[0]);
  EXPECT_EQ(1, data[1]);
  EXPECT_EQ(2, data[2]);
  EXPECT_EQ(3, data[3]);
  EXPECT_EQ(4, data[4]);
  EXPECT_EQ(5, data[5]);
}

TEST(FileInputDataGenerator, specifiedShape) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("basic.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  tensorOr = gen->generate(
      /*queryIndex=*/0, TensorShape({mlir::ShapedType::kDynamic, int64_t{3}}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  tensorOr = gen->generate(
      /*queryIndex=*/0, TensorShape({int64_t{2}, mlir::ShapedType::kDynamic}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  tensorOr = gen->generate(/*queryIndex=*/0, TensorShape(kDynamicallyRanked));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
}

TEST(FileInputDataGenerator, multiQuery) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("multable.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(3, gen->getNumQueries());
  for (size_t i = 0; i < 3; ++i) {
    ErrorOr<BenchmarkCLI::Tensor> tensorOr =
        gen->generate(/*queryIndex=*/i, TensorShape({3}));
    ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
    EXPECT_EQ(TensorSpec({3}, DType::si64), tensorOr->getSpec());
    auto data = static_cast<const int64_t *>(tensorOr->getBuffer());
    EXPECT_EQ(int64_t{0}, data[0]);
    EXPECT_EQ(int64_t(i), data[1]);
    EXPECT_EQ(int64_t(2 * i), data[2]);
  }
}

TEST(FileInputDataGenerator, dtypeMismatch) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("basic.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{},
                 "Data type of file input 'si64' inconsistent with previously "
                 "declared dtype 'f32'"}}),
            errorReporter.errors);
}

TEST(FileInputDataGenerator, shapeMismatch) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("basic.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({5, 6}));
  ASSERT_TRUE(tensorOr.isError());
  EXPECT_STREQ("Declared shape 5x6 incompatible with file intrinsic shape 2x3: "
               "Specified shape 2x3 has dimension which doesn't match the "
               "required shape 5x6 at index 0.",
               tensorOr.getError());
}

TEST(FileInputDataGenerator, pathNotExist) {
  InputGenerationFile spec;
  spec.path = "path-that-does-not-exist";
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  using namespace testing;
  EXPECT_THAT(
      errorReporter.errors,
      ElementsAre(AllOf(Field("scope", &ReportedError::scope,
                              Eq(std::vector<std::string>{"path"})),
                        Field("detail", &ReportedError::detail,
                              StartsWith("Unable to open input file: ")))));
}

TEST(FileInputDataGenerator, formatDetectionFail) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("empty.dat");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"format"},
                 "Format not specified and unable to automatically detect"}}),
            errorReporter.errors);
}

TEST(FileInputDataGenerator, badFormat) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("basic.npy");
  spec.format = "bogus";
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{{{"format"}, "Unknown file format"}}),
            errorReporter.errors);
}

TEST(FileInputDataGenerator, npyReadError) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("empty.dat");
  spec.format = "npy";
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{},
                 "Unable to read NPY file: Stream Error: The stream is too "
                 "short to perform the requested operation."}}),
            errorReporter.errors);
}

TEST(FileInputDataGenerator, zeroRank) {
  InputGenerationFile spec;
  spec.path = getTestInputPath("zero-rank.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createFileInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Tensors sourced from input file must have rank > 0"}}),
            errorReporter.errors);
}

TEST(InputDataGenerator, none) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  // Note: The user should never experience this case in practice due to the
  // defaulting in InputGenerationDefaults.
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{}, "No data generation source specified"}}),
      errorReporter.errors);
}

TEST(InputDataGenerator, random) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.random = InputGenerationRandom{};
  spec.random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  EXPECT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  // TODO: Check that we got a random input data generator somehow.
}

TEST(InputDataGenerator, randomNoDType) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.random = InputGenerationRandom{};
  spec.random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{
          {{"random"}, "Random input data source requires dtype to be known"}}),
      errorReporter.errors);
}

TEST(InputDataGenerator, constant) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.constant = "123.4";
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f32), tensorOr->getEltType());
  const float *data = static_cast<const float *>(tensorOr->getBuffer());
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(123.4f, data[i]) << "At index " << i;
}

TEST(InputDataGenerator, constantNoDType) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.constant = "123";
  auto gen = createInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"constant"},
                 "Constant input data source requires dtype to be known"}}),
            errorReporter.errors);
}

TEST(InputDataGenerator, file) {
  InputGenerationData spec;
  spec.file = InputGenerationFile{};
  spec.file->path = getTestInputPath("basic.npy");
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createInputDataGenerator(std::nullopt, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(1, gen->getNumQueries());
  ErrorOr<BenchmarkCLI::Tensor> tensorOr =
      gen->generate(/*queryIndex=*/0, TensorShape({2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  BenchmarkCLI::Tensor tensor = tensorOr.takeValue();
  EXPECT_EQ(TensorSpec({2, 3}, DType::si64), tensor.getSpec());
  auto data = static_cast<const int64_t *>(tensor.getBuffer());
  EXPECT_EQ(0, data[0]);
  EXPECT_EQ(1, data[1]);
  EXPECT_EQ(2, data[2]);
  EXPECT_EQ(3, data[3]);
  EXPECT_EQ(4, data[4]);
  EXPECT_EQ(5, data[5]);
}

TEST(InputDataGenerator, multiple) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.random = InputGenerationRandom{};
  spec.random->uniform = InputGenerationRandomUniform{};
  spec.constant = "123";
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Only one data generation source may be specified"}}),
            errorReporter.errors);
}

TEST(InputGenerator, basic) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationInput spec;
  spec.inputName = "example";
  spec.shape = TensorSpec({1, 2, 3}, DType::f32);
  spec.data = InputGenerationData{};
  spec.data->random = InputGenerationRandom{};
  spec.data->random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputGenerator(spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  auto inputOr = gen->generateInput(0);
  ASSERT_FALSE(inputOr.isError()) << inputOr.getError();
  ASSERT_NE(nullptr, *inputOr);
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status.get());
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr ctx(M_newRuntimeContext(config.get(), status.get()),
                        M_freeRuntimeContext);
  ASSERT_NE(nullptr, ctx);
  TensorMapPtr atm(M_newAsyncTensorMap(ctx.get()), M_freeAsyncTensorMap);
  ASSERT_NE(nullptr, atm);
  (*inputOr)->borrowInto(atm.get(), status.get());
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  EXPECT_EQ(size_t{1}, M_getTensorMapSize(atm.get(), status.get()));
  TensorPtr tensor(M_getTensorByNameFrom(atm.get(), "example", status.get()),
                   M_freeTensor);
  ASSERT_NE(nullptr, tensor);
  EXPECT_EQ(size_t{1 * 2 * 3}, M_getTensorNumElements(tensor.get()));
  EXPECT_EQ(M_FLOAT32, M_getTensorType(tensor.get()));
}

TEST(InputGenerator, noName) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationInput spec;
  spec.shape = TensorSpec({1, 2, 3}, DType::f32);
  spec.data = InputGenerationData{};
  spec.data->random = InputGenerationRandom{};
  spec.data->random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputGenerator(spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"input_name"}, "Input name must be specified"}}),
            errorReporter.errors);
}

TEST(InputGenerator, indexNYI) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationInput spec;
  spec.inputIndex = 0;
  spec.shape = TensorSpec({1, 2, 3}, DType::f32);
  spec.data = InputGenerationData{};
  spec.data->random = InputGenerationRandom{};
  spec.data->random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputGenerator(spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"input_index"},
                 "Inputs specified by index are not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputGenerator, nameAndIndex) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationInput spec;
  spec.inputName = "example";
  spec.inputIndex = 0;
  spec.shape = TensorSpec({1, 2, 3}, DType::f32);
  spec.data = InputGenerationData{};
  spec.data->random = InputGenerationRandom{};
  spec.data->random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputGenerator(spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Cannot specify both input name and input index"}}),
            errorReporter.errors);
}

namespace {
class TestGeneratedInput final : public GeneratedInput {
public:
  explicit TestGeneratedInput(
      std::function<void(M_AsyncTensorMap *atm, M_Status *status)> borrowIntoFn)
      : borrowIntoFn(std::move(borrowIntoFn)) {}

  void borrowInto(M_AsyncTensorMap *atm, M_Status *status) const override {
    borrowIntoFn(atm, status);
  }

private:
  std::function<void(M_AsyncTensorMap *atm, M_Status *status)> borrowIntoFn;
};

class TestInputGenerator final : public InputGenerator {
public:
  TestInputGenerator(
      ssize_t numQueries,
      std::function<ErrorOr<std::unique_ptr<GeneratedInput>>(size_t queryIndex)>
          generateInputFn)
      : numQueries(numQueries), generateInputFn(std::move(generateInputFn)) {}

  ssize_t getNumQueries() const override { return numQueries; }
  ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(size_t queryIndex) override {
    return generateInputFn(queryIndex);
  }

private:
  ssize_t numQueries;
  std::function<ErrorOr<std::unique_ptr<GeneratedInput>>(size_t queryIndex)>
      generateInputFn;
};
} // namespace

TEST(QueryGenerator, noInputs) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(
      std::vector<std::unique_ptr<InputGenerator>>{}, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtime(M_newRuntimeContext(config.get(), status.get()),
                            M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtime);
  auto queryOr = gen->generateQuery(runtime.get(), 0);
  EXPECT_FALSE(queryOr.isError()) << queryOr.getError();
  EXPECT_NE(nullptr, *queryOr);
  auto atm = (*queryOr)->getAsyncTensorMap();
  EXPECT_NE(nullptr, atm);
  EXPECT_EQ(size_t{0}, M_getTensorMapSize(atm, status.get()));
}

TEST(QueryGenerator, oneInput) {
  int generateInputCalls = 0;
  int lastGenerateQueryIndex = 0;
  int borrowCalls = 0;
  M_AsyncTensorMap *lastBorrowedATM = nullptr;
  auto inputGen =
      std::make_unique<TestInputGenerator>(-1, [&](size_t queryIndex) {
        ++generateInputCalls;
        lastGenerateQueryIndex = queryIndex;
        return std::make_unique<TestGeneratedInput>(
            [&](M_AsyncTensorMap *atm, M_Status *status) {
              ++borrowCalls;
              lastBorrowedATM = atm;
            });
      });
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::move(inputGen));
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(std::move(inputGens), errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  EXPECT_EQ(0, generateInputCalls);
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtime(M_newRuntimeContext(config.get(), status.get()),
                            M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtime);
  auto queryOr = gen->generateQuery(runtime.get(), 123);
  EXPECT_FALSE(queryOr.isError()) << queryOr.getError();
  EXPECT_NE(nullptr, *queryOr);
  auto atm = (*queryOr)->getAsyncTensorMap();
  EXPECT_NE(nullptr, atm);
  EXPECT_EQ(1, generateInputCalls);
  EXPECT_EQ(123, lastGenerateQueryIndex);
  EXPECT_EQ(1, borrowCalls);
  EXPECT_EQ(atm, lastBorrowedATM);
}

TEST(QueryGenerator, oneInputError) {
  auto inputGen = std::make_unique<TestInputGenerator>(
      -1, [&](size_t queryIndex) { return Error("Tea kettle left on stove"); });
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::move(inputGen));
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(std::move(inputGens), errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtime(M_newRuntimeContext(config.get(), status.get()),
                            M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtime);
  auto queryOr = gen->generateQuery(runtime.get(), 0);
  ASSERT_TRUE(queryOr.isError());
  EXPECT_STREQ("Tea kettle left on stove", queryOr.getError());
}

TEST(QueryGenerator, numQueriesMatch) {
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      1234, [&](size_t queryIndex) { return Error("Not to be called"); }));
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      1234, [&](size_t queryIndex) { return Error("Not to be called"); }));
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(std::move(inputGens), errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(1234, gen->getNumQueries());
}

TEST(QueryGenerator, numQueriesPrecedence) {
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      -1, [&](size_t queryIndex) { return Error("Not to be called"); }));
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      1234, [&](size_t queryIndex) { return Error("Not to be called"); }));
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(std::move(inputGens), errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(1234, gen->getNumQueries());
}

TEST(QueryGenerator, numQueriesMismatch) {
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      1234, [&](size_t queryIndex) { return Error("Not to be called"); }));
  inputGens.emplace_back(std::make_unique<TestInputGenerator>(
      4321, [&](size_t queryIndex) { return Error("Not to be called"); }));
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createQueryGenerator(std::move(inputGens), errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{
          {{"inputs", "1"}, "Query count 4321 inconsistent with other inputs"},
          {{"inputs", "0"},
           "Note: Conflicting query count (1234) came from this input"}}),
      errorReporter.errors);
}

TEST(QueryGeneratorIntegration, basic) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationSchema schema;
  schema.inputs = std::vector<InputGenerationInput>{InputGenerationInput{}};
  schema.inputs->at(0).inputName = "example";
  schema.inputs->at(0).shape = TensorSpec({1, 2, 3}, DType::f32);
  schema.inputs->at(0).data = InputGenerationData{};
  schema.inputs->at(0).data->random = InputGenerationRandom{};
  schema.inputs->at(0).data->random->uniform = InputGenerationRandomUniform{};
  auto gen = createQueryGenerator(schema, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(-1, gen->getNumQueries());
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtime(M_newRuntimeContext(config.get(), status.get()),
                            M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtime);
  auto queryOr = gen->generateQuery(runtime.get(), 0);
  ASSERT_FALSE(queryOr.isError()) << queryOr.getError();
  ASSERT_NE(nullptr, *queryOr);
  auto atm = (*queryOr)->getAsyncTensorMap();
  ASSERT_NE(nullptr, atm);
  EXPECT_EQ(size_t{1}, M_getTensorMapSize(atm, status.get()));
  TensorPtr tensor(M_getTensorByNameFrom(atm, "example", status.get()),
                   M_freeTensor);
  ASSERT_NE(nullptr, tensor);
  EXPECT_EQ(size_t{1 * 2 * 3}, M_getTensorNumElements(tensor.get()));
  EXPECT_EQ(M_FLOAT32, M_getTensorType(tensor.get()));
}

TEST(QueryGeneratorIntegration, numQueries) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationSchema schema;
  schema.inputs = std::vector<InputGenerationInput>{InputGenerationInput{}};
  schema.inputs->at(0).inputName = "example";
  schema.inputs->at(0).data = InputGenerationData{};
  schema.inputs->at(0).data->file = InputGenerationFile{};
  schema.inputs->at(0).data->file->path = getTestInputPath("basic.npy");
  auto gen = createQueryGenerator(schema, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  EXPECT_EQ(1, gen->getNumQueries());
}

TEST(QueryGeneratorIntegration, errorPath) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationSchema schema;
  schema.inputs = std::vector<InputGenerationInput>{InputGenerationInput{}};
  schema.inputs->at(0).inputName = "example";
  schema.inputs->at(0).shape = TensorSpec({1, 2, 3}, DType::f32);
  schema.inputs->at(0).data = InputGenerationData{};
  schema.inputs->at(0).data->random = InputGenerationRandom{};
  schema.inputs->at(0).data->random->uniform = InputGenerationRandomUniform{};
  schema.inputs->at(0).data->random->uniform->min = "steamship";
  auto gen = createQueryGenerator(schema, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"inputs", "0", "data", "random", "uniform", "min"},
                 "Unable to parse 'steamship' as f32 value"}}),
            errorReporter.errors);
}

TEST(QueryGeneratorIntegration, unspecifiedInputs) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationSchema schema;
  auto gen = createQueryGenerator(schema, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{"inputs"}, "Inputs must be specified"}}),
      errorReporter.errors);
}

// TODO: Test for duplicate input names or indices in schema
