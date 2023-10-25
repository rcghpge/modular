//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/ModexBenchmark/InputGenerator.h"
#include "GenericML/ModexBenchmark/InputGenerationSchema.h"
#include "gtest/gtest.h"
#include <limits>

using namespace M;
using namespace M::ModexBench;

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
  void reportError(Twine errorDetail) override;

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
void TestInputSchemaErrorReporter::reportError(Twine errorDetail) {
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

TEST(UniformInputRandomDistribution, F32Default) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  ASSERT_TRUE(errorReporter.reportedNoErrors());
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

TEST(UniformInputRandomDistribution, F32Explicit) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "3.5";
  spec.max = "20";
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  ASSERT_TRUE(errorReporter.reportedNoErrors());
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

TEST(UniformInputRandomDistribution, F32ParseError) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  spec.min = "sea urchin";
  auto dist =
      createUniformInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  std::vector<ReportedError> expectedErrors{
      {{"min"}, "Unable to parse 'sea urchin' as f32 value"}};
  EXPECT_EQ(expectedErrors, errorReporter.errors);
}

TEST(UniformInputRandomDistribution, F32InvertedError) {
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
