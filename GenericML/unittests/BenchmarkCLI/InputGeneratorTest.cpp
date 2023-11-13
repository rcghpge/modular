//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/InputGenerator.h"
#include "GenericML/API/modular/c/tensor.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/BenchmarkCLI/InputGenerationSchema.h"
#include "LLCL/Runtime/Runtime.h"
#include "gtest/gtest.h"
#include <limits>

using namespace M;
using namespace M::BenchmarkCLI;

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

static std::unique_ptr<LLCL::Runtime> createRuntimeForTest() {
  return std::make_unique<LLCL::Runtime>(
      LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
      LLCL::createSingleThreadWorkQueue());
}

TEST(UniformInputRandomDistribution, F32Default) {
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

TEST(UniformInputRandomDistribution, F32Explicit) {
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

TEST(UniformInputRandomDistribution, F32ParseError) {
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

TEST(UniformInputRandomDistribution, NonF32) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomUniform spec;
  auto dist =
      createUniformInputRandomDistribution(DType::si32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{}, "Only f32 supported at this time"}}),
      errorReporter.errors);
}

TEST(NormalInputRandomDistribution, Unimplemented) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandomNormal spec;
  auto dist =
      createNormalInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Normal distribution not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputRandomDistribution, None) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  // TODO: The None case should default to a reasonable distribution for the
  // dtype instead of erroring.
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{}, "No random distribution specified"}}),
      errorReporter.errors);
}

TEST(InputRandomDistribution, Uniform) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_NE(nullptr, dist);
  // TODO: Would be good to test that this is a uniform distribution as
  // expected.
}

TEST(InputRandomDistribution, Normal) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.normal = InputGenerationRandomNormal{};
  auto dist = createInputRandomDistribution(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, dist);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"normal"}, "Normal distribution not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputRandomDistribution, Multiple) {
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

TEST(RandomInputDataGenerator, FromDistribution) {
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
  auto runtime = createRuntimeForTest();
  ErrorOr<GML::Tensor> tensorOr =
      gen->generate(*runtime, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f32), tensorOr->getEltType());
  EXPECT_EQ(1, numCalls);
  EXPECT_EQ(tensorOr->getBuffer(), lastFillCallOut);
  EXPECT_EQ(size_t{1 * 2 * 3}, lastFillCallNumElements);
}

TEST(RandomInputDataGenerator, Basic) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto gen = createRandomInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  auto runtime = createRuntimeForTest();
  ErrorOr<GML::Tensor> tensorOr =
      gen->generate(*runtime, TensorShape({1, 2, 3}));
  ASSERT_FALSE(tensorOr.isError()) << tensorOr.getError();
  EXPECT_EQ(TensorShape({1, 2, 3}), tensorOr->getShape());
  EXPECT_EQ(DType(DType::f32), tensorOr->getEltType());
}

TEST(RandomInputDataGenerator, NoShape) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationRandom spec;
  spec.uniform = InputGenerationRandomUniform{};
  auto gen = createRandomInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  ASSERT_NE(nullptr, gen);
  auto runtime = createRuntimeForTest();
  ErrorOr<GML::Tensor> tensorOr =
      gen->generate(*runtime, TensorShape(kDynamicallyRanked));
  ASSERT_TRUE(tensorOr.isError());
  // TODO: Ideally we would be able to detect this at generator creation time,
  // not at generation time.  But that might be tricky, so we can do that
  // later.
  EXPECT_STREQ("Random input requires fully defined input shape",
               tensorOr.getError());
}

TEST(RandomInputDataGenerator, ConstructionError) {
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

TEST(ConstantInputDataGenerator, Unimplemented) {
  TestInputSchemaErrorReporter errorReporter;
  auto gen = createConstantInputDataGenerator(DType::f32, "123", errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{}, "Constant input generation not yet implemented"}}),
            errorReporter.errors);
}

TEST(InputDataGenerator, None) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  // TODO: The None case should default to a reasonable distribution for the
  // dtype instead of erroring.
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{}, "No data generation source specified"}}),
      errorReporter.errors);
}

TEST(InputDataGenerator, Random) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.random = InputGenerationRandom{};
  spec.random->uniform = InputGenerationRandomUniform{};
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_TRUE(errorReporter.reportedNoErrors());
  EXPECT_NE(nullptr, gen);
  // TODO: Check that we got a random input data generator somehow.
}

TEST(InputDataGenerator, RandomNoDType) {
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

TEST(InputDataGenerator, Constant) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.constant = "123";
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{
          {{"constant"}, "Constant input generation not yet implemented"}}),
      errorReporter.errors);
}

TEST(InputDataGenerator, ConstantNoDType) {
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

TEST(InputDataGenerator, Multiple) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationData spec;
  spec.random = InputGenerationRandom{};
  spec.random->uniform = InputGenerationRandomUniform{};
  spec.constant = "123";
  auto gen = createInputDataGenerator(DType::f32, spec, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ((std::vector<ReportedError>{
                {{"constant"}, "Constant input generation not yet implemented"},
                {{}, "Only one data generation source may be specified"}}),
            errorReporter.errors);
}

TEST(InputGenerator, Basic) {
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
  auto runtime = createRuntimeForTest();
  auto inputOr = gen->generateInput(runtime->getCompactPtr(), 0);
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

TEST(InputGenerator, NoName) {
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

TEST(InputGenerator, IndexNYI) {
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

TEST(InputGenerator, NameAndIndex) {
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
  explicit TestInputGenerator(
      std::function<ErrorOr<std::unique_ptr<GeneratedInput>>(size_t queryIndex)>
          generateInputFn)
      : generateInputFn(std::move(generateInputFn)) {}

  ErrorOr<std::unique_ptr<GeneratedInput>>
  generateInput(LLCL::CompactRuntimePtr runtime, size_t queryIndex) override {
    // TODO: Not forwarding runtime for now, since I would like to get rid of
    // the runtime parameter entirely later.
    return generateInputFn(queryIndex);
  }

private:
  std::function<ErrorOr<std::unique_ptr<GeneratedInput>>(size_t queryIndex)>
      generateInputFn;
};
} // namespace

TEST(QueryGenerator, NoInputs) {
  auto gen = createQueryGenerator({});
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
  auto allocRuntime = createRuntimeForTest();
  auto queryOr =
      gen->generateQuery(allocRuntime->getCompactPtr(), runtime.get(), 0);
  EXPECT_FALSE(queryOr.isError()) << queryOr.getError();
  EXPECT_NE(nullptr, *queryOr);
  auto atm = (*queryOr)->getAsyncTensorMap();
  EXPECT_NE(nullptr, atm);
  EXPECT_EQ(size_t{0}, M_getTensorMapSize(atm, status.get()));
}

TEST(QueryGenerator, OneInput) {
  int generateInputCalls = 0;
  int lastGenerateQueryIndex = 0;
  int borrowCalls = 0;
  M_AsyncTensorMap *lastBorrowedATM = nullptr;
  auto inputGen = std::make_unique<TestInputGenerator>([&](size_t queryIndex) {
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
  auto gen = createQueryGenerator(std::move(inputGens));
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
  auto allocRuntime = createRuntimeForTest();
  auto queryOr =
      gen->generateQuery(allocRuntime->getCompactPtr(), runtime.get(), 123);
  EXPECT_FALSE(queryOr.isError()) << queryOr.getError();
  EXPECT_NE(nullptr, *queryOr);
  auto atm = (*queryOr)->getAsyncTensorMap();
  EXPECT_NE(nullptr, atm);
  EXPECT_EQ(1, generateInputCalls);
  EXPECT_EQ(123, lastGenerateQueryIndex);
  EXPECT_EQ(1, borrowCalls);
  EXPECT_EQ(atm, lastBorrowedATM);
}

TEST(QueryGenerator, OneInputError) {
  auto inputGen = std::make_unique<TestInputGenerator>(
      [&](size_t queryIndex) { return Error("Tea kettle left on stove"); });
  std::vector<std::unique_ptr<InputGenerator>> inputGens;
  inputGens.emplace_back(std::move(inputGen));
  auto gen = createQueryGenerator(std::move(inputGens));
  ASSERT_NE(nullptr, gen);
  StatusPtr status(M_newStatus(), M_freeStatus);
  ASSERT_NE(nullptr, status);
  RuntimeConfigPtr config(M_newRuntimeConfig(), M_freeRuntimeConfig);
  ASSERT_NE(nullptr, config);
  RuntimeContextPtr runtime(M_newRuntimeContext(config.get(), status.get()),
                            M_freeRuntimeContext);
  EXPECT_FALSE(M_isError(status.get())) << M_getError(status.get());
  ASSERT_NE(nullptr, runtime);
  auto allocRuntime = createRuntimeForTest();
  auto queryOr =
      gen->generateQuery(allocRuntime->getCompactPtr(), runtime.get(), 0);
  ASSERT_TRUE(queryOr.isError());
  EXPECT_STREQ("Tea kettle left on stove", queryOr.getError());
}

TEST(QueryGeneratorIntegration, Basic) {
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
  auto allocRuntime = createRuntimeForTest();
  auto queryOr =
      gen->generateQuery(allocRuntime->getCompactPtr(), runtime.get(), 0);
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

TEST(QueryGeneratorIntegration, ErrorPath) {
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

TEST(QueryGeneratorIntegration, UnspecifiedInputs) {
  TestInputSchemaErrorReporter errorReporter;
  InputGenerationSchema schema;
  auto gen = createQueryGenerator(schema, errorReporter);
  EXPECT_EQ(nullptr, gen);
  EXPECT_EQ(
      (std::vector<ReportedError>{{{"inputs"}, "Inputs must be specified"}}),
      errorReporter.errors);
}

// TODO: Test for duplicate input names or indices in schema
