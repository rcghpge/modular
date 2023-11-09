//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/ModexBenchmark/MLPerfIntegration.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/ModexBenchmark/InputGenerator.h"
#include "GenericML/ModexBenchmark/Main.h"
#include "mlperf-loadgen/loadgen.h"

using namespace M;
using namespace M::ModexBench;

static constexpr size_t kDefaultSampleCount = 1000;

static size_t pickSampleCount(const QueryGenerator &qgen) {
  ssize_t qgenQueries = qgen.getNumQueries();
  if (qgenQueries == -1)
    return kDefaultSampleCount;
  assert(qgenQueries >= 0);
  return static_cast<size_t>(qgenQueries);
}

QuerySampleLibrary::QuerySampleLibrary(
    LLCL::CompactRuntimePtr allocRuntime, M_RuntimeContext *runtimeContext,
    std::unique_ptr<QueryGenerator> queryGenerator)
    : allocRuntime(allocRuntime), runtimeContext(runtimeContext),
      sampleCount(pickSampleCount(*queryGenerator)),
      queryGenerator(std::move(queryGenerator)),
      loadedSamples(
          std::make_unique<std::atomic<GeneratedQuery *>[]>(sampleCount)) {}
QuerySampleLibrary::~QuerySampleLibrary() {
  for (size_t i = 0; i < sampleCount; ++i)
    unloadSample(i);
}

std::string &QuerySampleLibrary::Name() const {
  static std::string *name = new std::string(kModexBenchProductName);
  return *name;
}
size_t QuerySampleLibrary::TotalSampleCount() { return sampleCount; }
size_t QuerySampleLibrary::PerformanceSampleCount() {
  return TotalSampleCount();
}

void QuerySampleLibrary::LoadSamplesToRam(
    const std::vector<mlperf::QuerySampleIndex> &samples) {
  for (mlperf::QuerySampleIndex i : samples)
    loadSample(i);
}
void QuerySampleLibrary::UnloadSamplesFromRam(
    const std::vector<mlperf::QuerySampleIndex> &samples) {
  for (mlperf::QuerySampleIndex i : samples)
    unloadSample(i);
}

void QuerySampleLibrary::loadSample(mlperf::QuerySampleIndex index) {
  assert(index < sampleCount);
  auto queryOr =
      queryGenerator->generateQuery(allocRuntime, runtimeContext, index);
  if (queryOr.isError()) {
    llvm::errs() << "Error loading sample " << index << ": "
                 << queryOr.getError() << "\n";
    mlperf::AbortTest();
    return;
  }
  delete loadedSamples[index].exchange(queryOr->release());
}

void QuerySampleLibrary::unloadSample(mlperf::QuerySampleIndex index) {
  assert(index < sampleCount);
  delete loadedSamples[index].exchange(nullptr);
}

SystemUnderTest::SystemUnderTest(QuerySampleLibrary *qsl,
                                 M_RuntimeContext *runtimeContext,
                                 M_AsyncModel *model)
    : qsl(qsl), runtimeContext(runtimeContext), model(model) {}

std::string &SystemUnderTest::Name() const {
  static std::string *name = new std::string(kModexBenchProductName);
  return *name;
}

void SystemUnderTest::IssueQuery(
    const std::vector<mlperf::QuerySample> &samples) {
  ErrorOrSuccess result = issueQueryFallible(samples);
  if (result.isError()) {
    llvm::errs() << "Error issuing query: " << result.getError() << "\n";
    mlperf::AbortTest();
  }
}

ErrorOrSuccess SystemUnderTest::issueQueryFallible(
    const std::vector<mlperf::QuerySample> &samples) {
  StatusPtr status(M_newStatus(), M_freeStatus);
  if (!status)
    return Error("Failed to create M_Status (out of memory?)");
  std::vector<mlperf::QuerySampleResponse> responses{};
  std::vector<TensorMapPtr> outputATMs;
  responses.resize(samples.size());
  outputATMs.reserve(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    const mlperf::QuerySample &sample = samples[i];
    GeneratedQuery *generatedQuery = qsl->getLoadedSample(sample.index);
    if (generatedQuery == nullptr)
      return Error("Query issued on unloaded sample");
    TensorMapPtr outputATM(
        M_executeModelSync(runtimeContext, model,
                           generatedQuery->getAsyncTensorMap(), status.get()),
        M_freeAsyncTensorMap);
    if (!outputATM)
      return Error(llvm::Twine("When executing inference: ") +
                   M_getError(status.get()));
    outputATMs.emplace_back(std::move(outputATM));

    mlperf::QuerySampleResponse &response = responses[i];
    response.id = sample.id;
    static char dummyData = 1;
    response.data = reinterpret_cast<uintptr_t>(&dummyData);
    response.size = 1;
  }
  mlperf::QuerySamplesComplete(responses.data(), responses.size());
  return success();
}

void SystemUnderTest::FlushQueries() {
  // Don't need to do anything here, since Modular does not batch queries.
}
