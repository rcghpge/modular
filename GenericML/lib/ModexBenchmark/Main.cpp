//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/ModexBenchmark/Main.h"
#include "Config/Version.h"
#include "GenericML/API/modular/c/common.h"
#include "GenericML/API/modular/c/context.h"
#include "GenericML/API/modular/c/model.h"
#include "GenericML/API/modular/c/tensor.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/ModexBenchmark/InputGenerationSchema.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "mlperf-loadgen/loadgen.h"
#include "mlperf-loadgen/query_sample_library.h"
#include "mlperf-loadgen/system_under_test.h"
#include "mlperf-loadgen/test_settings.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <iostream>

using namespace M;
using namespace M::ModexBench;

static constexpr const char *kModexBenchProductName = "modex benchmark";

namespace {

struct ClOptions {
  cl::opt<std::string> modelPath{llvm::cl::Positional, llvm::cl::Required,
                                 cl::desc("Path to model file/directory")};
  cl::opt<mlperf::TestScenario> mlPerfScenario{
      "mlperf-scenario",
      cl::desc("MLPerf Inference scenario to test (see "
               "https://mlcommons.org/en/inference-datacenter-30/"
               "#scenarios-and-metrics for definitions)"),
      cl::init(mlperf::TestScenario::SingleStream),
      cl::values(
          clEnumValN(mlperf::TestScenario::SingleStream, "single-stream",
                     "Single stream"),
          clEnumValN(mlperf::TestScenario::MultiStream, "multiple-stream-2.0",
                     "Multiple Stream 2.0"),
          clEnumValN(mlperf::TestScenario::Server, "server", "Server"),
          clEnumValN(mlperf::TestScenario::Offline, "offline", "Offline"))};
  cl::opt<std::string> inputGenerationSchemaPath{
      "input-generation-schema",
      cl::desc("Path to input generation schema file")};
  cl::opt<bool> roundTripInputGenerationSchema{
      "round-trip-input-generation-schema",
      cl::desc("Validate and round trip the provided input generation schema "
               "and do not proceed to run benchmarking"),
      llvm::cl::Hidden};
};

class LoadedSample final {
public:
  static ErrorOr<LoadedSample *> Create();
  ~LoadedSample() = default;

  M_TensorSpec *getSpec() const { return spec.get(); }
  void *getData() const { return data.get(); }

private:
  LoadedSample() = default;

  TensorSpecPtr spec = TensorSpecPtr(nullptr, M_freeTensorSpec);
  std::unique_ptr<float[]> data;
};

class QuerySampleLibrary final : public mlperf::QuerySampleLibrary {
public:
  explicit QuerySampleLibrary(size_t sampleCount);
  ~QuerySampleLibrary();

  std::string &Name() const override {
    static std::string *name = new std::string(kModexBenchProductName);
    return *name;
  }

  size_t TotalSampleCount() override { return sampleCount; }

  size_t PerformanceSampleCount() override { return TotalSampleCount(); }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override;
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override;

  LoadedSample *getLoadedSample(mlperf::QuerySampleIndex index) {
    return loadedSamples[index].load();
  }

private:
  void loadSample(mlperf::QuerySampleIndex index);
  void unloadSample(mlperf::QuerySampleIndex index);

  std::unique_ptr<std::atomic<LoadedSample *>[]> loadedSamples;
  std::size_t sampleCount;
};

class SystemUnderTest final : public mlperf::SystemUnderTest {
public:
  explicit SystemUnderTest(QuerySampleLibrary *qsl,
                           M_RuntimeContext *runtimeContext,
                           M_AsyncModel *model);

  std::string &Name() const override {
    static std::string *name = new std::string(kModexBenchProductName);
    return *name;
  }

  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override;
  void FlushQueries() override;

private:
  ErrorOrSuccess
  issueQueryFallible(const std::vector<mlperf::QuerySample> &samples);

  QuerySampleLibrary *qsl;
  M_RuntimeContext *runtimeContext;
  M_AsyncModel *model;
};

} // namespace

namespace mlperf {
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     mlperf::TestScenario scenario) {
  switch (scenario) {
  case mlperf::TestScenario::SingleStream:
    return os << "Single Stream";
  case mlperf::TestScenario::MultiStream:
    return os << "Multiple Stream 2.0";
  case mlperf::TestScenario::Server:
    return os << "Server";
  case mlperf::TestScenario::Offline:
    return os << "Offline";
  }
  llvm::report_fatal_error("Unknown mlperf::TestScenario value");
}
} // namespace mlperf

static ErrorOr<RuntimeContextPtr> makeRuntimeContext(M_Status *status) {
  RuntimeConfigPtr runtimeConfig(M_newRuntimeConfig(), M_freeRuntimeConfig);
  if (!runtimeConfig)
    return Error("Failed to create M_RuntimeConfig (out of memory?)");
  RuntimeContextPtr context(M_newRuntimeContext(runtimeConfig.get(), status),
                            M_freeRuntimeContext);
  if (!context)
    return Error(M_getError(status));
  return std::move(context);
}

static ErrorOr<CompiledModelPtr> compileModel(const ClOptions &opts,
                                              M_RuntimeContext *runtimeContext,
                                              M_Status *status) {
  CompileConfigPtr compileConfig(M_newCompileConfig(), M_freeCompileConfig);
  if (!compileConfig)
    return Error("Failed to create M_CompileConfig (out of memory?)");
  M_setModelPath(compileConfig.get(), opts.modelPath.c_str());
  CompiledModelPtr model(
      M_compileModel(runtimeContext, compileConfig.get(), status),
      M_freeCompiledModel);
  if (!model)
    return Error(M_getError(status));
  M_waitForCompilation(model.get(), status);
  if (M_isError(status))
    return Error(M_getError(status));
  return std::move(model);
}

static ErrorOr<ModelPtr> prepareModel(M_RuntimeContext *runtimeContext,
                                      M_AsyncCompiledModel *compiledModel,
                                      M_Status *status) {
  ModelPtr model(M_initModel(runtimeContext, compiledModel, status),
                 M_freeModel);
  if (!model)
    return Error(M_getError(status));
  M_waitForModel(model.get(), status);
  if (M_isError(status))
    return Error(M_getError(status));
  return std::move(model);
}

static ErrorOr<TensorMapPtr> prepareInput(M_RuntimeContext *runtimeContext,
                                          M_Status *status,
                                          LoadedSample *sample) {
  TensorMapPtr atm(M_newAsyncTensorMap(runtimeContext), M_freeAsyncTensorMap);
  if (!atm)
    return Error("Failed to create M_AsyncTensorMap (out of memory?)");
  M_borrowTensorInto(atm.get(), sample->getData(), sample->getSpec(), status);
  if (M_isError(status))
    return Error(llvm::Twine("When borrowing tensor into input map: ") +
                 M_getError(status));
  return std::move(atm);
}

ErrorOr<LoadedSample *> LoadedSample::Create() {
  std::unique_ptr<LoadedSample> sample(new LoadedSample);
  sample->spec = TensorSpecPtr(
      M_newTensorSpec(
          std::initializer_list<std::int64_t>{1, 100, 100, 3}.begin(), 4,
          M_FLOAT32, "inputs"),
      M_freeTensorSpec);
  if (!sample->spec)
    return Error("Failed to create M_TensorSpec (out of memory?)");
  sample->data = std::make_unique<float[]>(1 * 100 * 100 * 3);
  std::fill(sample->data.get(), sample->data.get() + 1 * 100 * 100 * 3, 0.0f);
  return sample.release();
}

QuerySampleLibrary::QuerySampleLibrary(size_t sampleCount)
    : loadedSamples(
          std::make_unique<std::atomic<LoadedSample *>[]>(sampleCount)),
      sampleCount(sampleCount) {}
QuerySampleLibrary::~QuerySampleLibrary() {
  for (size_t i = 0; i < sampleCount; ++i)
    unloadSample(i);
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
  auto sampleOr = LoadedSample::Create();
  if (sampleOr.isError()) {
    llvm::errs() << "Error loading sample " << index << ": "
                 << sampleOr.getError() << "\n";
    mlperf::AbortTest();
    return;
  }
  delete loadedSamples[index].exchange(*sampleOr);
}

void QuerySampleLibrary::unloadSample(mlperf::QuerySampleIndex index) {
  assert(index < sampleCount);
  delete loadedSamples[index].exchange(nullptr);
}

SystemUnderTest::SystemUnderTest(QuerySampleLibrary *qsl,
                                 M_RuntimeContext *runtimeContext,
                                 M_AsyncModel *model)
    : qsl(qsl), runtimeContext(runtimeContext), model(model) {}

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
    LoadedSample *loadedSample = qsl->getLoadedSample(sample.index);
    if (loadedSample == nullptr)
      return Error("Query issued on unloaded sample");
    auto inputATMOr = prepareInput(runtimeContext, status.get(), loadedSample);
    if (inputATMOr.isError())
      return Error(llvm::Twine("When preparing input ATM: ") +
                   inputATMOr.getError());
    TensorMapPtr outputATM(M_executeModelSync(runtimeContext, model,
                                              inputATMOr->get(), status.get()),
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

int M::modexBenchmarkMain(int argc, char **argv) {
  ClOptions opts;

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kModexBenchProductName,
                        version.major, version.minor, version.patch,
                        version.label, version.revision);
    os << "Modular version " << M_version() << "\n";
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, kModexBenchProductName);

  llvm::outs() << "Modular Benchmarking Tool\n";
  llvm::outs() << "Modular version: " << M_version() << "\n";

  if (opts.roundTripInputGenerationSchema) {
    if (!roundTripInputGenerationSchema(opts.inputGenerationSchemaPath))
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }

  StatusPtr status(M_newStatus(), M_freeStatus);
  if (!status) {
    llvm::errs() << "Error: Failed to create M_Status (out of memory?)\n";
    return EXIT_FAILURE;
  }

  auto runtimeContextOr = makeRuntimeContext(status.get());
  if (runtimeContextOr.isError()) {
    llvm::errs() << "Error: Failed to create runtime context: "
                 << runtimeContextOr.getError() << "\n";
    return EXIT_FAILURE;
  }
  RuntimeContextPtr runtimeContext = std::move(*runtimeContextOr);

  llvm::outs() << "MLPerf scenario: " << opts.mlPerfScenario
               << " (use --mlperf-scenario to change)\n";

  CompiledModelPtr compiledModel(nullptr, M_freeCompiledModel);
  llvm::outs() << "Compiling model... ";
  {
    auto start = std::chrono::steady_clock::now();
    auto modelOr = compileModel(opts, runtimeContext.get(), status.get());
    auto end = std::chrono::steady_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() *
        1.0e-6;
    if (modelOr.isError()) {
      llvm::outs() << "Error (took " << duration << "s)\n";
      llvm::errs() << "Error compiling model: " << modelOr.getError() << "\n";
      return EXIT_FAILURE;
    }
    llvm::outs() << "OK (took " << duration << "s)\n";
    compiledModel = std::move(*modelOr);
  }

  ModelPtr model(nullptr, M_freeModel);
  {
    auto modelOr =
        prepareModel(runtimeContext.get(), compiledModel.get(), status.get());
    if (modelOr.isError()) {
      llvm::errs() << "Error preparing model: " << modelOr.getError() << "\n";
      return EXIT_FAILURE;
    }
    model = std::move(*modelOr);
  }

  QuerySampleLibrary qsl(1000);
  SystemUnderTest sut(&qsl, runtimeContext.get(), model.get());
  mlperf::TestSettings mlperfTestSettings;
  mlperfTestSettings.scenario = opts.mlPerfScenario;
  mlperf::LogSettings mlperfLogSettings;
  mlperfLogSettings.log_output.copy_summary_to_stdout = true;

  llvm::outs() << "Benchmarking\n";
  // MLPerf uses C++ <iostream> for IO.  To prevent incorrect interleaving, we
  // need to flush LLVM outs before mlperf::StartTest and flush C++ <iostream>
  // after.
  llvm::outs().flush();
  mlperf::StartTest(&sut, &qsl, mlperfTestSettings, mlperfLogSettings);
  std::cout << std::flush;
  llvm::outs() << "Benchmark finished\n";

  return EXIT_SUCCESS;
}
