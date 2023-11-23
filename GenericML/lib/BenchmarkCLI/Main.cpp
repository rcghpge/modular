//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/Main.h"
#include "Config/Version.h"
#include "GenericML/BenchmarkCLI/InputGenerationDefaults.h"
#include "GenericML/BenchmarkCLI/InputGenerationSchema.h"
#include "GenericML/BenchmarkCLI/InputGenerator.h"
#include "GenericML/BenchmarkCLI/MLPerfIntegration.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "mlperf-loadgen/loadgen.h"
#include "mlperf-loadgen/test_settings.h"
#include "modular/c/custom_ops.h"
#include "modular/cpp/aliases.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <iostream>

using namespace M;
using namespace M::BenchmarkCLI;

static constexpr const char *kTelemetryLoggerName = "max";
static constexpr const char *kTelemetryErrorEventName = "benchmark.error";
static constexpr const char *kTelemetrySuccessEventName = "benchmark.success";

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
  cl::opt<std::string> customOpsPath{
      "custom-ops-path", cl::desc("Path to custom ops"), cl::init("")};
  cl::opt<size_t> numThreads{"num-threads",
                             cl::desc("Number of threads to use"), cl::init(0)};
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

static ErrorOr<RuntimeContextPtr> makeRuntimeContext(M_Status *status,
                                                     size_t numThreads) {
  RuntimeConfigPtr runtimeConfig(M_newRuntimeConfig(), M_freeRuntimeConfig);
  if (!runtimeConfig)
    return Error("Failed to create M_RuntimeConfig (out of memory?)");

  M_setNumThreads(runtimeConfig.get(), numThreads);
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

  if (opts.customOpsPath != "") {
    M_useKernelsFrom(compileConfig.get(), opts.customOpsPath.c_str(), status);
    if (M_isError(status))
      return Error(M_getError(status));
  }

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

namespace {
struct StandardInputSchemaErrorReporter final
    : public InputSchemaErrorReporter {
public:
  void enterScope(StringRef scopeName) override {
    scopes.push_back(scopeName.str());
  }
  void exitScope(StringRef scopeName) override { scopes.pop_back(); }
  void reportError(Twine errorDetail) override {
    llvm::errs() << "In ";
    if (scopes.empty())
      llvm::errs() << "root of input generation schema";
    else
      llvm::interleave(scopes, llvm::errs(), ".");
    llvm::errs() << ": " << errorDetail << '\n';
  }

private:
  std::vector<std::string> scopes;
};
} // namespace

static ErrorOr<double> estimateLatency(M_RuntimeContext *runtimeContext,
                                       M_AsyncModel *model,
                                       QueryGenerator &queryGenerator,
                                       LLCL::CompactRuntimePtr allocRuntime) {
  ErrorOr<std::unique_ptr<GeneratedQuery>> queryOr =
      queryGenerator.generateQuery(allocRuntime, runtimeContext, 0);
  if (queryOr.isError())
    return Error(llvm::Twine("Unable to generate sample query: ") +
                 queryOr.getError());
  M_AsyncTensorMap *inputs = (*queryOr)->getAsyncTensorMap();
  StatusPtr status(M_newStatus(), M_freeStatus);
  if (!status)
    return Error("Failed to create M_Status (out of memory?)");
  auto start = std::chrono::steady_clock::now();
  auto end = start;
  int iterations = 0;
  // Run enough iterations to last at least a second.  That way we will be less
  // affected by noise for super tiny models.
  while (std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count() < 1000) {
    ++iterations;
    M_AsyncTensorMap *outputs =
        M_executeModelSync(runtimeContext, model, inputs, status.get());
    end = std::chrono::steady_clock::now();
    if (outputs)
      M_freeAsyncTensorMap(outputs);
    if (M_isError(status.get()))
      return Error(llvm::Twine("While executing model: ") +
                   M_getError(status.get()));
  }
  constexpr double kNsPerSec = 1e9;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
             .count() /
         kNsPerSec / iterations;
}

int M::maxBenchmarkMain(int argc, char **argv) {
  ClOptions opts;

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kBenchmarkCLIProductName,
                        version.major, version.minor, version.patch,
                        version.label, version.revision);
    os << "Modular version " << M_version() << "\n";
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, kBenchmarkCLIProductName);

  llvm::outs() << "Modular Benchmarking Tool\n";
  llvm::outs() << "Modular version: " << M_version() << "\n";

  InputGenerationSchema inputGenerationSchema;
  if (!opts.inputGenerationSchemaPath.empty()) {
    auto schemaOr = readInputGenerationSchema(opts.inputGenerationSchemaPath);
    if (schemaOr.isError()) {
      llvm::errs() << "Error parsing input generation schema: "
                   << schemaOr.getError() << "\n";
      return EXIT_FAILURE;
    }
    inputGenerationSchema = schemaOr.takeValue();
  }

  if (opts.roundTripInputGenerationSchema) {
    llvm::outs() << "Input generation schema is valid\n";
    llvm::outs() << "Round-tripped input generation schema:\n";
    dumpInputGenerationSchema(inputGenerationSchema);
    return EXIT_SUCCESS;
  }

  StatusPtr status(M_newStatus(), M_freeStatus);
  if (!status) {
    llvm::errs() << "Error: Failed to create M_Status (out of memory?)\n";
    return EXIT_FAILURE;
  }

  auto runtimeContextOr = makeRuntimeContext(status.get(), opts.numThreads);
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
      M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                           kTelemetryErrorEventName, modelOr.getError());
      return EXIT_FAILURE;
    }
    llvm::outs() << "OK (took " << duration << "s)\n";
    compiledModel = std::move(*modelOr);
  }

  applyInputGenerationDefaults(inputGenerationSchema, compiledModel.get(),
                               status.get());
  if (M_isError(status.get())) {
    llvm::errs() << "Error applying defaults to input generation schema: "
                 << M_getError(status.get()) << "\n";
    M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                         kTelemetryErrorEventName, M_getError(status.get()));
    return EXIT_FAILURE;
  }

  std::unique_ptr<QueryGenerator> queryGenerator;
  {
    StandardInputSchemaErrorReporter errorReporter;
    queryGenerator = createQueryGenerator(inputGenerationSchema, errorReporter);
  }
  if (!queryGenerator)
    return EXIT_FAILURE;

  ModelPtr model(nullptr, M_freeModel);
  {
    auto modelOr =
        prepareModel(runtimeContext.get(), compiledModel.get(), status.get());
    if (modelOr.isError()) {
      llvm::errs() << "Error preparing model: " << modelOr.getError() << "\n";
      M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                           kTelemetryErrorEventName, modelOr.getError());
      return EXIT_FAILURE;
    }
    model = std::move(*modelOr);
  }

  // The work queue of this runtime is never used (it's used only for the
  // allocator), so we use a single-thread work queue to avoid using
  // unnecessary resources.
  LLCL::Runtime allocRuntime(LLCL::createMallocAllocator(),
                             LLCL::createSingleThreadWorkQueue());
  mlperf::TestSettings mlperfTestSettings;
  mlperfTestSettings.scenario = opts.mlPerfScenario;
  mlperf::LogSettings mlperfLogSettings;
  mlperfLogSettings.log_output.copy_summary_to_stdout = true;

  {
    llvm::outs() << "Estimating latency for query generation... ";
    ErrorOr<double> estimatedLatencyOr =
        estimateLatency(runtimeContext.get(), model.get(), *queryGenerator,
                        allocRuntime.getCompactPtr());
    if (estimatedLatencyOr.isError()) {
      llvm::outs() << "Error\n";
      llvm::outs().flush();
      llvm::errs() << "Error estimating latency: "
                   << estimatedLatencyOr.getError() << "\n";
      M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                           kTelemetryErrorEventName,
                           estimatedLatencyOr.getError());
      return EXIT_FAILURE;
    }
    llvm::outs() << "OK\n";
    constexpr double kNsPerSec = 1e9;
    mlperfTestSettings.single_stream_expected_latency_ns =
        *estimatedLatencyOr * kNsPerSec;
    // N.B.: If we start supporting queries consisting of more than one sample,
    // this computation will need to change to take that into account.
    mlperfTestSettings.multi_stream_expected_latency_ns =
        *estimatedLatencyOr * kNsPerSec;
    mlperfTestSettings.multi_stream_samples_per_query = 1;
  }

  QuerySampleLibrary qsl(allocRuntime.getCompactPtr(), runtimeContext.get(),
                         std::move(queryGenerator));
  SystemUnderTest sut(&qsl, runtimeContext.get(), model.get());

  llvm::outs() << "Benchmarking\n";
  // MLPerf uses C++ <iostream> for IO.  To prevent incorrect interleaving, we
  // need to flush LLVM outs before mlperf::StartTest and flush C++ <iostream>
  // after.
  llvm::outs().flush();
  mlperf::StartTest(&sut, &qsl, mlperfTestSettings, mlperfLogSettings);
  std::cout << std::flush;
  llvm::outs() << "Benchmark finished\n";

  M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                       kTelemetrySuccessEventName, "");
  return EXIT_SUCCESS;
}
