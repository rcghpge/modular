//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Benchmark/Main.h"
#include "Config/Version.h"
#include "GenericML/EngineCLI/Benchmark/InputGenerationDefaults.h"
#include "GenericML/EngineCLI/Benchmark/InputGenerator.h"
#include "GenericML/EngineCLI/Benchmark/MLPerfIntegration.h"
#include "GenericML/EngineCLI/Common/APIBridge.h"
#include "GenericML/EngineCLI/Common/InputGenerationSchema.h"
#include "GenericML/EngineCLI/Common/TorchSupport.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "mlperf-loadgen/loadgen.h"
#include "mlperf-loadgen/test_settings.h"
#include "modular/c/custom_ops.h"
#include "modular/c/internal/telemetry.h"
#include "modular/cpp/aliases.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <iostream>

#define DRIVER_OPTIONS_PATH "BenchmarkOptions.inc"
#include "Support/Driver/OptTable.inc"

using namespace M;
using namespace M::EngineCLI;
using namespace M::BenchmarkCLI;

static constexpr const char *kTelemetryLoggerName = "max";
static constexpr const char *kTelemetryErrorEventName = "benchmark.error";
static constexpr const char *kTelemetrySuccessEventName = "benchmark.success";

namespace {

struct BenchmarkOptTable : public llvm::opt::PrecomputedOptTable {
  BenchmarkOptTable()
      : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};

struct ClOptions {
  std::string modelPath;
  mlperf::TestScenario mlPerfScenario = mlperf::TestScenario::SingleStream;
  std::string inputGenerationSchemaPath;
  bool roundTripInputGenerationSchema = false;
  std::string customOpsPath;
  size_t numThreads = 0;
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

static void printVersion(llvm::raw_ostream &os) {
  ModularVersion version = getModularVersion();
  os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kBenchmarkCLIProductName,
                      version.major, version.minor, version.patch,
                      version.label, version.revision);
  os << "Modular version " << M_version() << "\n";
}

static std::optional<mlperf::TestScenario> parseMLPerfScenario(StringRef str) {
  return llvm::StringSwitch<std::optional<mlperf::TestScenario>>(str)
      .Case("single-stream", mlperf::TestScenario::SingleStream)
      .Case("multiple-stream-2.0", mlperf::TestScenario::MultiStream)
      .Case("server", mlperf::TestScenario::Server)
      .Case("offline", mlperf::TestScenario::Offline)
      .Default(std::nullopt);
}

static ClOptions parseOptions(const State &state) {
  ClOptions opts;
  BenchmarkOptTable optTable;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  auto args = optTable.ParseArgs(state.arguments, missingIndex, missingCount);
  if (int exitCode = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    std::exit(exitCode);
  if (args.hasArg(options::OPT_help))
    std::exit(state.printHelp(
#include "BenchmarkOptionsHelpText.inc"
        ));
  if (args.hasArg(options::OPT_version)) {
    printVersion(llvm::outs());
    std::exit(EXIT_SUCCESS);
  }
  if (!args.hasArg(options::OPT_INPUT))
    std::exit(state.reportError("no model path provided"));
  if (args.hasMultipleArgs(options::OPT_INPUT))
    std::exit(state.reportError("must provide at most one model path"));
  opts.modelPath = args.getLastArgValue(options::OPT_INPUT);
  if (args.hasMultipleArgs(options::OPT_mlperf_scenario))
    std::exit(state.reportError("must provide at most one MLPerf scenario"));
  if (args.hasArg(options::OPT_mlperf_scenario)) {
    if (auto scenario = parseMLPerfScenario(
            args.getLastArgValue(options::OPT_mlperf_scenario));
        scenario.has_value())
      opts.mlPerfScenario = *scenario;
    else
      std::exit(
          state.reportError("unrecognized MLPerf scenario `" +
                            args.getLastArgValue(options::OPT_mlperf_scenario) +
                            "' (known scenarios: single-stream, "
                            "multiple-stream-2.0, server, offline)"));
  }
  if (args.hasMultipleArgs(options::OPT_input_generation_schema))
    std::exit(
        state.reportError("must provide at most one input generation schema"));
  if (args.hasArg(options::OPT_input_generation_schema))
    opts.inputGenerationSchemaPath =
        args.getLastArgValue(options::OPT_input_generation_schema);
  opts.roundTripInputGenerationSchema =
      args.hasArg(options::OPT_round_trip_input_generation_schema);
  if (args.hasMultipleArgs(options::OPT_custom_ops_path))
    std::exit(state.reportError("must provide at most one custom ops path"));
  if (args.hasArg(options::OPT_custom_ops_path))
    opts.customOpsPath = args.getLastArgValue(options::OPT_custom_ops_path);
  if (args.hasMultipleArgs(options::OPT_num_threads))
    std::exit(state.reportError("must provide at most one number of threads"));
  if (args.hasArg(options::OPT_num_threads) &&
      args.getLastArgValue(options::OPT_num_threads)
          .getAsInteger(10, opts.numThreads))
    std::exit(state.reportError("invalid value `" +
                                args.getLastArgValue(options::OPT_num_threads) +
                                "' for --num-threads"));
  return opts;
}

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
                                       QueryGenerator &queryGenerator) {
  ErrorOr<std::unique_ptr<GeneratedQuery>> queryOr =
      queryGenerator.generateQuery(runtimeContext, 0);
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
  State state(kBenchmarkCLIProductName,
              {argv + 1, static_cast<size_t>(argc) - 1});
  ClOptions opts = parseOptions(state);

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
    llvm::outs() << "Input data schema is valid\n";
    llvm::outs() << "Round-tripped input data schema:\n";
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

  CompileConfigPtr compileConfig(M_newCompileConfig(), M_freeCompileConfig);
  if (!compileConfig) {
    llvm::errs()
        << "Error: Failed to create M_CompileConfig (out of memory?)\n";
    return EXIT_FAILURE;
  }
  M_setModelPath(compileConfig.get(), opts.modelPath.c_str());
  if (!opts.customOpsPath.empty()) {
    M_useKernelsFrom(compileConfig.get(), opts.customOpsPath.c_str(),
                     status.get());
    if (M_isError(status.get())) {
      llvm::errs() << "Error: Unable to use custom kernels: "
                   << M_getError(status.get()) << "\n";
      return EXIT_FAILURE;
    }
  }
  if (auto result =
          setTorchInputSpecs(compileConfig.get(), inputGenerationSchema)) {
    llvm::errs() << "Error: Failed to set input specs for PyTorch: "
                 << result.getError() << "\n";
    return EXIT_FAILURE;
  }

  M::Telemetry::TelemetryContext *telemetryCtx =
      M_unwrapTelemetryContext(runtimeContext.get());
  std::shared_ptr<Telemetry::Logs::Logger> logger =
      telemetryCtx->getLogger(kTelemetryLoggerName);

  CompiledModelPtr compiledModel(nullptr, M_freeCompiledModel);
  M_CompileConfig *compileConfigPtr = compileConfig.release();
  llvm::outs() << "Compiling model... ";
  {
    auto start = std::chrono::steady_clock::now();
    compiledModel =
        CompiledModelPtr(M_compileModelSync(runtimeContext.get(),
                                            &compileConfigPtr, status.get()),
                         M_freeCompiledModel);
    auto end = std::chrono::steady_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() *
        1.0e-6;
    if (!compiledModel) {
      llvm::outs() << "Error (took " << duration << "s)\n";
      llvm::outs().flush();
      llvm::errs() << "Error compiling model: " << M_getError(status.get())
                   << "\n";
      logger->emitL1Event(kTelemetryErrorEventName,
                          {{"error", M_getError(status.get())}});
      return EXIT_FAILURE;
    }
    llvm::outs() << "OK (took " << duration << "s)\n";
  }

  applyInputGenerationDefaults(inputGenerationSchema, compiledModel.get(),
                               status.get());
  if (M_isError(status.get())) {
    llvm::errs() << "Error applying defaults to input generation schema: "
                 << M_getError(status.get()) << "\n";
    logger->emitL1Event(kTelemetryErrorEventName,
                        {{"error", M_getError(status.get())}});
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
      logger->emitL1Event(kTelemetryErrorEventName,
                          {{"error", modelOr.getError()}});
      return EXIT_FAILURE;
    }
    model = std::move(*modelOr);
  }

  mlperf::TestSettings mlperfTestSettings;
  mlperfTestSettings.scenario = opts.mlPerfScenario;
  mlperf::LogSettings mlperfLogSettings;
  mlperfLogSettings.log_output.copy_summary_to_stdout = true;

  {
    llvm::outs() << "Estimating latency for query generation... ";
    ErrorOr<double> estimatedLatencyOr =
        estimateLatency(runtimeContext.get(), model.get(), *queryGenerator);
    if (estimatedLatencyOr.isError()) {
      llvm::outs() << "Error\n";
      llvm::outs().flush();
      llvm::errs() << "Error estimating latency: "
                   << estimatedLatencyOr.getError() << "\n";
      logger->emitL1Event(kTelemetryErrorEventName,
                          {{"error", estimatedLatencyOr.getError()}});
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

  QuerySampleLibrary qsl(runtimeContext.get(), std::move(queryGenerator));
  SystemUnderTest sut(&qsl, runtimeContext.get(), model.get());

  llvm::outs() << "Benchmarking\n";
  // MLPerf uses C++ <iostream> for IO.  To prevent incorrect interleaving, we
  // need to flush LLVM outs before mlperf::StartTest and flush C++ <iostream>
  // after.
  llvm::outs().flush();
  mlperf::StartTest(&sut, &qsl, mlperfTestSettings, mlperfLogSettings);
  std::cout << std::flush;
  llvm::outs() << "Benchmark finished\n";

  logger->emitL1Event(kTelemetrySuccessEventName);
  return EXIT_SUCCESS;
}
