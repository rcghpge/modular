//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/BenchmarkCLI/Main.h"
#include "Config/Version.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/BenchmarkCLI/InputGenerationSchema.h"
#include "GenericML/BenchmarkCLI/InputGenerator.h"
#include "GenericML/BenchmarkCLI/MLPerfIntegration.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "mlperf-loadgen/loadgen.h"
#include "mlperf-loadgen/test_settings.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <iostream>

using namespace M;
using namespace M::BenchmarkCLI;

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

static std::unique_ptr<QueryGenerator>
createQueryGeneratorFromPath(StringRef schemaPath) {
  InputGenerationSchema schema;
  if (!schemaPath.empty()) {
    auto schemaOr = readInputGenerationSchema(schemaPath);
    if (schemaOr.isError()) {
      llvm::errs() << "Error parsing input generation schema: "
                   << schemaOr.getError() << '\n';
      return nullptr;
    }
    schema = std::move(*schemaOr);
  }
  StandardInputSchemaErrorReporter errorReporter;
  return createQueryGenerator(schema, errorReporter);
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

  if (opts.roundTripInputGenerationSchema) {
    if (!roundTripInputGenerationSchema(opts.inputGenerationSchemaPath))
      return EXIT_FAILURE;
    return EXIT_SUCCESS;
  }

  std::unique_ptr<QueryGenerator> queryGenerator =
      createQueryGeneratorFromPath(opts.inputGenerationSchemaPath);
  if (!queryGenerator)
    return EXIT_FAILURE;

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

  // The work queue of this runtime is never used (it's used only for the
  // allocator), so we use a single-thread work queue to avoid using
  // unnecessary resources.
  LLCL::Runtime allocRuntime(LLCL::createMallocAllocator(),
                             LLCL::createSingleThreadWorkQueue());
  QuerySampleLibrary qsl(allocRuntime.getCompactPtr(), runtimeContext.get(),
                         std::move(queryGenerator));
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
