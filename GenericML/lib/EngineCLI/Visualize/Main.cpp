//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/EngineCLI/Visualize/Main.h"
#include "Config/Version.h"
#include "GenericML/API/modular/c/custom_ops.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "GenericML/EngineCLI/Common/InputGenerationSchema.h"
#include "GenericML/EngineCLI/Common/TorchSupport.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/Process.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace M;
using namespace M::EngineCLI;

static constexpr const char *kTelemetryLoggerName = "max";
static constexpr const char *kTelemetryErrorEventName = "visualize.error";
static constexpr const char *kTelemetrySuccessEventName = "visualize.success";

namespace {

struct ClOptions {
  cl::opt<std::string> modelPath{llvm::cl::Positional, llvm::cl::Required,
                                 cl::desc("Path to model")};
  cl::opt<std::string> inputGenerationSchemaPath{
      "input-generation-schema",
      cl::desc("Path to input generation schema file "
               "(only used for compile-time shape specifications)")};

  cl::opt<std::string> visualizationDir{
      "output-directory", cl::desc("Path to directory for output maxviz file."),
      cl::init("")};
  cl::opt<std::string> customOpsPath{
      "custom-ops-path", cl::desc("Path to custom ops"), cl::init("")};
  cl::opt<size_t> numThreads{"num-threads",
                             cl::desc("Number of threads to use"), cl::init(0)};
};

} // namespace

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

static ErrorOr<CompiledModelPtr>
compileModel(const ClOptions &opts,
             const InputGenerationSchema &inputGenerationSchema,
             M_RuntimeContext *runtimeContext, M_Status *status) {
  CompileConfigPtr compileConfig(M_newCompileConfig(), M_freeCompileConfig);
  if (!compileConfig)
    return Error("Failed to create M_CompileConfig (out of memory?)");
  if (auto result =
          setTorchInputSpecs(compileConfig.get(), inputGenerationSchema))
    return Error(Twine("Failed to set input specs for PyTorch: ") +
                 result.getError());

  M_enableVisualization(compileConfig.get(), opts.visualizationDir.c_str());

  if (opts.customOpsPath != "") {
    M_useKernelsFrom(compileConfig.get(), opts.customOpsPath.c_str(), status);
    if (M_isError(status))
      return Error(Twine("Failed to set custom ops path: ") +
                   M_getError(status));
  }
  M_setModelPath(compileConfig.get(), opts.modelPath.c_str());
  CompiledModelPtr model(
      M_compileModelSync(runtimeContext, compileConfig.get(), status),
      M_freeCompiledModel);
  if (!model)
    return Error(M_getError(status));
  return std::move(model);
}

int M::maxVisualizeMain(int argc, char **argv) {

  // TODO: this is a temp workaround for some issues in the options pipeline. It
  // will be replaced with a proper flag soon.

  (void)M::setProcessEnv("GENERATE_VISUALIZATION", "1");

  ClOptions opts;

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", maxVisualize,
                        version.major, version.minor, version.patch,
                        version.label, version.revision);
    os << "Modular version " << M_version() << "\n";
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, maxVisualize);

  llvm::outs() << "Modular Visualization Tool\n";
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

  auto modelOr = compileModel(opts, inputGenerationSchema, runtimeContext.get(),
                              status.get());
  if (modelOr.isError()) {
    llvm::errs() << "Error compiling model: " << modelOr.getError() << "\n";
    M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                         kTelemetryErrorEventName, M_getError(status.get()));
    return EXIT_FAILURE;
  }
  (void)M::setProcessEnv("GENERATE_VISUALIZATION", "");

#ifndef MODULAR_PRODUCTION
  std::string output;
  if (!opts.visualizationDir.empty()) {
    output = std::filesystem::current_path() /
             std::filesystem::path(opts.visualizationDir.c_str());
  } else {
    output = std::filesystem::current_path();
  }
  llvm::outs() << "\nModular maxviz file saved to " << output << "\n";

  llvm::outs()
      << "To visualize model, drop maxviz file into https://netron.app/\n";
#endif // !MODULAR_PRODUCTION
  M_emitTelemetryEvent(runtimeContext.get(), kTelemetryLoggerName,
                       kTelemetrySuccessEventName, "");
  return EXIT_SUCCESS;
}
