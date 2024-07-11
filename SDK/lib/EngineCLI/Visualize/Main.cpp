//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/EngineCLI/Visualize/Main.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Config/Version.h"
#include "GenericML/API/max/c/custom_ops.h"
#include "GenericML/API/max/c/internal/telemetry.h"
#include "GenericML/API/max/cpp/aliases.h"
#include "SDK/EngineCLI/Common/InputGenerationSchema.h"
#include "SDK/EngineCLI/Common/TorchSupport.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "Support/Process.h"
#include "Support/VisualizationUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#define DRIVER_OPTIONS_PATH "VisualizeOptions.inc"
#include "Support/Driver/OptTable.inc"

using namespace M;
using namespace M::EngineCLI;

static constexpr const char *kTelemetryLoggerName = "max";
static constexpr const char *kTelemetryErrorEventName = "visualize.error";
static constexpr const char *kTelemetrySuccessEventName = "visualize.success";

namespace {

struct VisualizeOptTable : public llvm::opt::PrecomputedOptTable {
  VisualizeOptTable()
      : llvm::opt::PrecomputedOptTable(InfoTable, PrefixTable) {}
};

struct ClOptions {
  std::string modelPath;
  std::string inputGenerationSchemaPath;
  std::string visualizationPath;
  std::string customOpsPath;
};

} // namespace

static void printVersion(llvm::raw_ostream &os) {
  ModularVersion version = getModularVersion();
  os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", maxVisualize, version.major,
                      version.minor, version.patch, version.label,
                      version.revision);
  os << "Modular version " << M_version() << "\n";
}

static ClOptions parseOptions(const State &state) {
  ClOptions opts;
  VisualizeOptTable optTable;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  auto args = optTable.ParseArgs(state.arguments, missingIndex, missingCount);
  if (int exitCode = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    std::exit(exitCode);
  if (args.hasArg(options::OPT_help))
    std::exit(state.printHelp(
#include "VisualizeOptionsHelpText.inc"
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
  if (args.hasMultipleArgs(options::OPT_input_generation_schema))
    std::exit(
        state.reportError("must provide at most one input generation schema"));
  if (args.hasArg(options::OPT_input_generation_schema))
    opts.inputGenerationSchemaPath =
        args.getLastArgValue(options::OPT_input_generation_schema);
  if (args.hasMultipleArgs(options::OPT_output))
    std::exit(state.reportError(
        "must provide at most one visualization output path"));
  if (args.hasArg(options::OPT_output))
    opts.visualizationPath = args.getLastArgValue(options::OPT_output);
  if (args.hasMultipleArgs(options::OPT_custom_ops_path))
    std::exit(state.reportError("must provide at most one custom ops path"));
  if (args.hasArg(options::OPT_custom_ops_path))
    opts.customOpsPath = args.getLastArgValue(options::OPT_custom_ops_path);
  return opts;
}

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

  M_enableVisualization(compileConfig.get(), opts.visualizationPath.c_str());

  if (!opts.customOpsPath.empty()) {
    M_useKernelsFrom(compileConfig.get(), opts.customOpsPath.c_str(), status);
    if (M_isError(status))
      return Error(Twine("Failed to set custom ops path: ") +
                   M_getError(status));
  }
  M_setModelPath(compileConfig.get(), opts.modelPath.c_str());
  M_CompileConfig *compileConfigPtr = compileConfig.release();
  CompiledModelPtr model(
      M_compileModelSync(runtimeContext, &compileConfigPtr, status),
      M_freeCompiledModel);
  if (!model)
    return Error(M_getError(status));
  return std::move(model);
}

int M::maxVisualizeMain(int argc, char **argv) {

  State state(maxVisualize, {argv + 1, static_cast<size_t>(argc) - 1});
  ClOptions opts = parseOptions(state);

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

  auto runtimeContextOr = makeRuntimeContext(status.get());
  if (runtimeContextOr.isError()) {
    llvm::errs() << "Error: Failed to create runtime context: "
                 << runtimeContextOr.getError() << "\n";
    return EXIT_FAILURE;
  }
  RuntimeContextPtr runtimeContext = std::move(*runtimeContextOr);
  auto telemetryCtx = M_unwrapTelemetryContext(runtimeContext.get());
  auto logger = telemetryCtx->getLogger(kTelemetryLoggerName);

  auto modelOr = compileModel(opts, inputGenerationSchema, runtimeContext.get(),
                              status.get());
  if (modelOr.isError()) {
    llvm::errs() << "Error compiling model: " << modelOr.getError() << "\n";
    logger->emitL1Event(kTelemetryErrorEventName,
                        {{"error", M_getError(status.get())}});
    return EXIT_FAILURE;
  }

  // TODO: Fix this filename.
  ErrorOr<std::string> outputOr = createFilepath(opts.visualizationPath);
  if (outputOr.isError()) {
    llvm::errs()
        << "Unable to determine generated output visualization file path\n";
  }

  llvm::outs() << "\nModular maxviz file saved to: " << outputOr.get() << "\n";

  llvm::outs()
      << "To visualize model, drop this maxviz file into https://netron.app/\n";
  logger->emitL1Event(kTelemetrySuccessEventName);
  return EXIT_SUCCESS;
}
