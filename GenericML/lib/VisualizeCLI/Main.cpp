//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/VisualizeCLI/Main.h"
#include "Config/Version.h"
#include "GenericML/API/modular/c/custom_ops.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/Process.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdlib>
#include <iostream>

using namespace M;

namespace {

struct ClOptions {
  cl::opt<std::string> modelPath{llvm::cl::Positional, llvm::cl::Required,
                                 cl::desc("Path to model file/directory")};
  cl::opt<std::string> customOpsPath{
      "custom-ops-path", cl::desc("Path to custom ops"), cl::init("")};
};

} // namespace

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

  auto modelOr = compileModel(opts, runtimeContext.get(), status.get());
  if (modelOr.isError()) {
    llvm::errs() << "Error compiling model: " << modelOr.getError() << "\n";
    return EXIT_FAILURE;
  }
  (void)M::setProcessEnv("GENERATE_VISUALIZATION", "");

  // TODO: Add to the spec and instructions for how to use with Netron.
  return EXIT_SUCCESS;
}
