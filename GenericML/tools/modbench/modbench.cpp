//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Config/Version.h"
#include "GenericML/API/modular/c/common.h"
#include "GenericML/API/modular/c/context.h"
#include "GenericML/API/modular/c/model.h"
#include "GenericML/API/modular/c/tensor.h"
#include "GenericML/API/modular/cpp/aliases.h"
#include "Support/CommandLine.h"
#include "Support/CrashReporting.h"
#include "Support/ErrorOr.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace M;

static constexpr const char *kModbenchProductName = "modbench";

namespace {

struct ClOptions {
  cl::opt<std::string> modelPath{llvm::cl::Positional, llvm::cl::Required,
                                 cl::desc("Path to model file/directory")};
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

static ErrorOr<std::tuple<TensorMapPtr, std::unique_ptr<float[]>>>
prepareInput(M_RuntimeContext *runtimeContext, M_Status *status) {
  TensorSpecPtr spec(
      M_newTensorSpec(
          std::initializer_list<std::int64_t>{1, 100, 100, 3}.begin(), 4,
          M_FLOAT32, "inputs"),
      M_freeTensorSpec);
  if (!spec)
    return Error("Failed to create M_TensorSpec (out of memory?)");
  std::unique_ptr<float[]> data = std::make_unique<float[]>(1 * 100 * 100 * 3);
  std::fill(data.get(), data.get() + 1 * 100 * 100 * 3, 0.0f);
  TensorMapPtr atm(M_newAsyncTensorMap(runtimeContext), M_freeAsyncTensorMap);
  if (!atm)
    return Error("Failed to create M_AsyncTensorMap (out of memory?)");
  M_borrowTensorInto(atm.get(), data.get(), spec.get(), status);
  if (M_isError(status))
    return Error(llvm::Twine("When borrowing tensor into input map: ") +
                 M_getError(status));
  return std::make_tuple(std::move(atm), std::move(data));
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  initCrashpadForProgram(argv[0], kModbenchProductName);
  llvm::setBugReportMsg(
      "Please submit a bug report to eap@modular.com and include the crash "
      "backtrace along with your command line invocation.\n");

  ClOptions opts;

  llvm::cl::SetVersionPrinter([](llvm::raw_ostream &os) {
    ModularVersion version = getModularVersion();
    os << llvm::formatv("{0} {1}.{2}.{3}{4} ({5})\n", kModbenchProductName,
                        version.major, version.minor, version.patch,
                        version.label, version.revision);
    os << "Modular version " << M_version() << "\n";
  });
  llvm::cl::ParseCommandLineOptions(argc, argv, kModbenchProductName);

  llvm::outs() << "Modular Benchmarking Tool\n";
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

  auto preparedInputOr = prepareInput(runtimeContext.get(), status.get());
  if (preparedInputOr.isError()) {
    llvm::errs() << "Error preparing input: " << preparedInputOr.getError()
                 << "\n";
    return EXIT_FAILURE;
  }
  auto [inputATM, inputData] = std::move(*preparedInputOr);

  llvm::outs() << "Benchmarking\n";
  double qps;
  {
    constexpr int kIterations = 100;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
      M_AsyncTensorMap *outputATM = M_executeModelSync(
          runtimeContext.get(), model.get(), inputATM.get(), status.get());
      if (!outputATM) {
        if (i > 0)
          llvm::outs() << '\n';
        llvm::errs() << "Error encountered during model execution: "
                     << M_getError(status.get()) << "\n";
        return EXIT_FAILURE;
      }
      M_freeAsyncTensorMap(outputATM);
      llvm::outs() << '.';
    }
    auto end = std::chrono::steady_clock::now();
    llvm::outs() << '\n';
    double duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() *
        1.0e-6;
    qps = kIterations / duration;
  }
  llvm::outs() << "Benchmark finished\n";
  llvm::outs() << "Average QPS: " << qps << "\n";

  return EXIT_SUCCESS;
}
