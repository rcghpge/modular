//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CUDASupport/CUDARuntimeOptions.h"
#include "Config/Version.h"
#include "GenericML/CUDARegistration/CUDARegistration.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/Driver/DriverAPI.h"
#include "GenericML/Driver/DriverCLOptions.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "GenericML/Driver/Utils.h"
#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/MGPPrimitives/MGPPrimitives.h"
#include "GenericML/Support/CASConfig.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "MOF/Driver/DriverConfig.h"
#include "MOF/Driver/DriverImpl.h"
#include "MOF/ModularModel/ModularModelFile.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"

#include <string>
#include <utility>

using namespace M;
using namespace M::LLCL;
using namespace M::MOF;
using llvm::MemoryBuffer;

namespace {

class MOFDriverCLOptions : public DriverCLOptions,
                           public MO::GraphCompilerCLOptions,
                           public CUDA::CUDARuntimeCLOptions {
public:
  using DriverCLOptions::DriverCLOptions;

  cl::opt<bool> enableJitKernels{"enable-jit-kernels",
                                 cl::desc("Enable jitting of MOGG kernels."),
                                 cl::init(true)};
  cl::opt<std::string> customOpsPath{
      "user-ops", cl::desc("Replace MO ops with give ops"), cl::init("")};
};

class MOFContext {
public:
  /// Constructor when invoking from command line.
  MOFContext(int &argc, char **argv)
      : clOptions(std::make_unique<MOFDriverCLOptions>(argc, argv,
                                                       /*skipInitLLVM=*/true)),
        commonCasConfig{} {
    setup();
  }

  /// Constructor when invoking via API.
  MOFContext() = default;

  ~MOFContext() {
    if (clOptions)
      llvm::cl::ResetCommandLineParser();
  }

  MOFDriverCLOptions *getOptions() { return clOptions.get(); }

  ErrorOrSuccess setOptions(ArrayRef<std::string> opts) {
    if (!clOptions)
      return Error("CL Options parser is not initialized");

    // Reset all option occurrences every time this is called.
    llvm::cl::ResetAllOptionOccurrences();

    // Parse the options one at a time, so we don't have to combine the list of
    // strings.
    for (StringRef opt : opts) {
      const char *optChars[] = {clOptions->getProgramName().data(), opt.data()};
      if (!llvm::cl::ParseCommandLineOptions(std::size(optChars), optChars))
        return Error("unable to parse argument: " + opt);
    }

    setup();

    return success();
  }

  void setCasConfig(const CASConfig &casConfig) { commonCasConfig = casConfig; }

  ErrorOrSuccess setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs) {
    for (auto &config : configs) {
      if (auto customOpConfig = dyn_cast<CustomOpsDriverConfig>(config.get()))
        userOpsPath = customOpConfig->getPath();
    }
    return success();
  }

  const CASConfig &getCasConfig() { return commonCasConfig; }
  const std::optional<std::filesystem::path> &getCustomOpsPath() const {
    return userOpsPath;
  }
  const CUDA::CUDARuntimeOptions &getCUDARuntimeOptions() const {
    return cudaRuntimeOptions;
  }

  // Transfer options from this context into the runtime.
  void setRuntimeContextObjects(Runtime &runtime) {
    runtime.emplaceContextIfMissing<CUDA::CUDARuntimeOptions>(
        cudaRuntimeOptions);
  }

private:
  // Transfer options from the command line into this context (if available),
  // and establish any other global registration etc.
  void setup() {
    // Start from a blank slate in case the options have been reset.
    userOpsPath = {};
    commonCasConfig = {};
    cudaRuntimeOptions = {};

    // Transfer command line options into our POD options.
    // These options must to apply to all models to be executed for the
    // lifetime of this context.
    if (clOptions) {
      if (!clOptions->customOpsPath.empty())
        userOpsPath = clOptions->customOpsPath.getValue();
      clOptions->overrideCUDARuntimeOptions(cudaRuntimeOptions);
    }

    // Register the extra compile-time and runtime support.
    static llvm::once_flag onceFlag;
    llvm::call_once(onceFlag, []() { M::CUDA::registerCUDA(); });
  }

  /// If invoked from the command line, all the command line options relevant
  /// to this framework. Otherwise null.
  std::unique_ptr<MOFDriverCLOptions> clOptions;

  /// Collect options in their 'plain old data' representation, collected
  /// either from the command line options or set programatically.
  std::optional<std::filesystem::path> userOpsPath;
  CASConfig commonCasConfig;
  CUDA::CUDARuntimeOptions cudaRuntimeOptions;
};

} // namespace

void M_getVersionString(std::string *rev) { *rev = getModularVersionString(); }

M::ErrorOrSuccess M_initFrameworkContext(M::FrameworkContextHandle *ctx,
                                         int &argc, char **&argv) {
  ctx->ctx = new MOFContext(argc, argv);
  return success();
}

M::ErrorOrSuccess
M_initFrameworkContextWithConfig(M::FrameworkContextHandle *ctx) {
  ctx->ctx = new MOFContext();
  return success();
}

void M_freeFrameworkContext(M::FrameworkContextHandle ctx) {
  delete ctx.unwrap<MOFContext>();
}

ErrorOrSuccess M_setOptions(FrameworkContextHandle ctx,
                            ArrayRef<std::string> opts) {
  if (opts.empty())
    return success();

  return ctx.unwrap<MOFContext>()->setOptions(opts);
}

ErrorOrSuccess M_setConfigs(FrameworkContextHandle ctx,
                            ArrayRef<std::unique_ptr<DriverConfig>> configs,
                            const CASConfig &casConfig) {
  ctx.unwrap<MOFContext>()->setCasConfig(casConfig);
  return ctx.unwrap<MOFContext>()->setConfigs(configs);
}

bool M_isBinary(FrameworkContextHandle ctx, const llvm::MemoryBufferRef &file,
                FileFormat &format) {
  return false;
}

ErrorOrSuccess M_compileToBinary(FrameworkContextHandle ctxHandle,
                                 const llvm::MemoryBufferRef &file,
                                 raw_ostream &os, FileFormat &format) {
  auto *mofCtx = ctxHandle.unwrap<MOFContext>();
  // CAUTION: Options will be null for C API calls.
  auto *options = mofCtx->getOptions();
  auto casConfig = mofCtx->getCasConfig();
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  // Convert the input file to an MLIR module.
  switch (format) {
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    module = parseMODialect(file.getBuffer(), &ctx);
    break;

  default: {
    auto identityOr = getFileContentIdentity(file, format);
    if (identityOr.isError())
      return identityOr.takeError();
    return Error("cannot compile input with format " + *identityOr);
  }
  }

  // We have an MLIR model, try to lower it to a MEF-capable encoding.
  switch (format) {
  case FileFormat::kMODialect: {

    auto writeIntermediateFileIfRequested = [&](const std::string &identifier) {
      std::string ext = "." + identifier + ".mlir";
      if (options && options->saveTemps) {
        std::unique_ptr<llvm::ToolOutputFile> intermediateFile =
            options->getIntermediateFile(file.getBufferIdentifier(), ext);
        module->print(intermediateFile->os());
        intermediateFile->keep();
      }
    };

    writeIntermediateFileIfRequested("mo");

    // The MO graph dialect can be lowered to TFP, which is MEF emittable.
    MO::MoToMgpOptions moToMgpOptions;
    moToMgpOptions.compiledFrameworkLabel = "mof";
    if (options) {
      moToMgpOptions.minimumCpuAlignment.setValue(options->minimumCpuAlignment);
      moToMgpOptions.cacheDir.setValue(options->cacheDir);
      moToMgpOptions.enableExtraDevices.setValue(options->enableExtraDevices);
    }
    if (mofCtx->getCustomOpsPath())
      moToMgpOptions.extraLibPaths = mofCtx->getCustomOpsPath()->string();

    std::unique_ptr<Runtime> runtime;
    if (options) {
      // Create a runtime to use during compilation. It will convey global
      // options to the passes which need them.
      // TODO: Would prefer to just call mofCtx->run, however we have a
      // std::unique_ptr<MemoryBuffer> vs llvm::MemoryBufferRef mismatch.
      // So need to first refactor to use M::BufferRef throughout.
      runtime = options->createRuntime();
      mofCtx->setRuntimeContextObjects(*runtime);
    }

    if (failed(lowerMOToMGP(module.get(), moToMgpOptions, runtime.get())))
      return Error("unable to lower MO to MGP");
    writeIntermediateFileIfRequested("mgp");

    format = FileFormat::kMGPDialect;
    break;
  }
  case FileFormat::kMGPDialect: {
    break;
  }
  default:
    auto identityOr = getFileContentIdentity(file, format);
    if (identityOr.isError())
      return identityOr.takeError();
    auto identity = std::move(*identityOr);
    return Error("cannot lower to MEF data of format " + identity);
  }

  bool enableMEFMetadata = options ? options->enableMEFMetadata : true;
  if (failed(convertToMEF(module.get(), os, enableMEFMetadata)))
    return Error("unable to convert file to a MEF binary");
  return success();
}

ErrorOrSuccess M_getMetadataFromBinary(FrameworkContextHandle ctx,
                                       DriverHandle modelBinaryHandle,
                                       llvm::ToolOutputFile &outputFile,
                                       int command) {

  return Error("Not implemented");
}

ErrorOrSuccess M_openBinaryFile(FrameworkContextHandle ctx,
                                const llvm::MemoryBuffer &file,
                                DriverHandle *modelBinaryHandle) {
  auto mefFileOr = MEFFile::open(file.getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();
  *modelBinaryHandle = DriverHandle{std::move(*mefFileOr).release()};
  return success();
}

void M_closeBinaryFile(DriverHandle modelBinaryHandle) {
  // We're going to use RCRef to destroy the MEFFile handle for simplicity's
  // sake. We take the RCRef and then immediately destroy it, which should
  // essentially just decrement the refcount to 0 and destroy the MEFFile.
  (void)takeRCRef(modelBinaryHandle.unwrap<MEFFile>());
}

ErrorOrSuccess M_checkExpectedContents(FrameworkContextHandle ctx,
                                       DriverHandle modelBinaryHandle) {
  auto expectedFramework =
      CompiledFrameworkLabel{CompiledFrameworkLabel::kModularModel};
  auto actualFramework =
      modelBinaryHandle.unwrap<MEFFile>()->sniffFileContents();
  if (actualFramework != expectedFramework)
    return Error(Twine("expected ") + expectedFramework.getAsString() +
                 ", found " + actualFramework.getAsString() + " instead");
  return success();
}

void M_sniffBinaryFileContents(FrameworkContextHandle ctx,
                               DriverHandle modelBinaryHandle,
                               std::string &str) {
  auto mefContents = modelBinaryHandle.unwrap<MEFFile>()->sniffFileContents();
  str = std::string("input is a MEF file containing a ") +
        mefContents.getAsString();
}

ErrorOrSuccess M_loadModels(FrameworkContextHandle ctx,
                            llvm::MemoryBufferRef fileBuffer,
                            DriverHandle handle,
                            std::unique_ptr<ModelCollection> &modelsPtr) {
  auto modelsPtrOr = loadMGPMEFFile(copyRCRef(handle.unwrap<MEFFile>()));
  if (modelsPtrOr.isError())
    return modelsPtrOr.takeError();
  modelsPtr = std::move(*modelsPtrOr);
  return success();
}

int M_runBinaryFile(FrameworkContextHandle ctx,
                    std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                    DriverHandle modelBinaryHandle) {
  auto *mofCtx = ctx.unwrap<MOFContext>();
  auto *clOptions = mofCtx->getOptions();
  std::unique_ptr<ModelCollection> models;
  if (auto err = M_loadModels(ctx, fileBuffer->getMemBufferRef(),
                              modelBinaryHandle, models)) {
    return clOptions->reportError("unable to load models: " +
                                  Twine(err.getError()));
  }

  auto runModels = [mofCtx, clOptions, models = models.get()](
                       Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofCtx->setRuntimeContextObjects(runtime);
    for (const auto &model : (*models)) {
      if (failed(runModel(ctx, model.get(), runtime, *clOptions))) {
        if (clOptions->stopOnFirstError())
          return failure();
      }
    }
    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *clOptions, "Modular model", std::move(fileBuffer), runModels);
}

int M_runModelWithData(FrameworkContextHandle ctx,
                       std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                       const M::Model *model, DriverHandle inputsHandle,
                       DriverHandle outputsHandle) {
  auto *mofCtx = ctx.unwrap<MOFContext>();
  auto *clOptions = mofCtx->getOptions();
  auto inputs = inputsHandle.unwrap<llvm::SmallVector<const void *>>();
  auto outputs = outputsHandle.unwrap<llvm::SmallVector<void *>>();

  auto runModels = [mofCtx, clOptions, model, inputs, outputs](
                       Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofCtx->setRuntimeContextObjects(runtime);
    if (runModelWithData(ctx, model, runtime, *clOptions, *inputs, *outputs)
            .isError()) {
      if (clOptions->stopOnFirstError())
        return failure();
    }

    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *clOptions, "Modular model", std::move(fileBuffer), runModels);
}

ErrorOrSuccess M_openPath(M::FrameworkContextHandle ctx,
                          const std::filesystem::path &path,
                          std::unique_ptr<llvm::MemoryBuffer> &buffer) {
  llvm_unreachable("Not implemented");
}
