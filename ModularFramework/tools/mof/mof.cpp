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
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/Support/CASConfig.h"
#include "GenericML/Support/HostDeviceSpecs.h"
#include "GenericML/Support/ModelCollection.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "MOF/Driver/DriverConfig.h"
#include "MOF/Driver/DriverImpl.h"
#include "MOF/ModularModel/ModularModelFile.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/bit.h"
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
private:
  llvm::cl::OptionCategory MOFDriverCLOptionsCategory{"MOF driver options"};

public:
  using DriverCLOptions::DriverCLOptions;

  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) const {
    overrideCUDARuntimeOptions(mofLoweringOptions.cudaRuntimeOptions);
    mofLoweringOptions.debugTensorPrintOptions.style = resultOutputStyle;
    mofLoweringOptions.debugTensorPrintOptions.precision = printPrecision;
    mofLoweringOptions.debugTensorPrintOptions.binaryDir = binaryOutputDir;

    GraphCompilerCLOptions::overrideLoweringOptions(moToMgpOptions);
    moToMgpOptions.compiledFrameworkLabel = "mof";
    moToMgpOptions.minimumCpuAlignment = minimumCpuAlignment.getValue();
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
  }
};

/// Union of configuration gathered from MOFContext::setConfigs().
struct MOFConfig {
  SmallVector<std::string> kernelLibraryPaths;
  std::optional<std::string> defaultDeviceLabel;
  std::optional<bool> enableExtraDevices;
  std::optional<size_t> deviceID;
  CASConfig commonCasConfig;

  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) const {
    if (deviceID && defaultDeviceLabel && *defaultDeviceLabel == "cuda") {
      mofLoweringOptions.cudaRuntimeOptions.deviceOptions.emplace_back()
          .deviceId = *deviceID;
    }

    moToMgpOptions.compiledFrameworkLabel = "mof";
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
    if (defaultDeviceLabel)
      moToMgpOptions.defaultDeviceLabel = *defaultDeviceLabel;
    if (enableExtraDevices)
      moToMgpOptions.enableExtraDevices = *enableExtraDevices;
    if (commonCasConfig.getCacheDir())
      moToMgpOptions.cacheDir = commonCasConfig.getCacheDir()->getPath().str();
  }
};

class MOFContext {
public:
  /// Constructor when invoking from command line.
  MOFContext(int &argc, char **argv)
      : clOptions(std::make_unique<MOFDriverCLOptions>(argc, argv,
                                                       /*skipInitLLVM=*/true)) {
    setup();
  }

  /// Constructor when invoking via API.
  MOFContext() { setup(); }

  ~MOFContext() {
    if (clOptions)
      llvm::cl::ResetCommandLineParser();
  }

  const CommonCLOptions *getCommonCLOptions() const { return clOptions.get(); }
  const ExecutorDriverCLOptions *getExecutorDriverCLOptions() const {
    return clOptions.get();
  }
  bool getEnableMEFMetadata() const {
    return clOptions ? clOptions->enableMEFMetadata : true;
  }
  bool getSaveTemps() const { return clOptions ? clOptions->saveTemps : false; }

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

  ErrorOrSuccess setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs,
                            const CASConfig &casConfig) {
    if (!mofConfig)
      mofConfig = std::make_unique<MOFConfig>();
    mofConfig->commonCasConfig = casConfig;

    for (auto &driverConfig : configs) {
      if (auto kernelLibraryConfig =
              dyn_cast<KernelLibraryDriverConfig>(driverConfig.get())) {
        mofConfig->kernelLibraryPaths.push_back(
            kernelLibraryConfig->getPath().string());
      } else if (auto deviceConfig =
                     dyn_cast<DeviceConfig>(driverConfig.get())) {
        mofConfig->defaultDeviceLabel = deviceConfig->getDevice();
        mofConfig->deviceID = deviceConfig->getDeviceID();
        mofConfig->enableExtraDevices = !deviceConfig->getDevice().empty();
      }
    }
    return success();
  }

  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) {
    if (clOptions)
      clOptions->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
    if (mofConfig)
      mofConfig->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
  }

private:
  void setup() {
    // Register the extra compile-time and runtime support.
    static llvm::once_flag onceFlag;
    llvm::call_once(onceFlag, []() { M::CUDA::registerCUDA(); });
  }

  std::unique_ptr<MOFDriverCLOptions> clOptions;
  std::unique_ptr<MOFConfig> mofConfig;
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
  return ctx.unwrap<MOFContext>()->setConfigs(configs, casConfig);
}

bool M_isBinary(FrameworkContextHandle ctx, const llvm::MemoryBufferRef &file,
                FileFormat format) {
  return false;
}

ErrorOrSuccess M_compileToBinary(FrameworkContextHandle ctxHandle,
                                 const llvm::MemoryBufferRef &file,
                                 raw_ostream &os, FileFormat format) {
  auto *mofContext = ctxHandle.unwrap<MOFContext>();
  // Create a runtime to use during compilation. It will convey global options
  // to the passes which need them.
  // This runtime must outlive `ctx` because `ConstantFoldPass` creates new
  // attributes owned by `ctx` and hence must be declared before `ctx`.
  auto ownedRuntime = LLCL::createRuntimeIfNeeded();
  mlir::MLIRContext ctx;
  mlir::OwningOpRef<mlir::ModuleOp> module;

#ifdef MODULAR_PRODUCTION
  ctx.printOpOnDiagnostic(false);
#endif // MODULAR_PRODUCTION

  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  // Convert the input file to an MLIR module.
  switch (format) {
  case FileFormat::kMODialect:
    [[fallthrough]];
  case FileFormat::kMGPDialect:
    module = parseMODialect(file, &ctx);
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
    writeIntermediateFileIfRequested("mo", module.get(),
                                     mofContext->getCommonCLOptions(), file);

    mofLoweringOptions.setupRuntimeOptions(*ownedRuntime);

    if (failed(lowerMOToMGP(module.get(), moToMgpOptions, *ownedRuntime)))
      return Error("unable to lower MO to MGP");
    writeIntermediateFileIfRequested("mgp", module.get(),
                                     mofContext->getCommonCLOptions(), file);

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

  if (failed(
          convertToMEF(module.get(), os, mofContext->getEnableMEFMetadata())))
    return Error("unable to convert file to a MEF binary");
  return success();
}

ErrorOrSuccess M_compileToBinaryFromSource(FrameworkContextHandle ctxHandle,
                                           void *source, raw_ostream &os,
                                           FileFormat format) {
  auto *mofContext = ctxHandle.unwrap<MOFContext>();
  auto ownedRuntime = LLCL::createRuntimeIfNeeded();
  ModuleOp moduleOp = ModuleOp::getFromOpaquePointer(source);

#ifdef MODULAR_PRODUCTION
  moduleOp.getContext()->printOpOnDiagnostic(false);
#endif // MODULAR_PRODUCTION

  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  mofLoweringOptions.setupRuntimeOptions(*ownedRuntime);

  if (failed(lowerMOToMGP(moduleOp, moToMgpOptions, *ownedRuntime)))
    return Error("unable to lower MO to MGP");

  if (failed(convertToMEF(moduleOp, os, mofContext->getEnableMEFMetadata())))
    return Error("unable to convert file to a MEF binary");

  return success();
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
  auto *mofContext = ctx.unwrap<MOFContext>();
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
  const ExecutorDriverCLOptions *driverOptions =
      mofContext->getExecutorDriverCLOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  std::unique_ptr<ModelCollection> models;
  if (auto err = M_loadModels(ctx, fileBuffer->getMemBufferRef(),
                              modelBinaryHandle, models)) {
    return driverOptions->reportError("unable to load models: " +
                                      Twine(err.getError()));
  }

  auto runModels = [&](Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofLoweringOptions.setupRuntimeOptions(runtime);
    for (const auto &model : (*models)) {
      if (failed(runModel(ctx, model.get(), runtime, *driverOptions))) {
        if (driverOptions->stopOnFirstError())
          return failure();
      }
    }
    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *driverOptions, "Modular model", std::move(fileBuffer), runModels);
}

int M_runModelWithData(FrameworkContextHandle ctx,
                       std::unique_ptr<llvm::MemoryBuffer> &&fileBuffer,
                       const M::Model *model, DriverHandle inputsHandle,
                       DriverHandle outputsHandle) {
  auto *mofContext = ctx.unwrap<MOFContext>();
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
  const ExecutorDriverCLOptions *driverOptions =
      mofContext->getExecutorDriverCLOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  auto inputs = inputsHandle.unwrap<llvm::SmallVector<const void *>>();
  auto outputs = outputsHandle.unwrap<llvm::SmallVector<void *>>();

  auto runModels = [&](Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofLoweringOptions.setupRuntimeOptions(runtime);
    if (runModelWithData(ctx, model, runtime, *driverOptions, *inputs, *outputs)
            .isError()) {
      if (driverOptions->stopOnFirstError())
        return failure();
    }

    return success();
  };

  // Run it.
  return setupMLIRAndRunWithLeakCheckedRuntime(
      *driverOptions, "Modular model", std::move(fileBuffer), runModels);
}

ErrorOrSuccess M_openPath(M::FrameworkContextHandle ctx,
                          const std::filesystem::path &path,
                          std::unique_ptr<llvm::MemoryBuffer> &buffer) {
  llvm_unreachable("Not implemented");
}
