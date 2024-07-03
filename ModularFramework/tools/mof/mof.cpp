//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CUDASupport/CUDARuntimeOptions.h"
#include "Cache/Support/Keys.h"
#include "Config/Version.h"
#include "GenericML/CUDARegistration/CUDARegistration.h"
#include "GenericML/Driver/CompileDriver.h"
#include "GenericML/Driver/CompileToBinary.h"
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
#include "GenericML/Support/Telemetry.h"
#include "GraphRT/MEFExecutor/MEFFile.h"
#include "LLCL/CompilerSupport/Context.h"
#include "MOF/Driver/DriverConfig.h"
#include "MOF/Driver/DriverImpl.h"
#include "MOF/ModularModel/ModularModelFile.h"
#include "Support/ErrorOr.h"
#include "Support/ML/CompiledFrameworkLabel.h"
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

class MOFDriverOptions : public DriverOptions,
                         public MO::GraphCompilerOptions,
                         public CUDA::CUDARuntimeOptions {
public:
  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) const {
    overrideCUDARuntimeOptions(mofLoweringOptions.cudaRuntimeOptions);
    mofLoweringOptions.cudaRuntimeOptions.cudaDriverPath = cudaDriverPath;
    mofLoweringOptions.cudaRuntimeOptions.deviceOptions = deviceOptions;
    mofLoweringOptions.debugTensorPrintOptions.style = resultOutputStyle;
    mofLoweringOptions.debugTensorPrintOptions.precision = printPrecision;
    mofLoweringOptions.debugTensorPrintOptions.binaryDir = binaryOutputDir;

    GraphCompilerOptions::overrideLoweringOptions(moToMgpOptions);
    moToMgpOptions.compiledFrameworkLabel =
        CompiledFrameworkLabel::asLabelString(
            CompiledFrameworkLabel::kModularModel);
    moToMgpOptions.minimumCpuAlignment = minimumCpuAlignment;
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
  }
};

class MOFDriverCLOptions : public DriverCLOptions,
                           public MO::GraphCompilerCLOptions,
                           public CUDA::CUDARuntimeCLOptions {
private:
  llvm::cl::OptionCategory MOFDriverCLOptionsCategory{"MOF driver options"};

public:
  using DriverCLOptions::DriverCLOptions;
  MOFDriverCLOptions(int argc, char **argv, MOFDriverOptions &o,
                     bool skipInitLLVM = false)
      : DriverCLOptions(argc, argv, o, skipInitLLVM),
        MO::GraphCompilerCLOptions(o), CUDA::CUDARuntimeCLOptions(o) {}
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

    moToMgpOptions.compiledFrameworkLabel =
        CompiledFrameworkLabel::asLabelString(
            CompiledFrameworkLabel::kModularModel);
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
      : clOptions(std::make_unique<MOFDriverOptions>()),
        parser(std::make_unique<MOFDriverCLOptions>(argc, argv, *clOptions,
                                                    /*skipInitLLVM=*/true)) {
    setup();
  }

  /// Constructor when invoking via API.
  MOFContext() { setup(); }

  ~MOFContext() {
    if (clOptions)
      llvm::cl::ResetCommandLineParser();
  }

  const CommonOptions *getCommonCLOptions() const { return clOptions.get(); }
  const ExecutorDriverOptions *getExecutorDriverOptions() const {
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

    // Parse the options one at a time, so we don't have to combine the list
    // of strings.
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
        mofConfig->defaultDeviceLabel = deviceConfig->getDeviceLabel();
        mofConfig->deviceID = deviceConfig->getDeviceID();
        mofConfig->enableExtraDevices = !deviceConfig->getDeviceLabel().empty();
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

  std::unique_ptr<MOFDriverOptions> clOptions;
  std::unique_ptr<MOFDriverCLOptions> parser;
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

void M_freeFrameworkContext(M::FrameworkContextHandle *ctx) {
  assert(ctx && "null framework context");
  delete ctx->unwrap<MOFContext>();
  ctx->ctx = nullptr;
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

bool M_isBinary(FrameworkContextHandle ctx, M::BufferRef file,
                FileFormat format) {
  return false;
}

M::ErrorOr<M::Buffer *>
M_compileToBinaryFromSource(FrameworkContextHandle ctxHandle, void *source,
                            FileFormat format) {
  auto *mofContext = ctxHandle.unwrap<MOFContext>();

  // Clone the ModuleOp so compilation doesn't have side effects on the input
  // op. Cloning should be reasonably cheap since all model weights are stored
  // as dense_resource attributes owned by the MLIRContext.
  OwningOpRef<ModuleOp> moduleOp =
      ModuleOp::getFromOpaquePointer(source).clone();

  MLIRContext *mlirCtx = moduleOp->getContext();
  registerContext(*mlirCtx, ctxHandle.maxContext);

  auto telemetryContext =
      ctxHandle.maxContext->get<GML::MaxEngineTelemetryContext>();

  // Install a source manager diagnostic handler to properly display MLIR
  // diagnostics that can originate from downstream compilation errors. Do this
  // *before* the telemetry diagnostic handler.
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, mlirCtx);

  // Emit telemetry upon any lowering error.
  GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
      loweringDiagScope(mlirCtx, telemetryContext);

  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  mofLoweringOptions.setupContext(ctxHandle.maxContext.copy());

  // Enable saving intermediate IR by setting `max.temps_dir` in modular.cfg.
  // Save MO IR pre mo-to-mgp.
  writeIntermediateFileIfRequested("mo", *moduleOp, /*options=*/nullptr,
                                   BufferRef());

  if (failed(lowerMOToMGP(*moduleOp, moToMgpOptions)))
    return Error("unable to lower MO to MGP");

  // Save MGP IR pre export to MEF.
  writeIntermediateFileIfRequested("mgp", *moduleOp, /*options=*/nullptr,
                                   BufferRef());

  auto bufOr =
      MO::convertMGPToMEF(*moduleOp, mofContext->getEnableMEFMetadata());
  if (bufOr)
    return bufOr.takeError();
  M::BufferRef outputBuffer = bufOr.takeValue();
  return outputBuffer.release();
}

ErrorOrSuccess M_openBinaryFile(FrameworkContextHandle ctx, M::BufferRef file,
                                DriverHandle *modelBinaryHandle) {
  auto mefFileOr = MEFFile::open(file->getBuffer());
  if (mefFileOr.isError())
    return mefFileOr.takeError();
  *modelBinaryHandle = DriverHandle{std::move(*mefFileOr).release()};
  return success();
}

ErrorOr<M::Buffer *> M_compileToBinary(FrameworkContextHandle ctxHandle,
                                       M::BufferRef file, FileFormat format) {
  auto *mofContext = ctxHandle.unwrap<MOFContext>();
  CompileToBinaryCtx compileCtx(ctxHandle.maxContext.copy());
  auto telemetryContext =
      ctxHandle.maxContext->get<GML::MaxEngineTelemetryContext>();
  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  auto loweringTransform =
      [&, file = file.copy()]() mutable -> ErrorOr<BufferRef> {
    // Convert the input file to an MLIR module.
    switch (format) {
    case FileFormat::kMODialect:
      [[fallthrough]];
    case FileFormat::kMGPDialect:
      compileCtx.module = parseMODialect(file.copy(), &compileCtx.mlirCtx);
      break;

    default: {
      auto identityOr = getFileContentIdentity(file.copy(), format);
      if (identityOr.isError())
        return identityOr.takeError();
      return Error("cannot compile input with format " + *identityOr);
    }
    }

    // Emit telemetry upon any lowering error from hereon.
    GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
        loweringDiagScope(&compileCtx.mlirCtx, telemetryContext);

    // We have an MLIR model, try to lower it to a MEF-capable encoding.
    switch (format) {
    case FileFormat::kMODialect: {
      writeIntermediateFileIfRequested("mo", compileCtx.module.get(),
                                       mofContext->getCommonCLOptions(),
                                       file.copy());

      mofLoweringOptions.setupContext(compileCtx.context.copy());

      if (failed(lowerMOToMGP(compileCtx.module.get(), moToMgpOptions)))
        return Error("unable to lower MO to MGP");
      writeIntermediateFileIfRequested("mgp", compileCtx.module.get(),
                                       mofContext->getCommonCLOptions(),
                                       file.copy());

      format = FileFormat::kMGPDialect;
      break;
    }
    case FileFormat::kMGPDialect: {
      break;
    }
    default:
      auto identityOr = getFileContentIdentity(file.copy(), format);
      if (identityOr.isError())
        return identityOr.takeError();
      auto identity = std::move(*identityOr);
      return Error("cannot lower to MEF data of format " + identity);
    }

    return MO::convertMGPToMEF(compileCtx.module.get(),
                               mofContext->getEnableMEFMetadata());
  };

  // If kernel paths are not empty then we'd need to inspect them to get the
  // key, so default to not caching at this time.
  if (!moToMgpOptions.kernelLibraryPaths.empty() || !isModelIRCacheEnabled()) {
    auto bufOr = loweringTransform();
    if (bufOr)
      return bufOr.takeError();
    M::BufferRef outputBuffer = bufOr.takeValue();
    return outputBuffer.release();
  }

  WriteableBufferRef keyBuffer =
      makeCacheKey(std::move(file), mofLoweringOptions, moToMgpOptions);

  auto bufOr =
      M::cacheMEFTransform("mof", loweringTransform, std::move(keyBuffer));
  if (bufOr.isError())
    return bufOr.takeError();
  M::BufferRef outputBuffer = bufOr.takeValue();
  return outputBuffer.release();
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

ErrorOrSuccess M_loadModels(FrameworkContextHandle ctx, M::BufferRef fileBuffer,
                            DriverHandle handle,
                            std::unique_ptr<ModelCollection> &modelsPtr) {
  auto modelsPtrOr = loadMGPMEFFile(copyRCRef(handle.unwrap<MEFFile>()));
  if (modelsPtrOr.isError())
    return modelsPtrOr.takeError();
  modelsPtr = std::move(*modelsPtrOr);
  return success();
}

int M_runBinaryFile(FrameworkContextHandle ctx, M::BufferRef fileBuffer,
                    DriverHandle modelBinaryHandle) {
  auto *mofContext = ctx.unwrap<MOFContext>();
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  mofContext->overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);
  const ExecutorDriverOptions *driverOptions =
      mofContext->getExecutorDriverOptions();
  assert(driverOptions && "cannot run a binary file via the API");

  std::unique_ptr<ModelCollection> models;
  if (auto err =
          M_loadModels(ctx, fileBuffer.copy(), modelBinaryHandle, models)) {
    return driverOptions->reportError("unable to load models: " +
                                      Twine(err.getError()));
  }

  auto runModels = [&, context = ctx.maxContext.copy()](
                       Runtime &runtime, MLIRContext *ctx) -> LogicalResult {
    mofLoweringOptions.setupContext(context);
    for (const auto &model : (*models)) {
      if (failed(runModel(ctx, model.get(), runtime, *driverOptions))) {
        if (driverOptions->stopOnFirstError())
          return failure();
      }
    }
    return success();
  };

  // Run it.
  std::unique_ptr<llvm::MemoryBuffer> aliasedBuffer =
      llvm::MemoryBuffer::getMemBuffer(fileBuffer->getMemBufferRef(),
                                       /*RequiresNullTerminator=*/
                                       false);

  return setupMLIRAndRun(ctx.maxContext.copy(), *driverOptions, "Modular model",
                         std::move(aliasedBuffer), runModels);
}

M::ErrorOr<M::Buffer *> M_openPath(FrameworkContextHandle ctx,
                                   const std::filesystem::path &path) {
  llvm_unreachable("Not implemented");
}
