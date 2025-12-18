//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/FrameworkFrontend.h"
#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "Config/Version.h"
#include "GenericML/CommonDriver/DriverConfig.h"
#include "GenericML/CommonDriver/DriverInputFormat.h"
#include "GenericML/CommonDriver/Utils.h"
#include "GenericML/CompileDriver/CompileDriver.h"
#include "GenericML/CompileDriver/CompileToBinary.h"
#include "GenericML/ExecutionDriver/DriverCLOptions.h"
#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENUtils.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GenericML/GraphCompiler/Version.h"
#include "GenericML/MEFModel/MEFModel.h"
#include "GenericML/MOFOptions.h"
#include "GenericML/Support/ModelCollection.h"
#include "GenericML/Support/Telemetry.h"
#include "KGEN/KGENDialect/KGENCompilationContext.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "MLRT/AsyncRT/Runtime/Utils.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/DeviceSpecs.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/Filesystem/Paths.h"
#include "Support/ML/CompiledFrameworkLabel.h"
#include "max/MOF/Driver/DriverImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"

#include <string>
#include <utility>

using namespace M;
using namespace M::AsyncRT;
using namespace M::MOF;

namespace {

class MOFDriverOptions : public DriverOptions, public MO::GraphCompilerOptions {
public:
  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) const {
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
                           public MO::GraphCompilerCLOptions {
private:
  llvm::cl::OptionCategory MOFDriverCLOptionsCategory{"MOF driver options"};

public:
  using DriverCLOptions::DriverCLOptions;
  MOFDriverCLOptions(int argc, char **argv, MOFDriverOptions &o,
                     bool skipInitLLVM = false)
      : DriverCLOptions(argc, argv, o, skipInitLLVM),
        MO::GraphCompilerCLOptions(o) {}
};

/// Union of configuration gathered from MOFContext::setConfigs().
struct MOFConfig {
  SmallVector<std::string> kernelLibraryPaths;
  std::optional<std::string> defaultDeviceLabel;
  std::optional<size_t> deviceID;
  std::optional<DeviceSpec> deviceSpec;

  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::MoToMgpOptions &moToMgpOptions) const {
    moToMgpOptions.compiledFrameworkLabel =
        CompiledFrameworkLabel::asLabelString(
            CompiledFrameworkLabel::kModularModel);
    moToMgpOptions.kernelLibraryPaths = kernelLibraryPaths;
    if (defaultDeviceLabel) {
      moToMgpOptions.defaultDeviceLabel = *defaultDeviceLabel;
      if (deviceID)
        moToMgpOptions.defaultDeviceId = *deviceID;
    }
    moToMgpOptions.defaultDeviceTargetTriple =
        deviceSpec ? deviceSpec.value().target.triple.str() : "";
    moToMgpOptions.defaultDeviceTargetArch =
        deviceSpec ? deviceSpec.value().target.arch : "";
  }
};

} // namespace

struct MOFContext::Impl final {
  ContextRef maxContext;
  std::unique_ptr<MOFDriverOptions> clOptions;
  std::unique_ptr<MOFDriverCLOptions> parser;
  std::unique_ptr<MOFConfig> mofConfig;

  const CommonOptions *getCommonCLOptions() const { return clOptions.get(); }

  bool getEnableMEFMetadata() const {
    return clOptions ? clOptions->enableMEFMetadata : true;
  }
  bool getSaveTemps() const { return clOptions ? clOptions->saveTemps : false; }
};

/// Constructor when invoking from command line.
MOFContext::MOFContext(ContextRef maxContext, int &argc, char **&argv)
    : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
  impl->clOptions = std::make_unique<MOFDriverOptions>();
  impl->parser =
      std::make_unique<MOFDriverCLOptions>(argc, argv, *impl->clOptions,
                                           /*skipInitLLVM=*/true);
}

/// Constructor when invoking via API.
MOFContext::MOFContext(ContextRef maxContext) : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
}

MOFContext::~MOFContext() {
  if (impl->clOptions)
    llvm::cl::ResetCommandLineParser();
}

const ExecutorDriverOptions *MOFContext::getExecutorDriverOptions() const {
  return impl->clOptions.get();
}

ErrorOrSuccess MOFContext::setOptions(ArrayRef<std::string> opts) {
  if (opts.empty())
    return success();

  if (!impl->clOptions)
    return Error("CL Options parser is not initialized");

  // Reset all option occurrences every time this is called.
  llvm::cl::ResetAllOptionOccurrences();

  // Parse the options one at a time, so we don't have to combine the list
  // of strings.
  for (StringRef opt : opts) {
    const char *optChars[] = {impl->clOptions->getProgramName().data(),
                              opt.data()};
    if (!llvm::cl::ParseCommandLineOptions(std::size(optChars), optChars))
      return Error("unable to parse argument: " + opt);
  }

  return success();
}

ErrorOrSuccess
MOFContext::setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs) {
  auto &mofConfig = impl->mofConfig;
  if (!mofConfig)
    mofConfig = std::make_unique<MOFConfig>();

  for (auto &driverConfig : configs) {
    if (auto kernelLibraryConfig =
            dyn_cast<KernelLibraryDriverConfig>(driverConfig.get())) {
      mofConfig->kernelLibraryPaths.push_back(
          kernelLibraryConfig->getPath().string());
    } else if (auto deviceConfig = dyn_cast<DeviceConfig>(driverConfig.get())) {
      mofConfig->defaultDeviceLabel = deviceConfig->getDeviceLabel();
      mofConfig->deviceID = deviceConfig->getDeviceID();
      mofConfig->deviceSpec = deviceConfig->deviceSpec;
    }
  }
  return success();
}

void MOFContext::overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                                         MO::MoToMgpOptions &moToMgpOptions) {
  if (impl->clOptions)
    impl->clOptions->overrideLoweringOptions(mofLoweringOptions,
                                             moToMgpOptions);
  if (impl->mofConfig)
    impl->mofConfig->overrideLoweringOptions(mofLoweringOptions,
                                             moToMgpOptions);
}

/// A diagnostic handler for the graph api.
/// It uses the graph api python location information to print a python like
/// stacktrace. It also only prints the first error diagnostic (as that would be
/// equivalent to the first python exception).
///
/// If a python stacktrace does not exist, falls back to the
/// FormatScopedDiagnosticHandler.
class GraphApiDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  GraphApiDiagnosticHandler(MLIRContext *ctx)
      : ScopedDiagnosticHandler(ctx, [&](Diagnostic &diag) {
          diagnostics.push_back(std::move(diag));
        }) {}

  std::string formatMessage() const {
    std::string message;
    llvm::raw_string_ostream messageStream(message);
    emitDiagnosticToStream(messageStream);
    return message;
  }

private:
  // The full set of diagnostics printed on fallback.
  std::vector<mlir::Diagnostic> diagnostics;

  void emitDiagnosticToStream(raw_ostream &os) const {
    auto errorDiagnostics =
        llvm::make_filter_range(diagnostics, [](const Diagnostic &diag) {
          return diag.getSeverity() == mlir::DiagnosticSeverity::Error;
        });
    if (errorDiagnostics.empty()) {
      // Still use fallback diagnostics when there are no errors.
      emitFallbackDiagnosticsToStream(os);
      return;
    }

    for (const Diagnostic &diag : errorDiagnostics) {
      // Try to get the Python stack trace if available.
      //
      // Python Stack Trace Capture Flow:
      // 1. When a Graph operation is created in Python (see graph.py:_add_op
      //    method), the _location() method creates source location info:
      //    - Uses traceback.extract_stack() to get the current Python call
      //    stack.
      //    - Trims the _location() and _add_op() frames to get user code.
      //    - The remaining stack frames show where the user called the graph
      //    API.
      //
      // 2. These stack frames are passed via _graph.frame_loc() to the C++
      //    binding (defined in SDK/include/SDK/GraphAPI/python/Bindings.h)
      //    which:
      //    - Increments the Python frame object reference count (inc_ref()).
      //    - Calls MAXG_opaqueFrameLocNew with the frame pointer and callbacks.
      //    - Registers frame_traceback() to format Python stack frames as
      //      strings.
      //    - Registers frame_decref() to release the Python reference when
      //      done.
      //
      // 3. When an error occurs, this diagnostic handler retrieves the stack.
      if (auto attr = RMO::PythonFrameAttr::fromLocation(diag.getLocation())) {
        os << "\nBuild Time Traceback (most recent call last):\n"
           // attr.stack_trace() calls the frame_traceback() function that
           // uses Python's traceback.format_list to convert the frames to a
           // human-readable form.
           << attr.stack_trace() << "\nMAX Compile Error: " << diag << "\n";
      } else {
        // First emit the formatted diag itself before processing notes.
        FormatScopedDiagnosticHandler::emitDiagLocSeverity(os, diag);
      }

      for (const Diagnostic &note : diag.getNotes()) {
        // Skip printing the operation to avoid dumping lots of IR.
        // Do so by inspecting the first diagnostic argument: typically the
        // message.
        if (note.getArguments().empty() ||
            note.getArguments().front().getKind() !=
                mlir::DiagnosticArgument::DiagnosticArgumentKind::String ||
            !note.getArguments().front().getAsString().starts_with(
                "see current operation")) {
          FormatScopedDiagnosticHandler::emitDiagnosticToStream(os, note);
        }
      }
    }
  }

  // If there is no python stacktrace, fallback to the
  // FormatScopedDiagnosticHandler.
  void emitFallbackDiagnosticsToStream(raw_ostream &os) const {
    int pid = llvm::sys::Process::getProcessId();
    std::string fileName = "max-full-diagnostics-" + Twine(pid).str();

    auto diagFile = TempFile::create(fileName);
    if (diagFile.isError()) {
      os << "Error creating file for full diagnostics (" << fileName
         << "): " << diagFile.takeError() << "\n"
         << "All diagnostics will be emitted to the console.\n";
      for (auto &diag : diagnostics)
        FormatScopedDiagnosticHandler::emitDiagnosticToStream(os, diag);
      return;
    }

    // Emit minimal output to the console, but capture everything (which might
    // be a lot!) in a file.
    diagFile->keep();
    // Assume we can open the file stream because we created the file.
    std::error_code unused;
    llvm::raw_fd_ostream fullOutputStream(diagFile->getPath().c_str(), unused);

    std::string previousDiagnostic;
    for (auto &diag : diagnostics) {
      std::string currentDiagnostic;
      llvm::raw_string_ostream currentDiagnosticStream(currentDiagnostic);
      FormatScopedDiagnosticHandler::emitDiagnosticToStream(
          currentDiagnosticStream, fullOutputStream, diag);

      // We sometimes see duplicate diagnostics, so skip them.
      if (currentDiagnostic == previousDiagnostic)
        continue;
      previousDiagnostic = currentDiagnostic;
      os << currentDiagnostic << "\n";
    }

    os << "================================================\n";
    os << "Elided diagnostics have been emitted to "
       << diagFile->getPath().c_str() << "\n";
    os << "================================================\n";
  }
};

/// Lowers a module op to MGP then exports it to MEF.
static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,
                                         mlir::PassManager &moToMgpPasses,
                                         bool enableMEFMetadata) {
  GraphApiDiagnosticHandler diagnostics(moToMgpPasses.getContext());
  if (failed(moToMgpPasses.run(moduleOp))) {
    return Error(Twine("Failed to run the MOToMGP pass manager:\n") +
                 diagnostics.formatMessage());
  }

  // Save MGP IR pre export to MEF.
  writeIntermediateFileIfRequested("mgp", moduleOp, /*options=*/nullptr,
                                   BufferRef());

  auto bufOr = MO::convertMGPToMEF(moduleOp, enableMEFMetadata);
  if (bufOr)
    return bufOr.takeError();
  return bufOr.takeValue();
}

/// Lowers a module op to MGP then exports to MEF.
static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,
                                         const MO::MoToMgpOptions &options,
                                         bool enableMEFMetadata) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(options.kernelLibraryPaths);

  auto libraryAnalysis = MOGG::MojoLibraryAnalysis::fromPaths(
      moduleOp.getContext(), kernelLibraryPaths);
  if (failed(libraryAnalysis))
    return libraryAnalysis.takeError();

  auto moToMgpPasses = mlir::PassManager::on<ModuleOp>(
      moduleOp.getContext(), mlir::OpPassManager::Nesting::Implicit);
  MO::buildMoToMgpPipeline(moToMgpPasses, options, libraryAnalysis.takeValue());

  return convertMOToMEF(moduleOp, moToMgpPasses, enableMEFMetadata);
}

/// Get the textual key used to represent the compilation pipeline.
static void getMOToMGPPipelineKey(llvm::raw_ostream &os, MLIRContext *ctx,
                                  const MO::MoToMgpOptions &options) {
  auto moToMgpPasses = mlir::PassManager::on<ModuleOp>(
      ctx, mlir::OpPassManager::Nesting::Implicit);
  MO::buildMoToMgpPipeline(moToMgpPasses, options);
  moToMgpPasses.printAsTextualPipeline(os);
}

M::ErrorOr<M::BufferRef> MOFContext::compileToBinary(void *source,
                                                     FileFormat format) {
  // Clone the ModuleOp so compilation doesn't have side effects on the input
  // op. Cloning should be reasonably cheap since all model weights are stored
  // as dense_resource attributes owned by the MLIRContext.
  OwningOpRef<ModuleOp> moduleOp =
      ModuleOp::getFromOpaquePointer(source).clone();

  auto &maxContext = impl->maxContext;
  MLIRContext *mlirCtx = moduleOp->getContext();
  registerContext(*mlirCtx, maxContext);

  auto telemetryContext = maxContext->get<GML::MaxEngineTelemetryContext>();

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
  overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  mofLoweringOptions.setupContext(maxContext.copy());
  bool enableMEFMetadata = impl->getEnableMEFMetadata();

  // Enable saving intermediate IR by setting `max.temps_dir` in modular.cfg.
  // Save MO IR pre mo-to-mgp.
  writeIntermediateFileIfRequested("mo", *moduleOp, /*options=*/nullptr,
                                   BufferRef());

  auto baseCacheDirOr = M::getMaxCacheDir();

  // Disable caching if set or we can't find any suitable location to cache.
  if (!isModelIRCacheEnabled() || !baseCacheDirOr)
    return convertMOToMEF(*moduleOp, moToMgpOptions, enableMEFMetadata);

  // Create the MEF cache filesystem backend.
  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;
  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / "mof" / "mef").string(),
      std::string(M::getModularVersionString()));

  if (mefBackendOr.isError()) {
    // We couldn't create a cache, so don't cache.
    return convertMOToMEF(*moduleOp, moToMgpOptions, enableMEFMetadata);
  }

  // Write the MAX Graph MEF cache key:
  // - The pass pipeline as a list of pass names.
  // - Graph compiler options that affect lowering and export.
  // - The model IR, by writing the ModuleOp.
  //   This does NOT include the model weights
  // - The graph compiler version string
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  mofLoweringOptions.print(*keyBuf);
  moToMgpOptions.print(*keyBuf);
  getMOToMGPPipelineKey(*keyBuf, mlirCtx, moToMgpOptions);

  if (auto result = M::MOGG::graphCompilerCacheKey(*keyBuf, mlirCtx))
    return result.takeError();

  // We include the binary contents of any extra kernel mojo packages provided.
  // These paths have already been validated by MOToMGP pass pipeline.
  for (const std::string &libraryPath : moToMgpOptions.kernelLibraryPaths) {
    // TODO(MOCO-1570): If mojopkg files had a hash key as part of its format,
    // this wouldn't be needed. We would have to detect the file is a mojopkg
    // and act accordingly for this.
    auto bufferOr = M::openInputFile(libraryPath);

    if (bufferOr.isError())
      return Error(
          Twine(
              "failed to open file specified as additional kernel library: ") +
          libraryPath);

    *keyBuf << (*bufferOr)->getBuffer();
  }

  // We specifically use a version of the module without location information
  // for caching. Otherwise, python refactors that don't change the module would
  // invalidate the cache.
  // This means that runtime errors may print an old location.
  // This is preferred to constantly having cache misses.
  auto stripDebugInfo = mlir::PassManager::on<ModuleOp>(
      mlirCtx, mlir::OpPassManager::Nesting::Implicit);
  stripDebugInfo.addPass(mlir::createStripDebugInfoPass());
  OwningOpRef<ModuleOp> moduleOpNoDebug = moduleOp->clone();
  if (failed(stripDebugInfo.run(*moduleOpNoDebug)))
    return Error::getStaticString(
        "failed to strip location information from the module op");

  MOGG::normalizeAllNames(*moduleOpNoDebug);
  if (failed(Cache::writeOperationToCacheKey(*moduleOpNoDebug, keyBuf)))
    return Error::getStaticString("failed to write module op to cache key");

  // Look up the model in the MEF cache.
  auto mefCache = RCRef<Cache::TransformCache>::create(*mefBackendOr);
  std::string keyBufHash;
  auto maybeMefOr = mefCache->findSync(std::move(keyBuf), &keyBufHash);
  if (maybeMefOr.isError())
    return maybeMefOr.takeError();
  std::optional<BufferRef> maybeMef = maybeMefOr.takeValue();
  if (maybeMef) {
    // Report telemetry on MEF cache hit.
    auto onHitLambda = Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
        "mof::M_compileToBinaryFromSource");
    onHitLambda(*moduleOp);

    // On cache hit, directly return the MEF buffer.
    return std::move(*maybeMef);
  }

  // Otherwise, on cache miss, lower and export to MEF.
  // First, write telemetry data indicating a MEF cache miss for this model.
  auto onMissLambda = Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
      "mof::M_compileToBinaryFromSource",
      "mof.compiletobinaryfromsource.cache.miss.time",
      {{"pipeline", "mof::M_compileToBinaryFromSource"}});
  onMissLambda(*moduleOp);

  auto outBufOr = convertMOToMEF(*moduleOp, moToMgpOptions, enableMEFMetadata);
  if (outBufOr.isError())
    return outBufOr.takeError();
  BufferRef outputBuffer = outBufOr.takeValue();

  // Insert MEF buffer to cache, ignoring failure to insert.
  (void)mefCache->insertKeyedSync(std::move(keyBufHash), outputBuffer.copy());

  return outputBuffer;
}

ErrorOr<M::BufferRef> MOFContext::compileToBinary(M::BufferRef file,
                                                  FileFormat format) {
  CompileToBinaryCtx compileCtx(impl->maxContext.copy());
  auto telemetryContext =
      impl->maxContext->get<GML::MaxEngineTelemetryContext>();
  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::MoToMgpOptions moToMgpOptions;
  overrideLoweringOptions(mofLoweringOptions, moToMgpOptions);

  MOGG::GraphCompilerCacheKey graphCompilerOptions{&compileCtx.mlirCtx};

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

    // Emit telemetry upon any lowering error from here on.
    GML::MaxEngineTelemetryScope<GML::MaxEngineErrorCode::GraphLoweringError>
        loweringDiagScope(&compileCtx.mlirCtx, telemetryContext);

    // We have an MLIR model, try to lower it to a MEF-capable encoding.
    switch (format) {
    case FileFormat::kMODialect: {
      writeIntermediateFileIfRequested("mo", compileCtx.module.get(),
                                       impl->getCommonCLOptions(), file.copy());

      mofLoweringOptions.setupContext(compileCtx.context.copy());

      auto kernelLibraryPaths =
          llvm::to_vector_of<std::string>(moToMgpOptions.kernelLibraryPaths);

      auto libraryAnalysis = MOGG::MojoLibraryAnalysis::fromPaths(
          &compileCtx.mlirCtx, kernelLibraryPaths);
      if (failed(libraryAnalysis))
        return libraryAnalysis.takeError();

      GraphApiDiagnosticHandler diagnostics(&compileCtx.mlirCtx);
      if (failed(lowerMOToMGP(compileCtx.module.get(), moToMgpOptions)))
        return Error(Twine("unable to lower MO to MGP") +
                     diagnostics.formatMessage());
      writeIntermediateFileIfRequested("mgp", compileCtx.module.get(),
                                       impl->getCommonCLOptions(), file.copy());

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
                               impl->getEnableMEFMetadata());
  };

  // If kernel paths are not empty then we'd need to inspect them to get the
  // key, so default to not caching at this time.
  if (!moToMgpOptions.kernelLibraryPaths.empty() || !isModelIRCacheEnabled()) {
    return loweringTransform();
  }

  WriteableBufferRef keyBuffer =
      makeCacheKey(std::move(file), mofLoweringOptions, moToMgpOptions,
                   graphCompilerOptions);

  return M::cacheMEFTransform("mof", loweringTransform, std::move(keyBuffer));
}
