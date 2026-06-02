//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/FrameworkFrontend.h"
#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "Config/Version.h"
#include "GraphCompiler/CommonDriver/DriverConfig.h"
#include "GraphCompiler/CommonDriver/DriverInputFormat.h"
#include "GraphCompiler/CommonDriver/Utils.h"
#include "GraphCompiler/CompileDriver/CompileDriver.h"
#include "GraphCompiler/CompileDriver/CompileToBinary.h"
#include "GraphCompiler/ExecutionDriver/DriverCLOptions.h"
#include "GraphCompiler/GraphCompiler/GraphCompilerCLOptions.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/KernelDeclAdaptor.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Support/ParametersUtils.h"
#include "GraphCompiler/GraphCompiler/Passes.h"
#include "GraphCompiler/GraphCompiler/RMODialect/RMOAttrs.h"
#include "GraphCompiler/GraphCompiler/Version.h"
#include "GraphCompiler/MEFCacheKey.h"
#include "GraphCompiler/MOFOptions.h"
#include "GraphCompiler/Support/Telemetry.h"
#include "KGEN/KGENDialect/KGENCompilationContext.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/CPUDevice.h"
#include "MLRT/AsyncRT/Runtime/Utils.h"
#include "MLRT/GraphRT/MEFExecutor/MEFFile.h"
#include "Support/Configuration.h"
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
using namespace M::MLRT;
using namespace M::MOF;

namespace {

class MOFDriverOptions : public DriverOptions,
                         public MO::GraphCompilerOptions {};

/// Union of configuration gathered from MOFContext::setConfigs().
struct MOFConfig {
  SmallVector<std::string> kernelLibraryPaths;
  std::optional<std::string> defaultDeviceLabel;
  std::optional<size_t> deviceID;
  std::optional<DeviceSpec> deviceSpec;

  void overrideLoweringOptions(MOFLoweringOptions &mofLoweringOptions,
                               MO::GCPipelineOptions &gcPipelineOptions) const {
    gcPipelineOptions.compiledFrameworkLabel =
        CompiledFrameworkLabel::asLabelString(
            CompiledFrameworkLabel::kModularModel);
    gcPipelineOptions.kernelLibraryPaths = kernelLibraryPaths;
    if (defaultDeviceLabel) {
      gcPipelineOptions.defaultDeviceLabel = *defaultDeviceLabel;
      if (deviceID)
        gcPipelineOptions.defaultDeviceId = *deviceID;
    }
    gcPipelineOptions.defaultDeviceTargetTriple =
        deviceSpec ? deviceSpec.value().target.triple.str() : "";
    gcPipelineOptions.defaultDeviceTargetArch =
        deviceSpec ? deviceSpec.value().target.arch : "";
  }
};

} // namespace

struct MOFContext::Impl final {
  ContextRef maxContext;
  std::unique_ptr<MOFDriverOptions> clOptions;
  const DriverOptions *driverOptions{nullptr};
  std::unique_ptr<MO::GraphCompilerCLOptions> graphCompilerParser;
  std::unique_ptr<MOFConfig> mofConfig;
  bool optionsParsed{false};

  const CommonOptions *getCommonCLOptions() const {
    return driverOptions ? static_cast<const CommonOptions *>(driverOptions)
                         : clOptions.get();
  }

  bool getEnableMEFMetadata() const {
    return driverOptions ? driverOptions->enableMEFMetadata
                         : (clOptions ? clOptions->enableMEFMetadata : true);
  }
};

/// Constructor when invoking from command line.
MOFContext::MOFContext(ContextRef maxContext, int &argc, char **&argv)
    : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
  impl->clOptions = std::make_unique<MOFDriverOptions>();
  impl->driverOptions = impl->clOptions.get();
  impl->clOptions->programName = argv[0];
  (void)argc;
  impl->graphCompilerParser =
      std::make_unique<MO::GraphCompilerCLOptions>(*impl->clOptions);
}

/// Constructor when invoking via API.
MOFContext::MOFContext(ContextRef maxContext) : impl(std::make_unique<Impl>()) {
  impl->maxContext = std::move(maxContext);
}

MOFContext::~MOFContext() = default;

const ExecutorDriverOptions *MOFContext::getExecutorDriverOptions() const {
  return impl->driverOptions
             ? static_cast<const ExecutorDriverOptions *>(impl->driverOptions)
             : impl->clOptions.get();
}

void MOFContext::setDriverOptions(const DriverOptions &options) {
  impl->driverOptions = &options;
}

ErrorOrSuccess MOFContext::setOptions(ArrayRef<std::string> opts) {
  if (opts.empty())
    return success();

  if (!impl->clOptions)
    return Error("CL Options parser is not initialized");

  if (impl->optionsParsed)
    return success();

  // Parse the options one at a time, so we don't have to combine the list
  // of strings.
  for (StringRef opt : opts) {
    const char *optChars[] = {impl->clOptions->getProgramName().data(),
                              opt.data()};
    if (!llvm::cl::ParseCommandLineOptions(std::size(optChars), optChars))
      return Error("unable to parse argument: " + opt);
  }

  impl->optionsParsed = true;
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

void MOFContext::overrideLoweringOptions(
    MOFLoweringOptions &mofLoweringOptions,
    MO::GCPipelineOptions &gcPipelineOptions) {
  if (impl->driverOptions) {
    mofLoweringOptions.debugTensorPrintOptions.style =
        impl->driverOptions->resultOutputStyle;
    mofLoweringOptions.debugTensorPrintOptions.precision =
        impl->driverOptions->printPrecision;
    mofLoweringOptions.debugTensorPrintOptions.binaryDir =
        impl->driverOptions->binaryOutputDir;
    gcPipelineOptions.minimumCpuAlignment =
        impl->driverOptions->minimumCpuAlignment;
  }
  if (impl->clOptions)
    static_cast<const MO::GraphCompilerOptions &>(*impl->clOptions)
        .overrideLoweringOptions(gcPipelineOptions);
  gcPipelineOptions.compiledFrameworkLabel =
      CompiledFrameworkLabel::asLabelString(
          CompiledFrameworkLabel::kModularModel);
  if (impl->mofConfig)
    impl->mofConfig->overrideLoweringOptions(mofLoweringOptions,
                                             gcPipelineOptions);

  // Enable NaN/Inf checking from the max-debug.nan-check config key, and
  // read the sampling stride from max-debug.nan-check-stride if set.
  if (auto configOr = M::Config::open(); !configOr.isError()) {
    if (configOr->getValueAsBool("max-debug.nan-check", false))
      gcPipelineOptions.nanCheck = true;
    StringRef strideStr = configOr->getValue("max-debug.nan-check-stride");
    if (!strideStr.empty()) {
      uint64_t stride = 0;
      if (!strideStr.getAsInteger(/*radix=*/10, stride) && stride > 0)
        gcPipelineOptions.nanCheckStride = static_cast<size_t>(stride);
    }
  }
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
        os << "\nSource Traceback:\n"
           // attr.stack_trace() calls the frame_traceback() function that
           // uses Python's traceback.format_list to convert the frames to a
           // human-readable form.
           << attr.stack_trace() << '\n';
      }
      FormatScopedDiagnosticHandler::emitDiagnosticSeverityToStream(os, diag);

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

/// Lowers a module op to MGP then exports to MEF.
static ErrorOr<BufferRef> convertMOToMEF(ModuleOp moduleOp,
                                         const MO::GCPipelineOptions &options,
                                         bool enableMEFMetadata) {
  // Gate IR dumping on the unified Config-driven directory: this picks up
  // `MODULAR_DEBUG=ir-output-dir=...`, `[max-debug] ir-output-dir = ...`,
  // `InferenceSession.debug.ir_output_dir = ...`, and the legacy
  // `MODULAR_MAX_TEMPS_DIR` env var alike.  See `getMaxIRDumpDir`.
  bool const dumpStages = !getMaxIRDumpDir().empty();
  GraphApiDiagnosticHandler diagnostics(moduleOp.getContext());
  if (failed(MO::runGCPipeline(moduleOp, options, dumpStages)))
    return Error(Twine("Graph compilation failed:\n") +
                 diagnostics.formatMessage());

  auto bufOr = MO::convertMGPToMEF(moduleOp, enableMEFMetadata);
  if (bufOr)
    return bufOr.takeError();
  return bufOr.takeValue();
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
  MO::GCPipelineOptions gcPipelineOptions;
  overrideLoweringOptions(mofLoweringOptions, gcPipelineOptions);

  // If the caller (Python `KernelLibrary` via `Analysis::seed_kernel_decls`)
  // already seeded the model module with kernel decls + the opaque-type
  // mapping, signal the GC pipeline to skip `mogg-import-packages`. The
  // attribute serves as the canonical marker: present iff decls are too.
  if ((*moduleOp)->hasAttr(MOGG::kOpaqueTypeMappingAttrName))
    gcPipelineOptions.kernelDeclsPrePopulated = true;

  mofLoweringOptions.setupContext(maxContext.copy());
  bool enableMEFMetadata = impl->getEnableMEFMetadata();

  auto baseCacheDirOr = M::getMaxCacheDir();

  // Disable caching if set or we can't find any suitable location to cache.
  if (!isModelIRCacheEnabled() || !baseCacheDirOr)
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);

  // Create the MEF cache filesystem backend.
  const std::filesystem::path &baseCacheDir = *baseCacheDirOr;
  auto mefBackendOr = Cache::getLocalDefaultBackendChain(
      (baseCacheDir / "mof" / "mef").string(),
      std::string(M::getModularVersionString()));

  if (mefBackendOr.isError()) {
    // We couldn't create a cache, so don't cache.
    return convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  }

  auto keyHashOr = computeMEFCacheKey(*moduleOp, mofLoweringOptions,
                                      gcPipelineOptions, maxContext);
  if (keyHashOr.isError())
    return keyHashOr.takeError();
  std::string keyHash = keyHashOr.takeValue();

  auto mefCache = RCRef<Cache::TransformCache>::create(*mefBackendOr);
  auto maybeMefOr = mefCache->findKeyedSync(keyHash);
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

  auto outBufOr =
      convertMOToMEF(*moduleOp, gcPipelineOptions, enableMEFMetadata);
  if (outBufOr.isError())
    return outBufOr.takeError();
  BufferRef outputBuffer = outBufOr.takeValue();

  // Insert MEF buffer to cache, ignoring failure to insert.
  (void)mefCache->insertKeyedSync(keyHash, outputBuffer.copy());

  return outputBuffer;
}

ErrorOr<M::BufferRef> MOFContext::compileToBinary(M::BufferRef file,
                                                  FileFormat format) {
  CompileToBinaryCtx compileCtx(impl->maxContext.copy());
  auto telemetryContext =
      impl->maxContext->get<GML::MaxEngineTelemetryContext>();
  // Gather all the options from all the sources.
  MOFLoweringOptions mofLoweringOptions;
  MO::GCPipelineOptions gcPipelineOptions;
  overrideLoweringOptions(mofLoweringOptions, gcPipelineOptions);

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
      mofLoweringOptions.setupContext(compileCtx.context.copy());

      GraphApiDiagnosticHandler diagnostics(&compileCtx.mlirCtx);
      if (failed(runGcPipeline(compileCtx.module.get(), gcPipelineOptions)))
        return Error(Twine("Graph compilation failed") +
                     diagnostics.formatMessage());
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
  if (!gcPipelineOptions.kernelLibraryPaths.empty() ||
      !isModelIRCacheEnabled()) {
    return loweringTransform();
  }

  WriteableBufferRef keyBuffer =
      makeCacheKey(std::move(file), mofLoweringOptions, gcPipelineOptions,
                   graphCompilerOptions);

  // Include mojo compilation defines (LOGGING_LEVEL, ASSERT_LEVEL, etc.)
  // in the cache key so that changing them invalidates the cache.
  if (auto *compCtx = impl->maxContext->get<KGEN::CompilationContext>())
    compCtx->print(*keyBuffer);

  return M::cacheMEFTransform("mof", loweringTransform, std::move(keyBuffer));
}
