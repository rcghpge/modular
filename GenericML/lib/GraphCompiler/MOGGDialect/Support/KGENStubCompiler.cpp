//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "KGEN/ToolCommon/PassManagerConfigOptions.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/Preprocessor.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"

namespace M::MOGG {
namespace {
/// Helpers to compile the KGEN stubs to binary. This splits the compile to
/// binary stuff from the KGEN stub creation process.
class StubCompiler {
public:
  static CompileDiagnosticOr<StubCompiler>
  create(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
         std::string saveTempsPrefix, KernelBuildType kernelBuildType,
         bool enableTiming, bool debugKGENStubs);

  /// Compile to binary object. Tell the jit all of the given symbols as symbols
  /// within it that can be executed at runtime.
  CompileDiagnosticOr<MOGG::LoadCompiledKernelOp>
  compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming = false,
          bool debugKGENStubs = false);

private:
  StubCompiler(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
               std::string saveTempsPrefix, KernelBuildType kernelBuildType)
      : ctx(graph->getContext()), kgenModule(std::move(kgenStubs)),
        graph(graph), loc(graph->getLoc()),
        saveTempsPrefix(std::move(saveTempsPrefix)),
        kernelBuildType(kernelBuildType) {}

  /// Run the KGEN elaboration passes on the module.
  LogicalResult elaborateKGEN(bool enableTiming = false, bool superOpt = false,
                              bool debugKGENStubs = false);
  /// Lower to an binary object.
  CompileDiagnosticOr<DenseResourceElementsAttr>
  compileToLLVM(bool enableTiming, bool debugKGENStubs);

  MLIRContext *ctx;

  /// The compiler to use to compile the KGEN stubs to LLVM.
  std::unique_ptr<KGEN::ObjectCompiler> compiler;

  /// The KGEN module with all the stubs in it.
  OwningOpRef<ModuleOp> kgenModule;

  Operation *graph;

  /// Location to use.
  mlir::Location loc;

  /// If set to non empty string emit temporary IR and assembly outputs to the
  /// given prefix. Currently emits KGEN before and after elaboration, llvm
  /// before and after optimization, and the final assembly.
  std::string saveTempsPrefix;

  /// How should kernels be compiled. If debuginfo is included in kernels then
  /// it will be added to the stubs.
  KernelBuildType kernelBuildType;

  /// The target info for the module.
  TargetInfoAttr targetInfo;
};

} // namespace

static void initLLVM() {
  static std::once_flag initLLVMFlag;
  std::call_once(initLLVMFlag, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetAsmPrinter();

    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

static KGEN::CompilationOptions
createCompilationOptions(TargetInfoAttr target, KernelBuildType kernelBuildType,
                         bool debugKGENStubs, StringRef saveTempsPrefix) {

  KGEN::CompilationOptions compilationOptions;
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();
  compilationOptions.targetAccelerator =
      AsyncRT::Device::getAcceleratorArchOrEmpty();
  compilationOptions.enableLLVMPerFunctionSplitting = false;

  if (kernelBuildType == KernelBuildType::kDebug)
    compilationOptions.optimizationLevel = 0;

  if (debugKGENStubs || hasDebuginfo(kernelBuildType))
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix.str());

  return compilationOptions;
}

static std::unique_ptr<KGEN::ObjectCompiler>
createObjectCompiler(ModuleOp kgenModule,
                     KGEN::CompilationOptions compilationOptions,
                     bool enableTiming) {
  MLIRContext *ctx = kgenModule->getContext();
  // Compile to LLVM.
  KGEN::PassManagerConfigOptions pmOptions;
  if (enableTiming)
    pmOptions.enableTiming = true;
  ErrorOr<std::unique_ptr<KGEN::ObjectCompiler>> compilerOr =
      KGEN::ObjectCompiler::create(".mojo_cache", compilationOptions,
                                   /*isJit=*/true, *ctx, pmOptions);

  if (compilerOr.isError())
    kgenModule->emitError(compilerOr.getError());

  return compilerOr.takeValue();
}

static ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx) {
  ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
      ctx, llvm::sys::getDefaultTargetTriple(), llvm::sys::getHostCPUName(),
      getHostCPUFeatures(), /*tuneCpu=*/"",
      /*acceleratorArch=*/AsyncRT::Device::getAcceleratorArchOrEmpty());
  if (targetOr.isError())
    return targetOr;

  TargetInfoAttr target = targetOr.takeValue();

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, target.getTripleStr(), target.getArch(), target.getFeatures(),
      target.getTuneCpu(), AsyncRT::Device::getAcceleratorArchOrEmpty());
  return targetInfoOr;
}

/*static*/ CompileDiagnosticOr<StubCompiler>
StubCompiler::create(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
                     std::string saveTempsPrefix,
                     KernelBuildType kernelBuildType, bool enableTiming,
                     bool debugKGENStubs) {
  MLIRContext *ctx = graph->getContext();
  StubCompiler compiler(std::move(kgenStubs), graph, saveTempsPrefix,
                        kernelBuildType);
  auto targetInfoOr = getTargetInfo(ctx);
  if (targetInfoOr.isError())
    return compiler.graph->emitError(targetInfoOr.getError());
  compiler.targetInfo = targetInfoOr.takeValue();
  KGEN::CompilationOptions compilationOptions = createCompilationOptions(
      compiler.targetInfo, kernelBuildType, debugKGENStubs, saveTempsPrefix);
  compiler.compiler = createObjectCompiler(*compiler.kgenModule,
                                           compilationOptions, enableTiming);
  return compiler;
}

LogicalResult StubCompiler::elaborateKGEN(bool enableTiming, bool superOpt,
                                          bool debugKGENStubs) {
  // Set the environment (defines) for the module.
  auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
  KGEN::extendWithModularEnvAttr(*kgenModule, compileCtx);

  KGEN::CompilationOptions compilationOptions = createCompilationOptions(
      targetInfo, kernelBuildType, debugKGENStubs, saveTempsPrefix);
  KGEN::PassManagerConfigOptions pmOptions;
  if (enableTiming)
    pmOptions.enableTiming = true;
  KGEN::KGENCompiler compiler(*ctx, compilationOptions, pmOptions);

  if (!saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.prelaboration.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();

    kgenModule->print(outFile->os());
    outFile->keep();
  }

  LogicalResult res = success();
  {
    auto fileLine = loc->findInstanceOf<mlir::FileLineColLoc>();
    llvm::StringMap<Telemetry::MetricAttributeValue> attrs;

    if (fileLine)
      attrs["filename"] = fileLine.getFilename().str();

    [[maybe_unused]] auto timeScope =
        loadContext(ctx)
            ->get<M::Telemetry::TelemetryContext>()
            ->createUInt64Timer<std::chrono::milliseconds>(
                "mogg.kgen.elaboration.time", M::Telemetry::Level::L2, attrs);

    auto *runtime = loadContext(ctx)->get<AsyncRT::Runtime>();

    ErrorOrSuccess result = compiler.runElaborationPipeline(
        *kgenModule, targetInfo, *runtime, runtime->getReadyChain().copy(),
        Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
            "StubCompiler::elaborateKGEN",
            "mogg.elaborate.kgen.cache.miss.time",
            {{"pipeline", "StubCompiler::elaborateKGEN"}}),
        Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
            "StubCompiler::elaborateKGEN"));

    if (result.isError()) {
      kgenModule->emitError(result.getError());
      res = failure();
    }
  }

  if (succeeded(res) && !saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.elaborated.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();
    kgenModule->print(outFile->os());
    outFile->keep();
  }

  return res;
}

static DenseResourceElementsAttr
createArchiveAttr(MLIRContext *ctx, BufferRef buffer, StringRef name) {
  auto resourceManager = DenseResourceElementsHandle::getManagerInterface(ctx);

  // Pretend this is a "tensor" of data.
  auto attrType =
      RankedTensorType::get({(int64_t)buffer->getBufferSize()},
                            IntegerType::get(ctx, 8, IntegerType::Unsigned));
  std::string attrName = "archive_" + name.str();
  mlir::AsmResourceBlob blob(
      {buffer->getBufferStart(), buffer->getBufferSize()}, /*dataAlignment=*/8,
      /*deleter=*/
      [buffer = std::move(buffer)](void *, size_t, size_t) mutable {
        buffer.reset(); // Shouldn't need to do this, but just to be explicit.
      },
      /*isMutable=*/false);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

static ErrorOr<DenseResourceElementsAttr>
emitArchiveAttr(KGEN::ObjectCompiler &compiler, OwningOpRef<ModuleOp> module) {
  MLIRContext *moduleCtx = module->getContext();

  std::string outKeyHash;
  ErrorOr<BufferRef> bufferOr = compiler.emitArchive(
      std::move(module), /*emitAssembly=*/false, &outKeyHash);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  // Produce a DenseResourceElementsAttr from the file.
  return createArchiveAttr(moduleCtx, std::move(buffer), outKeyHash);
}

CompileDiagnosticOr<DenseResourceElementsAttr>
StubCompiler::compileToLLVM(bool enableTiming, bool debugKGENStubs) {
  auto start = std::chrono::high_resolution_clock::now();

  ErrorOr<DenseResourceElementsAttr> archiveAttrOr =
      emitArchiveAttr(*compiler, std::move(kgenModule));
  if (archiveAttrOr.isError())
    return graph->emitError(archiveAttrOr.getError());
  DenseResourceElementsAttr archiveAttr = *archiveAttrOr;

  auto end = std::chrono::high_resolution_clock::now();
  if (enableTiming) {
    llvm::errs() << "Compiling to llvm took "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                          start)
                        .count()
                 << "ms\n";
  }

  return archiveAttr;
}

CompileDiagnosticOr<MOGG::LoadCompiledKernelOp>
StubCompiler::compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming,
                      bool debugKGENStubs) {
#ifndef MODULAR_PRODUCTION
  if (failed(mlir::verify(*kgenModule)))
    return kgenModule->emitError("generated KGEN IR is invalid");
#endif

  LogicalResult ok = elaborateKGEN(enableTiming, debugKGENStubs);
  if (ok.failed())
    return graph->emitError("KGEN elaboration failed");

  CompileDiagnosticOr<DenseResourceElementsAttr> result =
      compileToLLVM(enableTiming, debugKGENStubs);
  if (result.isError())
    return result.takeError();

  DenseResourceElementsAttr archiveAttr = result.takeValue();

  OpBuilder builder{ctx};
  return builder.create<MOGG::LoadCompiledKernelOp>(
      loc, /*symbols=*/
      combinedSymbolName,
      /*value=*/archiveAttr,
      /*debuginfo=*/debugKGENStubs || hasDebuginfo(kernelBuildType));
}

CompileDiagnosticOr<MOGG::LoadCompiledKernelOp>
compileStubs(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
             std::string saveTempsPrefix, KernelBuildType kernelBuildType,
             mlir::ArrayAttr combinedSymbolName, bool enableTiming /*= false*/,
             bool debugKGENStubs /*= false*/) {
  initLLVM();
  CompileDiagnosticOr<StubCompiler> compilerOr =
      StubCompiler::create(std::move(kgenStubs), graph, saveTempsPrefix,
                           kernelBuildType, enableTiming, debugKGENStubs);
  if (compilerOr.isError())
    return compilerOr.takeError();
  StubCompiler compiler = compilerOr.takeValue();
  return compiler.compile(combinedSymbolName, enableTiming, debugKGENStubs);
}
} // namespace M::MOGG
