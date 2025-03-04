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

static std::once_flag initLLVMFlag;

static void initLLVM() {
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

namespace M::MOGG {

namespace {
/// Helpers to compile the KGEN stubs to binary. This splits the compile to
/// binary stuff from the KGEN stub creation process.
class StubCompiler {
public:
  StubCompiler(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
               std::string saveTempsPrefix, KernelBuildType kernelBuildType)
      : ctx(graph->getContext()), kgenModule(std::move(kgenStubs)),
        loc(graph->getLoc()), saveTempsPrefix(std::move(saveTempsPrefix)),
        kernelBuildType(kernelBuildType) {}

  /// Compile to binary object. Tell the jit all of the given symbols as symbols
  /// within it that can be executed at runtime.
  ErrorOr<MOGG::LoadCompiledKernelOp>
  compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming = false,
          bool debugKGENStubs = false);

  /// Run the KGEN elaboration passes on the module.
  LogicalResult elaborateKGEN(bool enableTiming = false, bool superOpt = false,
                              bool debugKGENStubs = false);

private:
  /// Lower to an binary object.
  ErrorOr<MOGG::LoadCompiledKernelOp>
  compileToLLVM(mlir::ArrayAttr combinedSymbolName, bool enableTiming,
                bool debugKGENStubs);

  MLIRContext *ctx;

  /// The KGEN module with all the stubs in it.
  OwningOpRef<ModuleOp> kgenModule;

  /// Location to use.
  mlir::Location loc;

  /// If set to non empty string emit temporary IR and assembly outputs to the
  /// given prefix. Currently emits KGEN before and after elaboration, llvm
  /// before and after optimization, and the final assembly.
  std::string saveTempsPrefix;

  /// How should kernels be compiled. If debuginfo is included in kernels then
  /// it will be added to the stubs.
  KernelBuildType kernelBuildType;
};

} // namespace

LogicalResult StubCompiler::elaborateKGEN(bool enableTiming, bool superOpt,
                                          bool debugKGENStubs) {
  MLIRContext *ctx = kgenModule->getContext();
  initLLVM();

  ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
      kgenModule->getContext(), llvm::sys::getDefaultTargetTriple(),
      llvm::sys::getHostCPUName(), getHostCPUFeatures(), /*tuneCpu=*/"",
      /*acceleratorArch=*/AsyncRT::Device::getAcceleratorArchOrEmpty());
  if (targetOr.isError())
    return mlir::emitError(kgenModule->getLoc(), targetOr.getError());
  TargetInfoAttr target = *targetOr;

  // Set the environment (defines) for the module.
  auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
  KGEN::extendWithModularEnvAttr(*kgenModule, compileCtx);

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, target.getTripleStr(), target.getArch(), target.getFeatures(),
      target.getTuneCpu(), AsyncRT::Device::getAcceleratorArchOrEmpty());
  if (targetInfoOr)
    return kgenModule->emitError(targetInfoOr.getError());

  KGEN::CompilationOptions compilationOptions;
  compilationOptions.enableLLVMPerFunctionSplitting = false;
  compilationOptions.optimizationLevel = superOpt ? 4 : 3;
  compilationOptions.saveTempsPrefix = saveTempsPrefix;
  compilationOptions.targetAccelerator = targetInfoOr->getAcceleratorArch();

  if (debugKGENStubs || hasDebuginfo(kernelBuildType))
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

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

    setTargetInfo(*kgenModule, *targetInfoOr);
    ErrorOrSuccess result = compiler.runElaborationPipeline(
        *kgenModule, *targetInfoOr, *runtime, runtime->getReadyChain().copy(),
        Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
            "StubCompiler::elaborateKGEN",
            "mogg.elaborate.kgen.cache.miss.time",
            {{"pipeline", "StubCompiler::elaborateKGEN"}}),
        Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
            "StubCompiler::elaborateKGEN"));

    if (result.isError()) {
      mlir::emitError(kgenModule->getLoc(), result.getError());
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

static ErrorOr<DenseResourceElementsAttr>
emitArchiveAttr(std::unique_ptr<KGEN::ObjectCompiler> compiler,
                OwningOpRef<ModuleOp> module) {
  MLIRContext *moduleCtx = module->getContext();

  std::string outKeyHash;
  ErrorOr<BufferRef> bufferOr = compiler->emitArchive(
      std::move(module), /*emitAssembly=*/false, &outKeyHash);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  // Produce a DenseResourceElementsAttr from the file.
  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(moduleCtx);

  // Pretend this is a "tensor" of data.
  auto attrType = RankedTensorType::get(
      {(int64_t)buffer->getBufferSize()},
      IntegerType::get(moduleCtx, 8, IntegerType::Unsigned));
  std::string attrName = "archive_" + outKeyHash;
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

ErrorOr<MOGG::LoadCompiledKernelOp>
StubCompiler::compileToLLVM(mlir::ArrayAttr combinedSymbolName,
                            bool enableTiming, bool debugKGENStubs) {
  // Compile to LLVM.
  TargetInfoAttr target = getTargetInfo(*kgenModule);
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

  initLLVM();
  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix);

  KGEN::PassManagerConfigOptions pmOptions;
  if (enableTiming)
    pmOptions.enableTiming = true;
  ErrorOr<std::unique_ptr<KGEN::ObjectCompiler>> compilerOr =
      KGEN::ObjectCompiler::create(".mojo_cache", compilationOptions,
                                   /*isJit=*/true, *ctx, pmOptions);

  if (compilerOr.isError())
    return compilerOr.takeError();

  std::unique_ptr<KGEN::ObjectCompiler> compiler = compilerOr.takeValue();

  auto start = std::chrono::high_resolution_clock::now();

  ErrorOr<DenseResourceElementsAttr> archiveAttrOr =
      emitArchiveAttr(std::move(compiler), std::move(kgenModule));
  if (archiveAttrOr.isError())
    return archiveAttrOr.takeError();
  DenseResourceElementsAttr archiveAttr = *archiveAttrOr;

  auto end = std::chrono::high_resolution_clock::now();
  if (enableTiming) {
    llvm::errs() << "Compiling to llvm took "
                 << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                          start)
                        .count()
                 << "ms\n";
  }

  // Add the compiled kernels as a constant.
  OpBuilder builder{ctx};
  return builder.create<MOGG::LoadCompiledKernelOp>(
      loc, /*symbols=*/combinedSymbolName, /*value=*/archiveAttr,
      /*debuginfo=*/debugKGENStubs || hasDebuginfo(kernelBuildType));
}

ErrorOr<MOGG::LoadCompiledKernelOp>
StubCompiler::compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming,
                      bool debugKGENStubs) {
  initLLVM();

#ifndef MODULAR_PRODUCTION
  if (failed(mlir::verify(*kgenModule)))
    return Error("generated KGEN IR is invalid");
#endif

  LogicalResult ok = elaborateKGEN(enableTiming, false, debugKGENStubs);
  if (ok.failed())
    return Error("KGEN elaboration failed");

  return compileToLLVM(combinedSymbolName, enableTiming, debugKGENStubs);
}

CompileDiagnosticOr<MOGG::LoadCompiledKernelOp>
compileStubs(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
             std::string saveTempsPrefix, KernelBuildType kernelBuildType,
             mlir::ArrayAttr combinedSymbolName, bool enableTiming /*= false*/,
             bool debugKGENStubs /*= false*/) {
  StubCompiler compiler(std::move(kgenStubs), graph, std::move(saveTempsPrefix),
                        kernelBuildType);
  ErrorOr<MOGG::LoadCompiledKernelOp> errOrLoad =
      compiler.compile(combinedSymbolName, enableTiming, debugKGENStubs);
  if (errOrLoad.isError())
    return graph->emitError(errOrLoad.takeError().get());
  return errOrLoad.takeValue();
}
} // namespace M::MOGG
