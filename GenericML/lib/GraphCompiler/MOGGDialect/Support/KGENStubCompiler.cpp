//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Context.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "Support/Preprocessor.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"

static std::mutex &getMut() {
  static std::mutex mut;
  return mut;
}

static bool hasInit = false;

namespace M::MOGG::Stub {

LogicalResult StubCompiler::elaborateKGEN(bool enableTiming, bool superOpt) {
  MLIRContext *ctx = kgenModule->getContext();

  {
    std::lock_guard<std::mutex> guard(getMut());
    if (!hasInit) {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmParser();
      llvm::InitializeNativeTargetAsmPrinter();
      hasInit = true;
    }
  }

  ErrorOr<TargetInfoAttr> targetOr = M::getTargetInfoFor(
      kgenModule->getContext(), llvm::sys::getDefaultTargetTriple(),
      llvm::sys::getHostCPUName(), getHostCPUFeatures());
  if (targetOr.isError())
    return mlir::emitError(kgenModule.getLoc(), targetOr.getError());
  TargetInfoAttr target = *targetOr;

  // Set the environment (defines) for the module.
  KGEN::extendWithModularEnvAttr(kgenModule);

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr =
      getTargetInfoFor(ctx, target.getTripleStr(), target.getArch(),
                       target.getFeatures(), target.getTuneCpu());
  if (targetInfoOr)
    return kgenModule->emitError(targetInfoOr.getError());

  KGEN::CompilationOptions compilationOptions;
  compilationOptions.enableSearch = shouldSearch;
  compilationOptions.enableLLVMPerFunctionSplitting = false;
  compilationOptions.optimizationLevel = superOpt ? 4 : 3;

  static constexpr llvm::StringLiteral kKernelsBuildType =
      STRINGIFY(KERNELS_BUILD_TYPE);
  if (kKernelsBuildType == "relwithdebinfo" || kKernelsBuildType == "debug")
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  KGEN::KGENCompiler compiler(*ctx, compilationOptions);
  if (enableTiming) {
    compiler.configurePassManager(
        [](mlir::PassManager &pm) { pm.enableTiming(); });
  }

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

    auto *runtime = loadContext(ctx)->get<LLCL::Runtime>();

    setTargetInfo(kgenModule, *targetInfoOr);
    LLCL::AnyAsyncValueRef ready = compiler.runElaborationPipeline(
        kgenModule, *targetInfoOr, *runtime, runtime->getReadyChain().copy(),
        Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
            "StubCompiler::elaborateKGEN",
            "mogg.elaborate.kgen.cache.miss.time",
            {{"pipeline", "StubCompiler::elaborateKGEN"}}),
        Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
            "StubCompiler::elaborateKGEN"));

    LLCL::await(ready);
    if (ready.isError()) {
      LLCL::EncodedDiagnostic diag = ready.takeDiagnostic();
      LLCL::DecodedLocation loc = diag.decodeLocation();
      mlir::emitError(
          FileLineColLoc::get(ctx, loc.filename, loc.line, loc.column),
          diag.getMessage().get());
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

ErrorOr<MOGG::LoadCompiledKernelOp>
StubCompiler::compileToLLVM(mlir::ArrayAttr combinedSymbolName,
                            bool enableTiming) {
  static constexpr llvm::StringLiteral kKernelsBuildType =
      STRINGIFY(KERNELS_BUILD_TYPE);

  mlir::PassManager mgr(ctx);

  if (enableTiming)
    mgr.enableTiming();

  // Compile to LLVM.
  mlir::SymbolTable symTable{kgenModule};
  TargetInfoAttr target = getTargetInfo(kgenModule);
  KGEN::CompilationOptions compilationOptions;
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();
  compilationOptions.enableLLVMPerFunctionSplitting = false;

  if (kKernelsBuildType == "debug")
    compilationOptions.optimizationLevel = 0;

  if (kKernelsBuildType == "relwithdebinfo" || kKernelsBuildType == "debug")
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  {
    std::lock_guard<std::mutex> guard(getMut());
    if (!hasInit) {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmParser();
      llvm::InitializeNativeTargetAsmPrinter();
      hasInit = true;
    }
  }
  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix);

  auto compilerOr = KGEN::ObjectCompiler::create(
      mgr, ".mojo_cache", compilationOptions, /*isJit=*/true);
  if (compilerOr.isError())
    return compilerOr.takeError();
  llvm::MapVector<StringAttr, KGEN::ExportedSymbol> exportedSymbols =
      KGEN::getExportedSymbols(kgenModule);

  KGEN::ObjectCompiler compiler = compilerOr.takeValue();

  auto start = std::chrono::high_resolution_clock::now();

  auto standaloneOr =
      compiler.produceStandaloneArchiveAttr(symTable, exportedSymbols, target);
  if (standaloneOr.isError())
    return standaloneOr.takeError();
  ElementsAttr archiveAttr = *standaloneOr;

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
      loc, /*symbols=*/combinedSymbolName, /*value=*/archiveAttr);
}

ErrorOr<MOGG::LoadCompiledKernelOp>
StubCompiler::compile(mlir::ArrayAttr combinedSymbolName, bool enableTiming) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  LogicalResult ok = elaborateKGEN(enableTiming);
  if (ok.failed())
    return Error("KGEN elaboration failed");

  return compileToLLVM(combinedSymbolName, enableTiming);
}

} // namespace M::MOGG::Stub
