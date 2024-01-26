//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "LLCL/Runtime/Runtime.h"

#include "Cache/CacheDialect/CachedTransform.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/Package/Package.h"
#include "KGEN/ToolCommon/CLOptions.h"
#include "LLCL/Runtime/Algorithms.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Telemetry/Telemetry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"

namespace M::MOGG::Stub {

LogicalResult StubCompiler::elaborateKGEN() {
  MLIRContext *ctx = kgenModule->getContext();

  auto ownedRuntime = LLCL::createRuntimeIfNeeded(
      LLCL::RuntimeOptions().withLeakCheckedAllocator());

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::PassManager pm(ctx);

  // Set the environment (defines) for the module.
  KGEN::extendWithModularEnvAttr(kgenModule);

  // Retrieve the target info from the device spec, however 'complete' it with
  // additional fields derived from the 'core' fields.
  ErrorOr<TargetInfoAttr> targetInfoOr = getTargetInfoFor(
      ctx, deviceSpec.getTarget().getTripleStr(),
      deviceSpec.getTarget().getArch(), deviceSpec.getTarget().getFeatures(),
      deviceSpec.getTarget().getTuneCpu());
  if (targetInfoOr)
    return kgenModule->emitError(targetInfoOr.getError());

  KGEN::CompilationOptions compilationOptions;
  compilationOptions.enableSearch = shouldSearch;

  KGEN::populateElaborateModulePasses(pm, *ownedRuntime, *targetInfoOr,
                                      compilationOptions);

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
    auto fileLine = graphOp->getLoc()->findInstanceOf<mlir::FileLineColLoc>();
    llvm::StringMap<Telemetry::MetricAttributeValue> attrs;

    if (fileLine)
      attrs["filename"] = fileLine.getFilename().str();

    [[maybe_unused]] auto timeScope =
        ownedRuntime->emplaceContextIfMissing<M::Telemetry::TelemetryContext>()
            .createUInt64Timer<std::chrono::milliseconds>(
                "mogg.kgen.elaboration.time", M::Telemetry::Level::L2, attrs);

    auto cacheBackends = KGEN::getMojoCacheBackends(*ownedRuntime);
    if (cacheBackends.isError()) {
      // Just run the passes.
      res = pm.run(kgenModule);
    } else {
      setTargetInfo(kgenModule, *targetInfoOr);
      LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
          kgenModule,
          RCRef<Cache::RegionCache>::create(std::move(cacheBackends->first)),
          RCRef<Cache::TransformCache>::create(
              std::move(cacheBackends->second)),
          ownedRuntime->getReadyChain().copy(), pm, /*deflateTarget=*/false);
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
  }

  if (!saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.elaborated.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile)
      return failure();
    kgenModule->print(outFile->os());
    outFile->keep();
  }

  return res;
}

ErrorOrSuccess StubCompiler::compileToLLVM(mlir::ArrayAttr combinedSymbolName) {
  static constexpr llvm::StringLiteral kKernelsBuildType =
      STRINGIFY(KERNELS_BUILD_TYPE);

  auto ownedRuntime = LLCL::createRuntimeIfNeeded(
      LLCL::RuntimeOptions().withLeakCheckedAllocator());
  mlir::PassManager mgr(ctx);

  // Compile to LLVM.
  mlir::SymbolTable symTable{kgenModule};
  TargetInfoAttr target = getTargetInfo(kgenModule);
  KGEN::CompilationOptions compilationOptions;
  compilationOptions.targetTriple = target.getTripleStr();
  compilationOptions.targetCpu = target.getArch();
  compilationOptions.targetFeatures = target.getFeatures();

  if (kKernelsBuildType == "debug")
    compilationOptions.optimizationLevel = 0;

  if (llvm::is_contained({"debug", "relwithdebinfo"}, kKernelsBuildType))
    compilationOptions.debugLevel = KGEN::CompilationOptions::kFullDebugInfo;

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  if (!saveTempsPrefix.empty())
    compilationOptions.setSaveTemps(saveTempsPrefix);

  auto compilerOr = KGEN::ObjectCompiler::create(
      *ownedRuntime, mgr, ".mojo_cache", compilationOptions, /*isJit=*/true);
  if (compilerOr.isError())
    return compilerOr.takeError();
  llvm::MapVector<StringAttr, KGEN::ExportedSymbol> exportedSymbols =
      KGEN::getExportedSymbols(kgenModule);

  KGEN::ObjectCompiler compiler = compilerOr.takeValue();

  auto standaloneOr =
      compiler.produceStandaloneArchiveAttr(symTable, exportedSymbols, target);
  if (standaloneOr.isError())
    return standaloneOr.takeError();
  ElementsAttr archiveAttr = *standaloneOr;

  // Add the compiled kernels as a constant.
  auto loadCompiledOp = moggBuilder.create<MOGG::LoadCompiledKernelOp>(
      loc, /*symbols=*/combinedSymbolName, /*value=*/archiveAttr);

  // If working with multiple devices, remember the device these kernels are
  // to be executed on.
  if (graphOp.getDeviceSpecs())
    loadCompiledOp.setDeviceRef(deviceSpec.getRef());

  return success();
}

ErrorOrSuccess StubCompiler::compile(mlir::ArrayAttr combinedSymbolName) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  LogicalResult ok = elaborateKGEN();
  if (ok.failed())
    return Error("KGEN elaboration failed");

  return compileToLLVM(combinedSymbolName);
}

} // namespace M::MOGG::Stub
