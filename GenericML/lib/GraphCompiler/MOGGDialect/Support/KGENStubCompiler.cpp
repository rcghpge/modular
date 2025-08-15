//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"

#include "Cache/CacheTelemetryContext.h"
#include "Cache/CachedTransform.h"
#include "Cache/Support/Keys.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"

#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/ToolCommon/PassManagerConfigOptions.h"

#include "Support/Context.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Telemetry/Telemetry.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/TargetParser/Host.h"

#include "xxh3.h"
#include "xxhash.h"

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
  CompileDiagnosticOr<SmallVector<MOGG::LoadCompiledKernelOp>> compile(
      const llvm::MapVector<mlir::StringAttr, KGEN::GeneratorOp> &symbolsAdded);

private:
  StubCompiler(OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
               std::string saveTempsPrefix, KernelBuildType kernelBuildType,
               bool enableTiming, bool debugKGENStubs)
      : ctx(graph->getContext()), kgenModule(std::move(kgenStubs)),
        graph(graph), loc(graph->getLoc()),
        saveTempsPrefix(std::move(saveTempsPrefix)),
        kernelBuildType(kernelBuildType), enableTiming(enableTiming),
        debugKGENStubs(debugKGENStubs) {}

  /// Run the KGEN elaboration passes on the module.
  CompileDiagnosticOrSuccess elaborateKGEN();

  /// Lower to an binary object.
  CompileDiagnosticOr<DenseResourceElementsAttr> compileToLLVM();

  /// Sets and returns the environment (defines) for the module.
  KGEN::EnvAttr setKGENEnvAttr();

  /// Get the hash of the KGEN module.
  std::string getKgenModuleHash() const;

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

  /// Whether to enable timing.
  bool enableTiming;

  /// Whether to emit debug info.
  bool debugKGENStubs;
};

/// Cache key which queries the semnatic hash created by the hashModuleContents
/// function.
struct SemanticHashKey {
  using KeyTy = M::KGEN::GeneratorOp;

  static std::string hashKey(M::KGEN::GeneratorOp op) {
    return op->getAttrOfType<StringAttr>("semantic_hash").str();
  }
};

} // namespace

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

ErrorOr<TargetInfoAttr> getTargetInfo(MLIRContext *ctx) {
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
                        kernelBuildType, enableTiming, debugKGENStubs);
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

KGEN::EnvAttr StubCompiler::setKGENEnvAttr() {
  // Sets and returns the environment (defines) for the module.
  auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
  KGEN::extendWithModularEnvAttr(*kgenModule, compileCtx);

  return KGEN::getModuleEnvAttr(*kgenModule);
}

CompileDiagnosticOrSuccess StubCompiler::elaborateKGEN() {
  KGEN::CompilationOptions compilationOptions = createCompilationOptions(
      targetInfo, kernelBuildType, debugKGENStubs, saveTempsPrefix);
  KGEN::PassManagerConfigOptions pmOptions;
  if (enableTiming)
    pmOptions.enableTiming = true;
  KGEN::KGENCompiler compiler(*ctx, compilationOptions, pmOptions);

  if (!saveTempsPrefix.empty()) {
    std::string outPath = saveTempsPrefix + ".kgen.prelaboration.mlir";
    auto outFile = mlir::openOutputFile(outPath);
    if (!outFile) {
      return kgenModule->emitError()
             << "Could not open pre-elaboration dump file named " << outPath;
    }

    kgenModule->print(outFile->os());
    outFile->keep();
  }

  CompileDiagnosticOrSuccess res = success();
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

    // These two next lines make it possible to get source code displayed when
    // an elaboration error occurs.
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, ctx);

    ErrorOrSuccess result = compiler.runElaborationPipeline(
        *kgenModule, targetInfo, *runtime, runtime->getReadyChain().copy(),
        Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
            "StubCompiler::elaborateKGEN",
            "mogg.elaborate.kgen.cache.miss.time",
            {{"pipeline", "StubCompiler::elaborateKGEN"}}),
        Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
            "StubCompiler::elaborateKGEN"));

    if (result.isError()) {
      res = kgenModule->emitError()
            << "Could not elaborate the provided code: " << result.getError();
    }
  }

  if (res.isError() || saveTempsPrefix.empty())
    return res;

  std::string outPath = saveTempsPrefix + ".kgen.elaborated.mlir";
  auto outFile = mlir::openOutputFile(outPath);
  if (!outFile) {
    return kgenModule->emitError()
           << "Could not open post-elaboration dump file named " << outPath;
  }
  kgenModule->print(outFile->os());
  outFile->keep();

  return res;
}

DenseResourceElementsAttr createArchiveAttr(MLIRContext *ctx, BufferRef buffer,
                                            StringRef name) {
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

CompileDiagnosticOr<DenseResourceElementsAttr> StubCompiler::compileToLLVM() {
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

CompileDiagnosticOr<SmallVector<MOGG::LoadCompiledKernelOp>>
StubCompiler::compile(const llvm::MapVector<mlir::StringAttr, KGEN::GeneratorOp>
                          &symbolsToCompile) {

  // Algorithm:
  // 1. For the GeneratorOp instances in symbolsAdded, check if the cache has
  //    a cached version.
  // 2. If yes, the cached value will be a key into mojo cache, which will need
  // to be pulled out and returned to the caller
  // 3. If no, the op will need to be appended to the kgen module
  // 4. The kgen module will then be compiled to LLVM
  // 5. The ops that didn't have a cache hit will be cached with the mojo cache
  // hash
  // 6. A vector of LoadCompiledKernelOps will be returned to the caller

  KGEN::EnvAttr envAttr = setKGENEnvAttr();
  WriteableBufferRef keyBufRef = WriteableBuffer::get();
  envAttr.printStripped(*keyBufRef);
  createCompilationOptions(targetInfo, kernelBuildType, debugKGENStubs,
                           saveTempsPrefix)
      .print(*keyBufRef);

  auto keyBuffer = keyBufRef->getBuffer();
  std::string envKeyHash =
      llvm::utohexstr(XXH3_64bits(keyBuffer.data(), keyBuffer.size()),
                      /*LowerCase=*/true,
                      /*Width=*/16);

  // The MOGG cache is identified by the hash of the environment and the hash
  // of the KGEN stub module.
  std::string keyHash = std::move(envKeyHash);

  ErrorOr<RCRef<Cache::BlobCacheBackend>> moggCacheBackendOr =
      Cache::getLocalDefaultBackendChain(std::filesystem::path(".mogg_cache") /
                                         keyHash);
  if (moggCacheBackendOr.isError())
    return kgenModule->emitError(moggCacheBackendOr.getError());

  auto moggCache = RCRef<Cache::BlobCache<SemanticHashKey>>::create(
      moggCacheBackendOr.takeValue());
  RCRef<Cache::TransformCache> mojoCache = compiler->getTransformCache();

  // this is a list of ops that had a cache hit and the contents of the cache
  // blob, which is a key into the mojo cache. if we don't find this value in
  // the mojo cache, this op needs to be considered a cache miss
  llvm::StringMap<std::pair<llvm::SmallVector<mlir::StringAttr>, BufferRef>>
      mojoCacheHits;

  SmallVector<mlir::StringAttr> cacheMissSymbols;
  SmallVector<std::string> cacheMissHashes;
  for (auto [symbolNameAttr, genOp] : symbolsToCompile) {
    std::string genOpHash;
    ErrorOr<std::optional<BufferRef>> moggCacheResult =
        moggCache->findSync(genOp, &genOpHash);
    if (moggCacheResult.isError()) {
#ifndef MODULAR_PRODUCTION
      kgenModule->emitWarning(
          moggCacheResult.takeError()
              .get()); // Swallow the error, because it's not critical
#endif
    }

    if (moggCacheResult->has_value()) {
      // we have a potential cache hit!
      BufferRef buffer = moggCacheResult->value();
      llvm::StringRef mojoCacheHashKey = buffer->getBuffer();

      // now we need to find the value in the mojo cache
      ErrorOr<std::optional<BufferRef>> mojoCacheResultOr =
          mojoCache->findKeyedSync(mojoCacheHashKey);
      if (mojoCacheResultOr.isError() || !mojoCacheResultOr->has_value()) {
        // this is a cache miss!
        cacheMissSymbols.push_back(symbolNameAttr);
        cacheMissHashes.push_back(genOpHash);
        if (mojoCacheResultOr.isError()) {
#ifndef MODULAR_PRODUCTION
          kgenModule->emitWarning(
              mojoCacheResultOr.takeError().get()); // Swallow the error,
                                                    // because it's not critical
#endif
        }
        continue;
      }
      std::optional<BufferRef> maybeMojoCachedItemBuffer =
          mojoCacheResultOr.takeValue();

      // we have a real cache hit!
      BufferRef mojoCachedItemBuffer = maybeMojoCachedItemBuffer.value();

      genOp.erase();
      auto [it, inserted] = mojoCacheHits.insert(
          {mojoCacheHashKey,
           {llvm::SmallVector<mlir::StringAttr>({symbolNameAttr}),
            std::move(mojoCachedItemBuffer)}});
      if (!inserted) {
        it->second.first.push_back(symbolNameAttr);
      }
    } else {
      cacheMissSymbols.push_back(symbolNameAttr);
      cacheMissHashes.push_back(genOpHash);
    }
  }

  SmallVector<MOGG::LoadCompiledKernelOp> results;
  results.reserve(mojoCacheHits.size() + !cacheMissSymbols.empty());
  OpBuilder builder{ctx};
  if (!cacheMissSymbols.empty()) {
#ifndef MODULAR_PRODUCTION
    if (failed(mlir::verify(*kgenModule)))
      return kgenModule->emitError("The generated KGEN IR is invalid");
#endif

    CompileDiagnosticOrSuccess elaborationResult = elaborateKGEN();
    if (elaborationResult.isError()) {
      return graph->emitError("The graph compiler tried to JIT compile the "
                              "provided kernels but failed during elaboration");
    }

    CompileDiagnosticOr<DenseResourceElementsAttr> result = compileToLLVM();
    if (result.isError()) {
      return graph->emitError(
          "The graph compiler tried to JIT compile the "
          "provided kernels but failed at lowering to LLVM");
    }

    DenseResourceElementsAttr archiveAttr = result.takeValue();
    DenseResourceElementsHandle binaryBlobHandle = archiveAttr.getRawHandle();
    StringRef binaryBlobKey = binaryBlobHandle.getKey();
    binaryBlobKey.consume_front("archive_");
    for (const std::string &hash : cacheMissHashes) {
      BufferRef buffer = Buffer::get(binaryBlobKey);
      if (auto error = moggCache->insertKeyedSync(hash, buffer)) {
#ifndef MODULAR_PRODUCTION
        kgenModule->emitWarning(
            error.takeError()
                .get()); // Swallow the error, because it's not critical
#endif
      }
    }

    ArrayAttr symbolsAttr = mlir::ArrayAttr::get(
        ctx, llvm::to_vector_of<mlir::Attribute>(cacheMissSymbols));
    results.push_back(builder.create<MOGG::LoadCompiledKernelOp>(
        loc, /*symbols=*/symbolsAttr, /*value=*/archiveAttr,
        /*debuginfo=*/debugKGENStubs || hasDebuginfo(kernelBuildType)));
    results.back()->setAttr("cache_miss", builder.getUnitAttr());
  }

  for (const auto &[blobHash, symbolsAndBufferPair] : mojoCacheHits) {
    ArrayAttr symbolsAttr = builder.getArrayAttr(
        llvm::to_vector_of<Attribute>(symbolsAndBufferPair.first));
    DenseResourceElementsAttr archiveAttr =
        createArchiveAttr(ctx, symbolsAndBufferPair.second, blobHash);
    results.push_back(builder.create<MOGG::LoadCompiledKernelOp>(
        loc, /*symbols=*/symbolsAttr, /*value=*/archiveAttr,
        /*debuginfo=*/debugKGENStubs || hasDebuginfo(kernelBuildType)));
    results.back()->setAttr("cache_hit", builder.getUnitAttr());
  }
  return results;
}

CompileDiagnosticOr<SmallVector<MOGG::LoadCompiledKernelOp>> compileStubs(
    OwningOpRef<ModuleOp> kgenStubs, Operation *graph,
    std::string saveTempsPrefix, KernelBuildType kernelBuildType,
    const llvm::MapVector<mlir::StringAttr, KGEN::GeneratorOp> &symbolsAdded,
    bool enableTiming /*= false*/, bool debugKGENStubs /*= false*/) {
  CompileDiagnosticOr<StubCompiler> compilerOr =
      StubCompiler::create(std::move(kgenStubs), graph, saveTempsPrefix,
                           kernelBuildType, enableTiming, debugKGENStubs);
  if (compilerOr.isError())
    return compilerOr.takeError();
  StubCompiler compiler = compilerOr.takeValue();
  return compiler.compile(symbolsAdded);
}
} // namespace M::MOGG
