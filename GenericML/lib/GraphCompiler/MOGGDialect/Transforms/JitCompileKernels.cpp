//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "MLRT/AsyncRT/CompilerSupport/Context.h"
#include "MLRT/AsyncRT/Runtime/Runtime.h"
#include "Support/HashUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Telemetry/Telemetry.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Convert all possible operations directly to the available operation.
struct JitCompileKernels : public MojoLibraryAnalysisPass<
                               impl::JitCompileKernelsBase<JitCompileKernels>> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    KGEN::CompilerTimeTraceScope traceScope("jit-compile-kernels",
                                            module->getName().getStringRef());

    initLLVM();

    if (compileMojoToBinary(module, kernelLibraryPaths).isError())
      signalPassFailure();
  }

  struct SemanticHashKey {
    using KeyTy = M::KGEN::GeneratorOp;

    static std::string hashKey(M::KGEN::GeneratorOp op) {
      return op->getAttrOfType<StringAttr>("semantic_hash").str();
    }
  };

  struct KGENCachingResult {
    SmallVector<StringAttr> cacheMissSymbols;
    SmallVector<std::string> cacheMissHashes;
    llvm::StringMap<std::pair<llvm::SmallVector<mlir::StringAttr>, BufferRef>>
        mojoCacheHits;
    RCRef<M::Cache::BlobCache<SemanticHashKey>> moggCache;
  };

  // Compilation state passed between functions to avoid too many arguments
  struct CompilationState {
    MLIRContext *ctx;
    ModuleOp mgpModule;
    OwningOpRef<ModuleOp> kgenModule;
    KGEN::ObjectCompiler *objectCompiler;
    KGEN::KGENCompiler *compiler;
    TargetInfoAttr targetInfo;
    AsyncRT::Runtime *runtime;
    KGEN::CompilationOptions opts;
    OpBuilder builder;
    RCRef<Cache::TransformCache> mojoCache;

    CompilationState(MLIRContext *ctx, ModuleOp mgpModule,
                     OwningOpRef<ModuleOp> kgenModule,
                     KGEN::ObjectCompiler *objectCompiler,
                     KGEN::KGENCompiler *compiler, TargetInfoAttr targetInfo,
                     AsyncRT::Runtime *runtime, KGEN::CompilationOptions opts)
        : ctx(ctx), mgpModule(mgpModule), kgenModule(std::move(kgenModule)),
          objectCompiler(objectCompiler), compiler(compiler),
          targetInfo(targetInfo), runtime(runtime), opts(std::move(opts)),
          builder(ctx), mojoCache(objectCompiler->getTransformCache()) {}
  };

  // Collect unique kernel symbols from ExecuteCompiledOp calls
  llvm::SetVector<StringAttr> collectRegionSymbols(ModuleOp mgpModuleOp) {
    llvm::SetVector<StringAttr> uniqueSymbols;
    for (auto &&modelOp : mgpModuleOp.getOps<MGP::ModelOp>()) {
      modelOp.walk([&](MOGG::ExecuteCompiledOp callOp) {
        uniqueSymbols.insert(callOp.getCalleeSymbolAttr().getAttr());
      });
    }
    return uniqueSymbols;
  }

  // Rename stub functions with semantic naming
  SmallVector<StringAttr> collectStubSymbols(ModuleOp kgenModuleOp) {
    SmallVector<StringAttr> stubs;
    kgenModuleOp.walk<mlir::WalkOrder::PreOrder>(
        [&](KGEN::GeneratorOp generator) {
          // <input>:: is the prefix of all functions created by the GC because
          // there is no input file (all in-memory).
          if (generator.getSymName().contains("<input>::"))
            stubs.emplace_back(generator.getSymNameAttr());
          return mlir::WalkResult::skip();
        });

    return stubs;
  }

  // Rename region functions with semantic naming
  DenseMap<StringAttr, KGEN::GeneratorOp>
  renameGenerators(ModuleOp kgenModuleOp,
                   const llvm::SetVector<StringAttr> &uniqueSymbols) {
    auto symbolsToRename = llvm::map_to_vector(
        uniqueSymbols, [](StringAttr symbol) { return symbol; });
    return MOGG::applySemanticNaming(kgenModuleOp, symbolsToRename,
                                     SymbolTable(kgenModuleOp),
                                     /*namePrefix=*/"__kernel_");
  }

  // Update call sites and remove old LoadCompiledKernelOp
  void updateMGPCallSites(
      ModuleOp mgpModuleOp,
      const DenseMap<StringAttr, KGEN::GeneratorOp> &renamedRegions) {
    for (auto &&modelOp : mgpModuleOp.getOps<MGP::ModelOp>()) {
      DenseSet<StringAttr> kernelsToLoadInThisModel;
      MOGG::LoadCompiledKernelOp loadKernelsOp = nullptr;
      modelOp.walk([&](Operation *op) {
        if (auto callOp = dyn_cast<MOGG::ExecuteCompiledOp>(op)) {
          auto oldName = callOp.getCalleeSymbolAttr().getAttr();
          auto it = renamedRegions.find(oldName);
          if (it != renamedRegions.end()) {
            auto genOp = it->second;
            auto newName = genOp.getSymNameAttr();
            callOp.setCalleeSymbol(newName);
            kernelsToLoadInThisModel.insert(newName);
          }
        } else if (auto loadCompiledKernelOp =
                       dyn_cast<MOGG::LoadCompiledKernelOp>(op)) {
          ASSERT_STREAM(
              !loadKernelsOp,
              << "Expected at most one LoadCompiledKernelOp per ModelOp");
          loadKernelsOp = loadCompiledKernelOp;
        }
      });

      if (loadKernelsOp)
        loadKernelsOp.erase();
    }
  }

  // Build cache key from environment attributes
  CompileDiagnosticOr<std::string> buildCacheKey(ModuleOp kgenModuleOp) {
    WriteableBufferRef keyBufRef = WriteableBuffer::get();
    auto ctx = kgenModuleOp.getContext();

    auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
    KGEN::extendWithModularEnvAttr(kgenModuleOp, compileCtx);
    auto envAttr = KGEN::getModuleEnvAttr(kgenModuleOp);
    envAttr.printStripped(*keyBufRef);

    if (auto error = MOGG::graphCompilerCacheKey(
            *keyBufRef, ctx,
            KernelBuildType::kRelease, // TODO: Support the other types.
            debugKGENStubs,
            "" // TODO: Consider removing this option.
        );
        error.isError()) {
      return kgenModuleOp->emitError() << error.takeError();
    }

    auto keyBuffer = keyBufRef->getBuffer();
    std::string envKeyHash =
        llvm::utohexstr(XXH3_64bits(keyBuffer.data(), keyBuffer.size()),
                        /*LowerCase=*/true,
                        /*Width=*/16);
    return envKeyHash;
  }

  // Check cache for each generator operation
  CompileDiagnosticOr<KGENCachingResult>
  checkCache(const CompilationState &state,
             const DenseMap<StringAttr, KGEN::GeneratorOp> &renaming,
             const std::string &keyHash) {
    ErrorOr<RCRef<Cache::BlobCacheBackend>> moggCacheBackendOr =
        Cache::getLocalDefaultBackendChain(
            std::filesystem::path(".mogg_cache") / keyHash);
    if (moggCacheBackendOr.isError())
      return (*state.kgenModule)->emitError(moggCacheBackendOr.getError());

    auto moggCache = RCRef<Cache::BlobCache<SemanticHashKey>>::create(
        moggCacheBackendOr.takeValue());

    // This is a list of ops that had a cache hit and the contents of the cache
    // blob, which is a key into the mojo cache. if we don't find this value in
    // the mojo cache, this op needs to be considered a cache miss
    llvm::StringMap<std::pair<llvm::SmallVector<mlir::StringAttr>, BufferRef>>
        mojoCacheHits;

    SmallVector<mlir::StringAttr> cacheMissSymbols;
    SmallVector<std::string> cacheMissHashes;
    DenseSet<KGEN::GeneratorOp> opsToErase;
    for (auto [oldName, genOp] : renaming) {
      auto genOpCopy = genOp;
      auto newName = genOpCopy.getSymNameAttr();
      std::string genOpHash;
      ErrorOr<std::optional<BufferRef>> moggCacheResult =
          moggCache->findSync(genOp, &genOpHash);
      if (moggCacheResult.isError()) {
#ifndef MODULAR_PRODUCTION
        (*state.kgenModule)
            ->emitWarning(
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
            state.mojoCache->findKeyedSync(mojoCacheHashKey);
        if (mojoCacheResultOr.isError() || !mojoCacheResultOr->has_value()) {
          // this is a cache miss!
          cacheMissSymbols.push_back(newName);
          cacheMissHashes.push_back(genOpHash);
          if (mojoCacheResultOr.isError()) {
#ifndef MODULAR_PRODUCTION
            (*state.kgenModule)
                ->emitWarning(mojoCacheResultOr.takeError()
                                  .get()); // Swallow the error,
                                           // because it's not critical
#endif
          }
          continue;
        }
        std::optional<BufferRef> maybeMojoCachedItemBuffer =
            mojoCacheResultOr.takeValue();

        // we have a real cache hit!
        BufferRef mojoCachedItemBuffer = maybeMojoCachedItemBuffer.value();

        // Collect the operation to erase later, after we're done iterating
        opsToErase.insert(genOp);
        auto [it, inserted] = mojoCacheHits.insert(
            {mojoCacheHashKey,
             {llvm::SmallVector<mlir::StringAttr>({newName}),
              std::move(mojoCachedItemBuffer)}});
        if (!inserted) {
          it->second.first.push_back(newName);
        }
      } else {
        cacheMissSymbols.push_back(newName);
        cacheMissHashes.push_back(genOpHash);
      }
    }

    // Erase operations after iteration to avoid use-after-free
    for (auto genOp : opsToErase)
      genOp.erase();

    KGENCachingResult result;
    result.cacheMissSymbols = std::move(cacheMissSymbols);
    result.cacheMissHashes = std::move(cacheMissHashes);
    result.mojoCacheHits = std::move(mojoCacheHits);
    result.moggCache = std::move(moggCache);

    return result;
  }

  CompileDiagnosticOr<KGENCachingResult>
  erasePreviouslyCompiledKernels(const CompilationState &state) {
    auto regionSymbols = collectRegionSymbols(state.mgpModule);
    auto stubSymbols = collectStubSymbols(*state.kgenModule);
    llvm::SetVector<StringAttr> symbolsToRename(regionSymbols.begin(),
                                                regionSymbols.end());
    for (auto &&stubSymbol : stubSymbols)
      symbolsToRename.insert(stubSymbol);
    auto renamedRegions = renameGenerators(*state.kgenModule, symbolsToRename);
    for (auto &&stubSymbol : stubSymbols)
      renamedRegions.erase(stubSymbol);

    updateMGPCallSites(state.mgpModule, renamedRegions);

    auto keyHashOr = buildCacheKey(*state.kgenModule);
    if (keyHashOr.isError())
      return keyHashOr.takeError();
    auto keyHash = keyHashOr.takeValue();

    return checkCache(state, renamedRegions, keyHash);
  }

  // Parse Mojo text from resources
  ErrorOr<OwningOpRef<ModuleOp>>
  parseMojoText(MLIRContext *ctx, AsyncRT::Runtime &runtime,
                const KGEN::CompilationOptions &opts) {
    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        getOperation()->getContext());
    auto mojoTextResource = resourceManager.getBlobManager().lookup("mojo");
    if (!mojoTextResource)
      return OwningOpRef<ModuleOp>();

    auto mojoTextRef = mojoTextResource->getBlob()->getData();
    auto mojoText = StringRef(mojoTextRef.data(), mojoTextRef.size());

    auto mojoTextBuffer = llvm::MemoryBuffer::getMemBufferCopy(mojoText);
    llvm::SourceMgr src;
    src.AddNewSourceBuffer(std::move(mojoTextBuffer), llvm::SMLoc());

    // Mojo parsing
    KGEN::LIT::ParserConfig config(ctx, opts);
    mlir::TimingScope ts;
    auto parsedMojoModule = KGEN::LIT::importMojoFile(runtime, src, config, ts);
    if (!parsedMojoModule)
      return Error("The GC could not parse the generated Mojo code");

    return parsedMojoModule;
  }

  // Setup compilation options
  KGEN::CompilationOptions
  setupCompilationOptions(TargetInfoAttr targetInfo,
                          ArrayRef<std::string> kernelLibraryPaths) {
    KGEN::CompilationOptions opts;
    opts.targetAccelerator = targetInfo.getAcceleratorArch();
    for (const auto &libraryPath : kernelLibraryPaths) {
      opts.extraSearchPaths.push_back(
          std::filesystem::path(libraryPath).parent_path().string());
    }
    opts.saveTempsPrefix = saveTempPrefix.getValue();
    opts.runMOGGPreElab = false;
    return opts;
  }

  // Extract and set bitcode libraries from packages
  void extractBitcodeLibs(MLIRContext *ctx, ModuleOp parsedMojoModule) {
    SmallVector<KGEN::LLVMBitcodeLibAttr> bitcodeLibAttrs;
    parsedMojoModule.walk([&](KGEN::LIT::PackageOp packageOp) {
      if (auto bitcodeModules = packageOp.getExternLLVMBitcodeModulesAttr()) {
        for (auto bitcodeAttr : bitcodeModules) {
          KGEN::LLVMBitcodeLibAttr libAttr =
              KGEN::LLVMBitcodeLibAttr::get(false, bitcodeAttr);
          bitcodeLibAttrs.push_back(libAttr);
        }
      }
    });
    if (!bitcodeLibAttrs.empty()) {
      KGEN::LLVMBitcodeLibArrayAttr arrayAttr =
          KGEN::LLVMBitcodeLibArrayAttr::get(ctx, bitcodeLibAttrs);
      parsedMojoModule->setAttr(
          KGEN::LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName(), arrayAttr);
    }
  }

  // Run LIT to KGEN pipeline
  CompileDiagnosticOrSuccess runLITToKGEN(KGEN::KGENCompiler &compiler,
                                          ModuleOp parsedMojoModule) {
    if (compiler.runGenerateLibraryPipeline(parsedMojoModule).isError()) {
      return parsedMojoModule.emitError(
          "The graph compiler could not lower LIT to KGEN");
    }
    return success();
  }

  // Handle cache misses: elaborate and compile to LLVM
  CompileDiagnosticOrSuccess
  handleCacheMisses(CompilationState &state,
                    const KGENCachingResult &cachingResult) {
    // KGENToElaboratedKGEN
    if (state.compiler
            ->runElaborationPipeline(
                *state.kgenModule, state.targetInfo, *state.runtime,
                state.runtime->getReadyChain().copy(),
                Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
                    "JitCompileKernels::elaborateKGEN",
                    "mogg.elaborate.kgen.cache.miss.time",
                    {{"pipeline", "JitCompileKernels::elaborateKGEN"}}),
                Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
                    "JitCompileKernels::elaborateKGEN"))
            .isError()) {
      return (*state.kgenModule)
          ->emitError(
              "The graph compiler could not elaborate the generated KGEN");
    }

    // KGENToLLVM
    std::string outKeyHash;
    ErrorOr<BufferRef> bufferOr = state.objectCompiler->emitArchive(
        std::move(state.kgenModule), /*emitAssembly=*/false, &outKeyHash);
    if (bufferOr.isError()) {
      return state.mgpModule->emitError(
          "An error occurred while lowering KGEN to LLVM "
          "in the graph compiler");
    }
    BufferRef buffer = bufferOr.takeValue();
    DenseResourceElementsAttr archiveAttr =
        createArchiveAttr(state.ctx, std::move(buffer), outKeyHash);
    DenseResourceElementsHandle binaryBlobHandle = archiveAttr.getRawHandle();
    StringRef binaryBlobKey = binaryBlobHandle.getKey();
    binaryBlobKey.consume_front("archive_");
    for (const std::string &hash : cachingResult.cacheMissHashes) {
      BufferRef buffer = Buffer::get(binaryBlobKey);
      if (auto error = cachingResult.moggCache->insertKeyedSync(hash, buffer)) {
#ifndef MODULAR_PRODUCTION
        state.mgpModule->emitWarning(
            error.takeError()
                .get()); // Swallow the error, because it's not critical
#endif
      }
    }

    ArrayAttr symbolsAttr = mlir::ArrayAttr::get(
        state.ctx,
        llvm::to_vector_of<mlir::Attribute>(cachingResult.cacheMissSymbols));
    for (auto &&modelOp : state.mgpModule.getOps<MGP::ModelOp>()) {
      state.builder.setInsertionPointToStart(modelOp.getInitBlock());
      MOGG::LoadCompiledKernelOp::create(
          state.builder, state.mgpModule->getLoc(), /*symbols=*/symbolsAttr,
          /*value=*/archiveAttr,
          /*debuginfo=*/debugKGENStubs ||
              hasDebuginfo(
                  KernelBuildType::kRelease // TODO: Support the other types.
                  ))
          ->setAttr("cache_miss", state.builder.getUnitAttr());
    }

    return success();
  }

  // Handle cache hits: load from cache
  void handleCacheHits(CompilationState &state,
                       const KGENCachingResult &cachingResult) {
    for (const auto &[blobHash, symbolsAndBufferPair] :
         cachingResult.mojoCacheHits) {
      ArrayAttr symbolsAttr = state.builder.getArrayAttr(
          llvm::to_vector_of<Attribute>(symbolsAndBufferPair.first));
      DenseResourceElementsAttr archiveAttr =
          createArchiveAttr(state.ctx, symbolsAndBufferPair.second, blobHash);
      for (auto &&modelOp : state.mgpModule.getOps<MGP::ModelOp>()) {
        state.builder.setInsertionPointToStart(modelOp.getInitBlock());
        MOGG::LoadCompiledKernelOp::create(
            state.builder, state.mgpModule->getLoc(), /*symbols=*/symbolsAttr,
            /*value=*/archiveAttr,
            /*debuginfo=*/debugKGENStubs ||
                hasDebuginfo(
                    KernelBuildType::kRelease // TODO: Support the other types.
                    ))
            ->setAttr("cache_hit", state.builder.getUnitAttr());
      }
    }
  }

  CompileDiagnosticOrSuccess
  compileMojoToBinary(ModuleOp module,
                      ArrayRef<std::string> kernelLibraryPaths) {
    auto ctx = module.getContext();
    auto targetInfoAttr = getTargetInfo(ctx);
    if (targetInfoAttr.isError()) {
      return module->emitError("Could not retrieve target info: ")
             << targetInfoAttr.getError();
    }

    AsyncRT::Runtime &runtime = *loadContext(ctx)->get<AsyncRT::Runtime>();
    auto targetInfo = targetInfoAttr.takeValue();
    auto opts = setupCompilationOptions(targetInfo, kernelLibraryPaths);

    auto parsedMojoModuleOr = parseMojoText(ctx, runtime, opts);
    if (parsedMojoModuleOr.isError()) {
      return module->emitError(
          "The graph compiler could not parse its Mojo generated code");
    }
    auto parsedMojoModule = parsedMojoModuleOr.takeValue();
    if (!parsedMojoModule)
      return success();

    setTargetInfo(*parsedMojoModule, targetInfo);

    // Set the mojo defines (e.g., LOGGING_LEVEL) on the parsed Mojo module
    // so they're available during elaboration.
    auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
    KGEN::extendWithModularEnvAttr(*parsedMojoModule, compileCtx);

    extractBitcodeLibs(ctx, *parsedMojoModule);

    // LITToKGEN
    KGEN::KGENCompiler compiler(*ctx, opts);
    auto litResult = runLITToKGEN(compiler, *parsedMojoModule);
    if (litResult.isError())
      return litResult.takeError();

    auto objectCompiler = KGEN::ObjectCompiler::create(
        ".mojo_cache", opts, /*isJit=*/true, *ctx, {});
    if (objectCompiler.isError()) {
      return module->emitError("The graph compiler could not create an "
                               "object compiler for KGEN to LLVM lowering");
    }

    CompilationState state(ctx, module, std::move(parsedMojoModule),
                           (*objectCompiler).get(), &compiler, targetInfo,
                           &runtime, opts);

    auto cachingResultOr = erasePreviouslyCompiledKernels(state);
    if (cachingResultOr.isError()) {
      return module->emitError("The graph compiler encountered an error while "
                               "caching previously compiled kernels");
    }
    auto cachingResult = cachingResultOr.takeValue();

    if (!cachingResult.cacheMissSymbols.empty()) {
      auto cacheMissResult = handleCacheMisses(state, cachingResult);
      if (cacheMissResult.isError())
        return cacheMissResult.takeError();
    }

    handleCacheHits(state, cachingResult);

    return success();
  }
};

} // namespace

} // namespace M::MOGG

std::unique_ptr<mlir::Pass> M::MOGG::createJitCompileKernels(
    const JitCompileKernelsOptions &opts,
    std::shared_ptr<MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<JitCompileKernels>(opts, std::move(libraryAnalysis));
}
