//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"
#include "GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/Support/Constants.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/CacheLog.h"
#include "Support/HashUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Telemetry/Telemetry.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_JITCOMPILEKERNELS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

namespace {

// Convert all possible operations directly to the available operation.
struct JitCompileKernels
    : public impl::JitCompileKernelsBase<JitCompileKernels> {
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
    AsyncRT::CPUDevice *cpuDevice;
    KGEN::CompilationOptions opts;
    OpBuilder builder;
    RCRef<Cache::TransformCache> mojoCache;

    CompilationState(MLIRContext *ctx, ModuleOp mgpModule,
                     OwningOpRef<ModuleOp> kgenModule,
                     KGEN::ObjectCompiler *objectCompiler,
                     KGEN::KGENCompiler *compiler, TargetInfoAttr targetInfo,
                     AsyncRT::CPUDevice *cpuDevice,
                     KGEN::CompilationOptions opts)
        : ctx(ctx), mgpModule(mgpModule), kgenModule(std::move(kgenModule)),
          objectCompiler(objectCompiler), compiler(compiler),
          targetInfo(targetInfo), cpuDevice(cpuDevice), opts(std::move(opts)),
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

    // Resolve the key's target the same way codegen does (honoring an explicit
    // accelerator override); otherwise a host-warmed artifact would falsely hit
    // for a cross-compiled target.
    auto hostTargetOr = MOGG::getHostTargetInfoAttr(
        ctx, defaultDeviceTargetArch, defaultDeviceTargetTriple);
    if (hostTargetOr.isError())
      return kgenModuleOp->emitError() << hostTargetOr.takeError();
    if (auto error = MOGG::graphCompilerCacheKey(
            *keyBufRef, hostTargetOr.takeValue(),
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
    MODULAR_CACHE_LOG("mogg") << "checkCache: keyHash=" << keyHash
                              << " kernelCount=" << renaming.size() << "\n";

    ErrorOr<RCRef<Cache::BlobCacheBackend>> moggCacheBackendOr =
        Cache::getLocalDefaultBackendChain(
            std::filesystem::path(".mogg_cache") / keyHash);
    if (moggCacheBackendOr.isError()) {
      MODULAR_CACHE_LOG("mogg")
          << "checkCache: failed to create MOGG cache backend: "
          << moggCacheBackendOr.getError() << "\n";
      return (*state.kgenModule)->emitError(moggCacheBackendOr.getError());
    }

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
    size_t moggHitCount = 0, moggMissCount = 0;
    size_t mojoHitCount = 0, mojoMissCount = 0;
    for (auto [oldName, genOp] : renaming) {
      auto genOpCopy = genOp;
      auto newName = genOpCopy.getSymNameAttr();
      std::string genOpHash;
      ErrorOr<std::optional<BufferRef>> moggCacheResult =
          moggCache->findSync(genOp, &genOpHash);
      if (moggCacheResult.isError()) {
        MODULAR_CACHE_LOG("mogg")
            << "checkCache: MOGG lookup error for " << newName.getValue()
            << ": " << moggCacheResult.getError() << "\n";
#ifndef MODULAR_PRODUCTION
        (*state.kgenModule)
            ->emitWarning(
                moggCacheResult.takeError()
                    .get()); // Swallow the error, because it's not critical
#endif
      }

      if (moggCacheResult->has_value()) {
        ++moggHitCount;
        // we have a potential cache hit!
        BufferRef buffer = moggCacheResult->value();
        llvm::StringRef mojoCacheHashKey = buffer->getBuffer();

        // now we need to find the value in the mojo cache
        ErrorOr<std::optional<BufferRef>> mojoCacheResultOr =
            state.mojoCache->findKeyedSync(mojoCacheHashKey);
        if (mojoCacheResultOr.isError() || !mojoCacheResultOr->has_value()) {
          // this is a cache miss!
          ++mojoMissCount;
          MODULAR_CACHE_LOG("mogg") << "checkCache: MOGG hit but Mojo miss for "
                                    << newName.getValue() << "\n";
          cacheMissSymbols.push_back(newName);
          cacheMissHashes.push_back(genOpHash);
          if (mojoCacheResultOr.isError()) {
            MODULAR_CACHE_LOG("mogg")
                << "checkCache: Mojo lookup error for " << newName.getValue()
                << ": " << mojoCacheResultOr.getError() << "\n";
#ifndef MODULAR_PRODUCTION
            (*state.kgenModule)
                ->emitWarning(mojoCacheResultOr.takeError()
                                  .get()); // Swallow the error,
                                           // because it's not critical
#endif
          }
          continue;
        }
        ++mojoHitCount;
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
        ++moggMissCount;
        MODULAR_CACHE_LOG("mogg")
            << "checkCache: MOGG miss for " << newName.getValue() << "\n";
        cacheMissSymbols.push_back(newName);
        cacheMissHashes.push_back(genOpHash);
      }
    }

    MODULAR_CACHE_LOG("mogg")
        << "checkCache summary: " << renaming.size() << " kernels, "
        << "MOGG hit=" << moggHitCount << " miss=" << moggMissCount << ", "
        << "Mojo hit=" << mojoHitCount << " miss=" << mojoMissCount << "\n";

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
    // Stub semantic naming is intentionally disabled for now.
    //
    // Historical behavior:
    // - `applySemanticNaming()` used to see both top-level `region_X` entry
    //   points and GC-generated `<input>::stub_N` helpers.
    // - That gave us some extra KGEN-level dedup for identical stub bodies.
    //
    // Why it is disabled today:
    // - The temporary trait-based fusion lowering emits module-scope
    //   `FusionStruct_*` declarations and methods instead of the older inline
    //   closure bodies.
    // - `ModuleHasher::setGeneratorHash()` intentionally normalizes
    //   `FlatSymbolRefAttr` inside attributes and types so generator names do
    //   not perturb semantic hashes.
    // - For fusion stubs, the only distinction between two otherwise-identical
    //   stub bodies may be the nominal `FusionStruct_*` type they bind.
    // - After normalization, those nominal references disappear from the
    //   stub bytecode. The two stubs can then hash identically and be merged.
    // - Once those stubs are merged, their callers can also collapse through
    //   `setInputsHash()`, which turns this from a missed optimization into
    //   wrong code.
    //
    // Why the blunt exclusion is acceptable:
    // - `region_X` generators are the cacheable entry points; stubs do not
    //   participate directly in cache lookup.
    // - Whole-stub dedup still happens earlier at the MOGG level in
    //   `MergeIdenticalKernelStubs`, where the original BindLambda regions are
    //   still available structurally.
    // - This keeps the workaround local and easy to remove once the fusion
    //   lowering stops depending on nominal struct identity.
    //
    // How to bring stub renaming/dedup back:
    // - When fusion moves off these temporary structs and onto unified closures
    //   again, the distinguishing fusion semantics should be represented in the
    //   generator body or in stable generator-to-generator edges rather than in
    //   nominal `FusionStruct_*` symbols.
    // - At that point, uncomment the `collectStubSymbols()` line above, insert
    //   those symbols into `symbolsToRename` below, and delete this comment
    //   block.
    // - If unified closures still leave semantically-important identity only in
    //   non-generator symbol references, we should fix that representation
    //   first instead of re-enabling stub semantic naming.
    // OLD LINE:
    // auto stubSymbols = collectStubSymbols(*state.kgenModule);

    llvm::SetVector<StringAttr> symbolsToRename(regionSymbols.begin(),
                                                regionSymbols.end());
    auto renamedRegions = renameGenerators(*state.kgenModule, symbolsToRename);

    updateMGPCallSites(state.mgpModule, renamedRegions);

    auto keyHashOr = buildCacheKey(*state.kgenModule);
    if (keyHashOr.isError())
      return keyHashOr.takeError();
    auto keyHash = keyHashOr.takeValue();

    return checkCache(state, renamedRegions, keyHash);
  }

  // Parse Mojo text from resources
  ErrorOr<OwningOpRef<ModuleOp>>
  parseMojoText(MLIRContext *ctx, ContextRef maxContext,
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
    auto parsedMojoModule =
        KGEN::LIT::importMojoFile(maxContext, src, config, ts);
    if (!parsedMojoModule)
      return Error("The GC could not parse the generated Mojo code");

    return parsedMojoModule;
  }

  // Setup compilation options
  KGEN::CompilationOptions
  setupCompilationOptions(TargetInfoAttr targetInfo,
                          ArrayRef<std::string> kernelLibraryPaths) {
    KGEN::CompilationOptions opts;
    // opts (not the module target-info stamp) drives the ObjectCompiler, so the
    // virtual-CPU override must reach it too or the cache key and codegen
    // desync. Gated: propagating the host target unconditionally would churn
    // every JIT cache key (resolved features != raw host CPUID delta).
    if (!M::Driver::getVirtualCpuTarget().empty()) {
      opts.targetTriple = targetInfo.getTripleStr();
      opts.targetCpu = targetInfo.getArch();
      opts.targetFeatures = targetInfo.getFeatures();
    }
    opts.targetAccelerator = targetInfo.getAcceleratorArch();
    for (const auto &libraryPath : kernelLibraryPaths) {
      opts.extraSearchPaths.push_back(
          std::filesystem::path(libraryPath).parent_path().string());
    }
    opts.saveTempsPrefix = saveTempPrefix.getValue();
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
                *state.kgenModule, state.targetInfo, *state.cpuDevice,
                state.cpuDevice->getReadyChain().copy(),
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
        MODULAR_CACHE_LOG("mogg")
            << "handleCacheMisses: MOGG cache insert failed for hash=" << hash
            << ": " << error.getError() << "\n";
#ifndef MODULAR_PRODUCTION
        state.mgpModule->emitWarning(
            error.takeError()
                .get()); // Swallow the error, because it's not critical
#endif
      } else {
        MODULAR_CACHE_LOG("mogg")
            << "handleCacheMisses: MOGG cache insert success for hash=" << hash
            << "\n";
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
    auto targetInfoAttr =
        getTargetInfo(ctx, defaultDeviceTargetArch, defaultDeviceTargetTriple);
    if (targetInfoAttr.isError()) {
      return module->emitError("Could not retrieve target info: ")
             << targetInfoAttr.getError();
    }

    ContextRef maxContext = loadContext(ctx);
    auto cpuDevice = maxContext->get<AsyncRT::CPUDevice>();

    auto targetInfo = targetInfoAttr.takeValue();
    auto opts = setupCompilationOptions(targetInfo, kernelLibraryPaths);

    auto parsedMojoModuleOr = parseMojoText(ctx, maxContext, opts);
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

    MODULAR_CACHE_LOG("mogg")
        << "compileMojoToBinary: creating ObjectCompiler with "
           "basePath=" +
               KGEN::kMojoCacheBaseDirName + "\n";
    auto objectCompiler = KGEN::ObjectCompiler::create(
        KGEN::kMojoCacheBaseDirName, opts, /*isJit=*/true, *ctx, {});
    if (objectCompiler.isError()) {
      return module->emitError("The graph compiler could not create an "
                               "object compiler for KGEN to LLVM lowering");
    }

    CompilationState state(ctx, module, std::move(parsedMojoModule),
                           (*objectCompiler).get(), &compiler, targetInfo,
                           cpuDevice, opts);

    auto cachingResultOr = erasePreviouslyCompiledKernels(state);
    if (cachingResultOr.isError()) {
      return module->emitError("The graph compiler encountered an error while "
                               "caching previously compiled kernels");
    }
    auto cachingResult = cachingResultOr.takeValue();

    if (!cachingResult.cacheMissSymbols.empty()) {
      MODULAR_CACHE_LOG("mogg")
          << "compileMojoToBinary: compiling "
          << cachingResult.cacheMissSymbols.size() << " cache-miss kernels\n";
      auto cacheMissResult = handleCacheMisses(state, cachingResult);
      if (cacheMissResult.isError())
        return cacheMissResult.takeError();
    } else {
      MODULAR_CACHE_LOG("mogg")
          << "compileMojoToBinary: all kernels cached, no compilation "
             "needed\n";
    }

    handleCacheHits(state, cachingResult);

    return success();
  }
};

} // namespace

} // namespace M::MOGG
