//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/KGENStubCompiler.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/MOGGCompiler.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "Kernels/mojo/Mogg/MOGGLoader.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "KGEN/ToolCommon/KGENPasses.h"
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

    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(getOperation().getLoc(),
                llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                              getPassName(), analysisOr.getError()));
      return signalPassFailure();
    }
    MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis = *analysisOr.takeValue();

    initLLVM();

    if (MOGG::useExperimentalMojoPath()) {
      if (compileMojoToBinary(module, mojoLibraryAnalysis, kernelLibraryPaths)
              .isError())
        signalPassFailure();
    }

    if (compileKGENToBinary(module, mojoLibraryAnalysis).isError())
      signalPassFailure();
  }

  CompileDiagnosticOrSuccess
  compileKGENToBinary(ModuleOp module,
                      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis) {
    CompileDiagnosticOrSuccess errOr = compileToBinary(
        /*moduleOp=*/module,
        /*mojoLibraryAnalysis=*/mojoLibraryAnalysis,
        /*dumpStub=*/dumpStub.getValue(),
        /*createMOGGReproducers=*/createMOGGReproducers.getValue(),
        /*saveTempPrefix=*/saveTempPrefix.getValue(),
        /*minimumCpuAlignment=*/minimumCpuAlignment.getValue(),
        /*enableTiming=*/timeMOGGCompile.getValue(),
        /*dryRun=*/dryRun.getValue(),
        /*debugKGENStubs=*/debugKGENStubs.getValue(),
        /*kernelBuildType=*/KernelBuildType::kRelease);

    if (errOr.isError())
      return errOr.takeError();

    for (Operation &op : llvm::make_early_inc_range(module.getOps())) {
      if (isa<KGEN::GeneratorOp, KGEN::ExternGeneratorOp,
              KGEN::StructGeneratorOp>(op)) {
        op.erase();
      }
    }

    return success();
  }

  CompileDiagnosticOrSuccess
  compileMojoToBinary(ModuleOp module,
                      MOGG::MojoLibraryAnalysis &mojoLibraryAnalysis,
                      ArrayRef<std::string> kernelLibraryPaths) {
    auto ctx = module.getContext();
    auto targetInfoAttr = getTargetInfo(ctx);
    if (targetInfoAttr.isError()) {
      return module->emitError("Could not retrieve target info: ")
             << targetInfoAttr.getError();
    }

    auto resourceManager = DenseResourceElementsHandle::getManagerInterface(
        getOperation()->getContext());
    auto mojoTextResource = resourceManager.getBlobManager().lookup("mojo");
    if (!mojoTextResource)
      return success();

    auto mojoTextRef = mojoTextResource->getBlob()->getData();
    auto mojoText = StringRef(mojoTextRef.data(), mojoTextRef.size());

    auto mojoTextBuffer = llvm::MemoryBuffer::getMemBufferCopy(mojoText);
    llvm::SourceMgr src;
    src.AddNewSourceBuffer(std::move(mojoTextBuffer), llvm::SMLoc());
    AsyncRT::Runtime &runtime = *loadContext(ctx)->get<AsyncRT::Runtime>();
    KGEN::CompilationOptions opts;
    opts.targetAccelerator = targetInfoAttr->getAcceleratorArch();
    for (const auto &libraryPath : kernelLibraryPaths) {
      opts.extraSearchPaths.push_back(
          std::filesystem::path(libraryPath).parent_path().string());
    }

    // Mojo parsing
    KGEN::LIT::ParserConfig config(ctx, opts);
    mlir::TimingScope ts;
    auto parsedMojoModule = KGEN::LIT::importMojoFile(runtime, src, config, ts);
    if (!parsedMojoModule) {
      return module->emitError(
          "The graph compiler could not parse its Mojo generated code");
    }

    setTargetInfo(*parsedMojoModule, *targetInfoAttr);

    // Set the mojo defines (e.g., LOGGING_LEVEL) on the parsed Mojo module
    // so they're available during elaboration.
    auto *compileCtx = loadContext(ctx)->get<KGEN::CompilationContext>();
    KGEN::extendWithModularEnvAttr(*parsedMojoModule, compileCtx);

    // LITToKGEN
    opts.saveTempsPrefix = saveTempPrefix.getValue();
    opts.runMOGGPreElab = false;
    KGEN::KGENCompiler compiler(*ctx, opts);
    if (compiler.runGenerateLibraryPipeline(*parsedMojoModule).isError()) {
      return module->emitError(
          "The graph compiler could not lower LIT to KGEN");
    }

    // KGENToElaboratedKGEN
    if (compiler
            .runElaborationPipeline(
                *parsedMojoModule, *targetInfoAttr, runtime,
                runtime.getReadyChain().copy(),
                Cache::CacheTelemetryContext::getTelemetryOnMissLambda(
                    "JitCompileKernels::elaborateKGEN",
                    "mogg.elaborate.kgen.cache.miss.time",
                    {{"pipeline", "JitCompileKernels::elaborateKGEN"}}),
                Cache::CacheTelemetryContext::getTelemetryOnHitLambda(
                    "JitCompileKernels::elaborateKGEN"))
            .isError()) {
      return module->emitError(
          "The graph compiler could not elaborate the generated KGEN");
    }

    // KGENToLLVM
    auto objectCompiler = KGEN::ObjectCompiler::create(
        ".mojo_cache", opts, /*isJit=*/true, *ctx, {});
    if (objectCompiler.isError()) {
      return module->emitError("The graph compiler could not create an "
                               "object compiler for KGEN to LLVM lowering");
    }
    std::string outKeyHash;
    ErrorOr<BufferRef> bufferOr = objectCompiler.get()->emitArchive(
        std::move(parsedMojoModule), /*emitAssembly=*/false, &outKeyHash);
    if (bufferOr.isError()) {
      return module->emitError("An error occurred while lowering KGEN to LLVM "
                               "in the graph compiler");
    }

    // Archiving the compiled code and storing it as a dense resource.
    BufferRef buffer = bufferOr.takeValue();
    auto compiledBlob = createArchiveAttr(ctx, std::move(buffer), outKeyHash);
    DenseResourceElementsHandle binaryBlobHandle = compiledBlob.getRawHandle();
    StringRef binaryBlobKey = binaryBlobHandle.getKey();
    binaryBlobKey.consume_front("archive_");

    // Replace the resource in LoadCompiledKernelOp with the compiled Mojo.
    for (auto modelOp : module.getOps<MGP::ModelOp>()) {
      auto loadkernelOps =
          modelOp.getInitBlock()->getOps<MOGG::LoadCompiledKernelOp>();
      if (!loadkernelOps.empty()) {
        (*loadkernelOps.begin()).setValueAttr(compiledBlob);
      }
    }

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
