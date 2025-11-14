//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/LowerABIHelper.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_EMITMOJO
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

MOGG::ExecuteCompiledOp
lowerGenericCompiledRegionOp(MOGG::GenericCompiledRegionOp kernel,
                             const std::string &kernelName,
                             OpBuilder &builder) {
  // Create a new ExecuteCompiledOp to replace the original kernel
  auto insertPoint = builder.saveInsertionPoint();
  builder.setInsertionPoint(kernel);
  auto newOp = MOGG::ExecuteCompiledOp::create(
      builder, kernel->getLoc(),
      /*resultTypes=*/kernel->getResultTypes(),
      /*operands=*/kernel->getOperands(),
      /*kernelName=*/StringAttr::get(kernel->getContext(), kernelName),
      /*usedParams=*/builder.getArrayAttr({}),
      /*outputParamDecls=*/kernel.getOutputParamDecls(),
      /*summary=*/MOGG::getSummaryString(kernel),
      /*isAsync=*/*kernel.getIsAsync(),
      /*constantResources=*/kernel.getConstantResources());
  builder.restoreInsertionPoint(insertPoint);

  newOp.setDeviceRef(kernel.getExecutionDevice());
  kernel->replaceAllUsesWith(newOp.getResults());
  kernel->erase();

  return newOp;
}

struct EmitMojo
    : public MOGG::MojoLibraryAnalysisPass<impl::EmitMojoBase<EmitMojo>> {
  using Base::Base;

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(getOperation().getLoc(),
                llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                              getPassName(), analysisOr.getError()));
      return signalPassFailure();
    }

    OpBuilder builder(ctx);
    SymbolTable symTable(moduleOp);
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      DenseSet<StringAttr> kernelsToLoad;
      MOGG::EmitMojoState state(*analysisOr.takeValue(), enableOpTraceComments,
                                enableComputeLambdas,
                                moggDebugDumpFusedTensorSpecs);
      for (auto package : MOGG::HardcodedMojoSymbols::packagesToImport)
        state.builder.import(package, "*");
      // Import mojo packages for the custom kernel libraries.
      for (const auto &kernelLibraryPath : kernelLibraryPaths) {
        auto moduleName = std::filesystem::path(kernelLibraryPath).stem();
        state.builder.import(moduleName.string(), "*");
      }

      // First do all the stubs (in case they appear after their references).
      for (auto &&kernelStub : llvm::make_early_inc_range(
               moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
        // Skip if this stub is not used by this graph.
        if (symTable.getSymbolUses(kernelStub, modelOp)->empty())
          continue;

        if (kernelStub.emitMojo(state).failed()) {
          signalPassFailure();
          return;
        }
      }

      // Then do all the kernels that may reference them.
      auto lowerGenericCompiledRegionOps =
          [&](MOGG::GenericCompiledRegionOp kernel) {
            // If none of the ops emits Mojo, skip it.
            if (!kernel.getEmitsMojo())
              return WalkResult::skip();

            if (kernel.emitMojo(state).failed())
              return WalkResult::interrupt();

            auto kernelCall = lowerGenericCompiledRegionOp(
                kernel, state.currentKernelName, builder);
            kernelsToLoad.insert(kernelCall.getKernelNameAttr());

            return WalkResult::advance();
          };
      if (modelOp->walk(lowerGenericCompiledRegionOps).wasInterrupted()) {
        signalPassFailure();
        return;
      }

      // If no kernels were emitted, emit a single empty kernel_wrapper
      // to avoid runtime errors about missing function symbols.
      // TODO (GEX-2668): This is a bandaid fix to enable constant folded
      // subgraphs. Prior to this change attempting to run a model that compiled
      // with EMIT_MOJO=1 and had all ops constant folded away would lead to
      // this error:
      //
      // mt: setup for model add returned error: expecting at least
      // one function symbol to load
      //
      // We should properly fix this in the long term.
      if (kernelsToLoad.empty()) {
        state.createKernelWrapper([&](const MOGG::FunctionBody &) {
          state.builder.comment(
              "Empty kernel wrapper - no actual kernels to execute.");
          state.builder.pass();
        });
        kernelsToLoad.insert(StringAttr::get(ctx, state.currentKernelName));
      }

      // Store the emitted text into a resource and point to it from a
      // MojoModuleOp.
      auto mojoText = state.builder.str();
      auto attrType =
          ArrayType::get((int64_t)mojoText.size(),
                         IntegerType::get(ctx, 8, IntegerType::Unsigned));
      auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
          {mojoText.data(), mojoText.size()});
      auto resourceManager =
          DenseResourceElementsHandle::getManagerInterface(ctx);
      /// TODO: Find a better name than mojo, maybe a hash.
      auto mojoTextAttr = DenseResourceElementsAttr::get(
          attrType, resourceManager.insert("mojo", std::move(blob)));

      auto loc = moduleOp.getLoc();
      builder.setInsertionPointToStart(modelOp.getInitBlock());
      MOGG::MojoModuleOp::create(builder, loc, mojoTextAttr);
      /// TODO: Add debug info support. For now, defaults to false.
      MOGG::LoadCompiledKernelOp::create(
          builder, loc,
          builder.getArrayAttr(llvm::to_vector_of<Attribute>(kernelsToLoad)),
          mojoTextAttr, /*debug*/ false);

      if (dumpMojoText)
        llvm::errs() << mojoText << "\n";

      if (!saveTempPrefix.empty()) {
        std::string outPath = saveTempPrefix + ".kernels.mojo";
        auto outFile = mlir::openOutputFile(outPath);
        if (!outFile) {
          modelOp->emitError()
              << "Could not open pre-elaboration dump file named " << outPath;
          signalPassFailure();
          return;
        }

        outFile->os() << mojoText << "\n";
        outFile->keep();
      }
    }
    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
      kernelStub->erase();
    }
  }
};

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createEmitMojo(
    const EmitMojoOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<EmitMojo>(opts, std::move(libraryAnalysis));
}
