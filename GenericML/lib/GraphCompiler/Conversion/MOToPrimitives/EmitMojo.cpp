//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GenericML/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "GenericML/GraphCompiler/MOGGDialect/Support/LowerABIHelper.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "Support/CommonCLOptions.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <filesystem>

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_EMITMOJO
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct EmitMojo
    : public MOGG::MojoLibraryAnalysisPass<impl::EmitMojoBase<EmitMojo>> {
  using Base::Base;

  /// TODO: If this is slow, parallelize over KernelStubOp (will need to take
  /// the first dummyDimAttr from any kernel and then run in //).
  void stripDeviceIdsAndDynamicShapes() {
    TypedAttr dummyDimAttr = nullptr;
    auto moduleOp = getOperation();
    auto ctx = moduleOp->getContext();
    for (auto &&kernelStub : moduleOp.getOps<MOGG::KernelStubOp>()) {
      mlir::AttrTypeReplacer walker;

      // Replace all device reference IDs with the same ID. This is because the
      // ID itself does not contribute to the emitted Mojo but we could hash on
      // it.
      walker.addReplacement([&](M::DeviceRefAttr deviceRef) {
        return M::DeviceRefAttr::get(kernelStub->getContext(),
                                     deviceRef.getLabel(), 0);
      });

      // Replace all dynamic shape parameters into the same parameter. This is
      // because the parameter does not contribute to the emitted Mojo but we
      // could hash on it. It always lowers to Dim().
      walker.addReplacement([&](MOSH::ShapeAttr shapeAttr) {
        if (shapeAttr.isStatic())
          return shapeAttr;

        SmallVector<TypedAttr> newShape;
        for (auto &&dimAttr : shapeAttr.getValues()) {
          if (isa<IntegerAttr>(dimAttr)) {
            newShape.push_back(dimAttr);
          } else {
            if (!dummyDimAttr)
              dummyDimAttr = dimAttr;
            newShape.push_back(dummyDimAttr);
          }
        }
        return MOSH::ShapeAttr::get(ctx, newShape, shapeAttr.getType());
      });

      walker.recursivelyReplaceElementsIn(kernelStub, true, true, true);
    }
  }

  void emitMojoText() {
    auto moduleOp = getOperation();
    auto ctx = moduleOp->getContext();
    auto loc = moduleOp->getLoc();
    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(loc, llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                                   getPassName(), analysisOr.getError()));
      return signalPassFailure();
    }

    OpBuilder builder(ctx);
    MOGG::EmitMojoState state(*analysisOr.takeValue(), enableOpTraceComments,
                              enableComputeLambdas,
                              moggDebugDumpFusedTensorSpecs);

    // Import mojo stdlib packages and compiler directives.
    for (auto package : MOGG::HardcodedMojoSymbols::packagesToImport)
      state.builder.import(package, "*");

    // Import mojo packages for the custom kernel libraries.
    for (const auto &kernelLibraryPath : kernelLibraryPaths) {
      auto moduleName = std::filesystem::path(kernelLibraryPath).stem();
      state.builder.import(moduleName.string(), "*");
      // Store the package name for opaque type resolution.
      state.customKernelPackages.push_back(moduleName.string());
    }

    // Emit mojo for all the kernel stubs that exist in the current module.
    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
      if (kernelStub.emitMojo(state).failed()) {
        signalPassFailure();
        return;
      }
    }

    // Store the emitted text into a resource.
    auto mojoText = state.builder.str();

#ifndef MODULAR_PRODUCTION
    // Save the emitted Mojo code to a file if max.temps_dir is set.
    if (auto outFile = openIntermediateTextFile("emitted", ".mojo")) {
      outFile->os() << mojoText;
      outFile->keep();
    }
#endif

    auto attrType =
        ArrayType::get((int64_t)mojoText.size(),
                       IntegerType::get(ctx, 8, IntegerType::Unsigned));
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
        {mojoText.data(), mojoText.size()});
    auto resourceManager =
        DenseResourceElementsHandle::getManagerInterface(ctx);
    auto &blobManager = resourceManager.getBlobManager();

    // If there is already a mojo resource, it's probably because of constant
    // folding. Update it instead of inserting a new one that would have a
    // different name.
    DenseResourceElementsAttr mojoTextAttr;
    static constexpr llvm::StringLiteral mojoResourceName = "mojo";
    if (blobManager.lookup(mojoResourceName)) {
      blobManager.update(mojoResourceName, std::move(blob));
      auto blobHandle = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>(
          blobManager.lookup(mojoResourceName),
          cast<mlir::BuiltinDialect>(resourceManager.getDialect()));
      mojoTextAttr = DenseResourceElementsAttr::get(attrType, blobHandle);
    } else {
      mojoTextAttr = DenseResourceElementsAttr::get(attrType, mojoResourceName,
                                                    std::move(blob));
    }

    // For each model, add a reference to the Mojo symbols in
    // LoadCompiledKernelOp.
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      llvm::SetVector<StringRef> kernelsToLoadInThisModel;
      modelOp.walk([&](MOGG::ExecuteCompiledOp callOp) {
        kernelsToLoadInThisModel.insert(callOp.getCalleeSymbol());
      });

      if (kernelsToLoadInThisModel.empty())
        continue;

      builder.setInsertionPointToStart(modelOp.getInitBlock());
      /// TODO: Add debug info support. For now, defaults to false.
      MOGG::LoadCompiledKernelOp::create(
          builder, loc,
          builder.getArrayAttr(llvm::map_to_vector(
              kernelsToLoadInThisModel,
              [&](auto name) {
                return cast<Attribute>(StringAttr::get(ctx, name));
              })),
          mojoTextAttr, /*debug*/ false);
    }

    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
      kernelStub->erase();
    }
  }

  // This pass emits Mojo text. [optional] phases exist strictly as compilation
  // time optimizations.
  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto moduleOp = getOperation();

    // [optional]
    // Strip device IDs (not the labels like cpu or gpu, just the numbers) and
    // dynamic shape parameters because those mean nothing when emitting Mojo
    // and thus they should not be an obstacle to deduplication.
    stripDeviceIdsAndDynamicShapes();

    // [optional]
    // Merge identical kernel stubs using their hashes.
    mergeIdenticalKernelStubs(moduleOp);

    // The following are a series of passes that run nested within the current
    // pass because we have to disable the verifier after stripping device
    // attributes.
    MLIRContext *ctx = &getContext();
    mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);

    // We disable the verifier because we have stripped device attributes. The
    // IR is in an invalid state and we do not want the verifiers to catch that
    // as we run SymbolDCE.
    pm.enableVerifier(false);

    // [optional]
    // Any remaining kernel that has the lowering ops of an index.constant op
    // will have that code replaced back with the index.constant because it's
    // fewer lines of Mojo.
    pm.addPass(MGP::createRaiseBackIndexConstant());

    // TODO: FixupOverhangsPass doesn't work well if ran before this pass
    // without EmitMojo enabled. Once the transition to EmitMojo is complete, we
    // should maybe move some of these passes to GraphCompiler.cpp.
    // Canonicalizer should follow suit.
    pm.addPass(MGP::createFixupOverhangsPass());
    pm.addNestedPass<MOGG::KernelStubOp>(mlir::createCanonicalizerPass());

    // Eliminate all stubs that have no callers (because of
    // mergeIdenticalKernelStubs).
    pm.addPass(mlir::createSymbolDCEPass());

    // This pass is run nested in the current pass because
    // recoverIndexConstantFromUniqueKernels can cause a kernel to become
    // non-raising again. This causes Mojo compiler warning because we would
    // have a try-except clause around non-raising code.
    pm.addPass(MGP::createAddTryExceptBlocks());

    if (pm.run(moduleOp).failed())
      return signalPassFailure();

    emitMojoText();
  }
};

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createEmitMojo(
    const EmitMojoOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<EmitMojo>(opts, std::move(libraryAnalysis));
}
