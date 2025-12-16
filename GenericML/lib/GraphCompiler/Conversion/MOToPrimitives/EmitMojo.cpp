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

#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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
      walker.addReplacement([&](M::DeviceRefAttr deviceRef) {
        return M::DeviceRefAttr::get(kernelStub->getContext(),
                                     deviceRef.getLabel(), 0);
      });
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

  /// Stubs that have only one caller were not deduplicated. Therefore, all the
  /// index.constant that were turned into mgp.buffer.constant +
  /// mgp.buffer.to_index should be raised back into index.constant to reduce
  /// the quantity of text that is emitted. Note that it is not required to
  /// update the constant resource attributes because they will just no longer
  /// be used.
  void recoverIndexConstantFromUniqueKernels() {
    auto moduleOp = getOperation();
    // Not using SymbolTable because it is very slow to call it on a
    // ModuleOp.
    llvm::DenseMap<StringAttr, SmallVector<MOGG::ExecuteCompiledOp>> symbolUses;
    for (auto &&modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      modelOp.walk([&](MOGG::ExecuteCompiledOp callOp) {
        symbolUses[callOp.getCalleeSymbolAttr().getAttr()].push_back(callOp);
      });
    }
    auto stubsRange = moduleOp.getOps<MOGG::KernelStubOp>();
    SmallVector<MOGG::KernelStubOp> kernels(stubsRange.begin(),
                                            stubsRange.end());
    mlir::parallelForEach(
        moduleOp->getContext(), kernels, [&](MOGG::KernelStubOp kernelStub) {
          if (kernelStub.getAbi() != MOGG::ABIKind::MGP)
            return;

          auto it = symbolUses.find(kernelStub.getSymNameAttr());
          if (it == symbolUses.end())
            return;

          if (it->second.size() != 1)
            return;

          auto resourceArr = it->second.front().getConstantResourcesAttr();

          OpBuilder builder(kernelStub);
          kernelStub.walk([&](MGP::BufferToIndexOp bufferToIndexOp) {
            auto resourceConstantOp =
                bufferToIndexOp.getInBuffer()
                    .getDefiningOp<MGP::ResourceConstantOp>();
            if (!resourceConstantOp)
              return;

            uint64_t slotIndex =
                resourceConstantOp.getResourceSlot().getZExtValue();
            auto resource = dyn_cast<AlignedBytesAttr>(resourceArr[slotIndex]);
            if (!resource)
              return;

            if (resource.size() != sizeof(int64_t))
              return;

            int64_t value =
                *reinterpret_cast<const int64_t *>(resource.getData().data());
            builder.setInsertionPoint(bufferToIndexOp);
            auto indexConstantOp = mlir::index::ConstantOp::create(
                builder, resourceConstantOp->getLoc(), value);
            bufferToIndexOp->replaceAllUsesWith(indexConstantOp);
          });
        });
  }

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    MLIRContext *ctx = &getContext();
    mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
        ctx, mlir::OpPassManager::Nesting::Implicit);
    // We disable the verifier because we have stripped device attributes. The
    // IR is in an invalid state and we do not want the verifiers to catch that
    // as we run SymbolDCE.
    pm.enableVerifier(false);
    // TODO: FixupOverhangsPass doesn't work well if ran before this pass
    // without EmitMojo enabled. Once the transition to EmitMojo is complete, we
    // should maybe move some of these passes to GraphCompiler.cpp
    pm.addPass(MGP::createFixupOverhangsPass());
    pm.addNestedPass<MOGG::KernelStubOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createSymbolDCEPass());
    pm.addPass(MGP::createAddTryExceptBlocks());

    auto moduleOp = getOperation();
    // Strip device IDs (not the labels like cpu or gpu, just the numbers) and
    // dynamic shape parameters because those mean nothing when emitting Mojo
    // and thus they should not be an obstacle to deduplication.
    stripDeviceIdsAndDynamicShapes();
    mergeIdenticalKernelStubs(moduleOp);

    // Any remaining kernel that has the lowering ops of an index.constant op
    // will have that code replaced back with the index.constant because it's
    // less emitted text.
    recoverIndexConstantFromUniqueKernels();

    if (pm.run(moduleOp).failed())
      return signalPassFailure();

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
};

} // namespace

} // namespace M::MGP

std::unique_ptr<mlir::Pass> M::MGP::createEmitMojo(
    const EmitMojoOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  return std::make_unique<EmitMojo>(opts, std::move(libraryAnalysis));
}
