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

  void runOnOperation() override {
    if (!MOGG::useExperimentalMojoPath())
      return;

    auto moduleOp = getOperation();
    MLIRContext *ctx = &getContext();
    auto loc = moduleOp->getLoc();

    auto analysisOr = getLibraryAnalysis();
    if (analysisOr.isError()) {
      emitError(loc, llvm::formatv("[{0}] Failed to load Mojo analysis: {1}",
                                   getPassName(), analysisOr.getError()));
      return signalPassFailure();
    }

    OpBuilder builder(ctx);
    SymbolTable symTable(moduleOp);
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
    }

    // Emit mojo for all the kernel stubs that exist in the current module.
    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
      if (kernelStub.emitMojo(state).failed()) {
        signalPassFailure();
        return;
      }
    }

    // Replace stub calls with ExecuteCompiledOp.
    llvm::DenseMap<StringAttr, llvm::SetVector<StringRef>>
        kernelsToLoadPerModel;
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      auto &kernelsToLoadInThisModel =
          kernelsToLoadPerModel[modelOp.getSymNameAttr()];
      modelOp.walk([&](MOGG::CallStubOp callOp) {
        kernelsToLoadInThisModel.insert(callOp.getName());
        // Create a new ExecuteCompiledOp to replace the original kernel
        auto insertPoint = builder.saveInsertionPoint();
        auto stubOp = callOp.getCallee();
        builder.setInsertionPoint(callOp);
        auto newOp = MOGG::ExecuteCompiledOp::create(
            builder, stubOp->getLoc(),
            /*resultTypes=*/callOp->getResultTypes(),
            /*operands=*/callOp->getOperands(),
            /*kernelName=*/callOp.getName(),
            /*usedParams=*/builder.getArrayAttr({}),
            /*outputParamDecls=*/
            KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
            /*summary=*/stubOp.getSummaryAttr(),
            /*isAsync=*/*stubOp.getIsAsync(),
            /*constantResources=*/stubOp.getConstantResources());
        builder.restoreInsertionPoint(insertPoint);

        newOp.setDeviceRef(stubOp.getExecutionDevice());
        callOp->replaceAllUsesWith(newOp.getResults());
        callOp->erase();
      });
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

    // For each model, add a reference to the Mojo from LoadCompiledKernelOp.
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      const auto &kernelsToLoadInThisModel =
          kernelsToLoadPerModel[modelOp.getSymNameAttr()];
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
