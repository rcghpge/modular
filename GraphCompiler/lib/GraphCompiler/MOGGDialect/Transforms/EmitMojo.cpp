//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/Conversion/MOGGToMGP.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/CommonCLOptions.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <filesystem>

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_EMITMOJO
#define GEN_PASS_DEF_LIFTBUFFERCACHESLOTS
#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

// Returns true if every use of `constOp` is as the slot operand of an
// `mgp.buffer.get_cached`. Used to decide whether the constant can be safely
// lifted to a kernel argument without breaking other ops that depend on its
// compile-time value.
static bool allUsesAreCachedBufferSlot(mlir::index::ConstantOp constOp) {
  for (mlir::Operation *user : constOp->getUsers()) {
    auto cached = mlir::dyn_cast<MGP::BufferGetCachedOp>(user);
    if (!cached || cached.getBBufferSlot() != constOp.getResult())
      return false;
  }
  return true;
}

} // namespace

// GEX-3685: Lift inline `index.constant` ops feeding into
// `mgp.buffer.get_cached` slot operands to kernel-stub function arguments,
// so that two kernels that differ only in the cached-buffer slot index get
// structurally identical bodies and `MergeIdenticalKernelStubs` can collapse
// them. Without this, kernels like `region_1078` and `region_1065` (a
// 537-line near-duplicate pair in gemma-4 vision) survive into Mojo emission
// as separate functions that the Mojo compiler then has to elaborate
// independently.
//
// Approach: for each MGP-ABI KernelStubOp body, find every `index.constant`
// whose sole uses are as the slot operand of `mgp.buffer.get_cached`.
// Promote each such constant to a new `index`-typed kernel argument; erase
// the constant; rewrite the kernel's `ExecuteCompiledOp` callers to
// materialize the constant at the call site and pass it as a new operand.
//
// Only MGP-ABI kernels are touched: `mgp.buffer.get_cached` lives in the
// MGP dialect and is only emitted into MGP-ABI kernel bodies (cf.
// `RaiseBackIndexConstant`, which gates on the same invariant), so a
// MOGG-ABI kernel can never have `index.constant` ops feeding `get_cached`.
// And the only caller op of an MGP-ABI kernel is `ExecuteCompiledOp` --
// `CallStubOp` / `CallStubAsyncOp` only target MOGG-ABI kernels (enforced
// by `CallStubOp::verifySymbolUses`).
void liftBufferCacheSlots(mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type indexType = mlir::IndexType::get(ctx);

  // Build symbol-name → callers map by walking ExecuteCompiledOp ops once.
  // CallStubOp / CallStubAsyncOp only target MOGG-ABI kernels and so cannot
  // be callers of the MGP-ABI kernels we transform here.
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *, 4>>
      callersByName;
  moduleOp.walk([&](MOGG::ExecuteCompiledOp exec) {
    callersByName[exec.getCalleeSymbolAttr().getAttr()].push_back(exec);
  });

  for (auto kernel : moduleOp.getOps<MOGG::KernelStubOp>()) {
    // See doc comment: only MGP-ABI kernel bodies can contain
    // `mgp.buffer.get_cached` and therefore the inline `index.constant`
    // ops we want to lift.
    if (kernel.getAbi() != MOGG::ABIKind::MGP)
      continue;
    mlir::Block &body = kernel.getBodyBlock();

    // Collect candidate index.constant ops (sole uses are
    // `mgp.buffer.get_cached`'s slot operand).
    llvm::SmallVector<mlir::index::ConstantOp, 8> toPromote;
    for (mlir::Operation &op : body) {
      auto constOp = mlir::dyn_cast<mlir::index::ConstantOp>(&op);
      if (!constOp || constOp->use_empty())
        continue;
      if (allUsesAreCachedBufferSlot(constOp))
        toPromote.push_back(constOp);
    }

    if (toPromote.empty())
      continue;

    // Capture values BEFORE mutating IR.
    auto values = llvm::map_to_vector(
        toPromote, [](auto cop) { return cop.getValue().getSExtValue(); });

    // Extend the kernel function type with one `index` argument per
    // promoted constant.
    auto fnType = kernel.getFunctionType();
    auto newInputs = llvm::to_vector(fnType.getInputs());
    newInputs.append(toPromote.size(), indexType);
    auto newFnType =
        mlir::FunctionType::get(ctx, newInputs, fnType.getResults());
    kernel.setFunctionTypeAttr(mlir::TypeAttr::get(newFnType));

    // Add Block arguments matching the new function inputs, redirect uses,
    // and erase the original constants.
    for (auto cop : toPromote) {
      mlir::BlockArgument newArg = body.addArgument(indexType, cop->getLoc());
      cop.getResult().replaceAllUsesWith(newArg);
      cop->erase();
    }

    // Update every caller to pass the original constant values as new
    // trailing operands.
    auto it = callersByName.find(kernel.getSymNameAttr());
    if (it == callersByName.end())
      continue;
    for (mlir::Operation *callOp : it->second) {
      mlir::OpBuilder builder(callOp);
      llvm::SmallVector<mlir::Value, 8> newCallOperands;
      for (int64_t value : values) {
        auto cst =
            mlir::index::ConstantOp::create(builder, callOp->getLoc(), value);
        newCallOperands.push_back(cst.getResult());
      }
      callOp->insertOperands(callOp->getNumOperands(), newCallOperands);
    }
  }
}

namespace {

struct LiftBufferCacheSlots
    : public impl::LiftBufferCacheSlotsBase<LiftBufferCacheSlots> {
  using LiftBufferCacheSlotsBase::LiftBufferCacheSlotsBase;
  void runOnOperation() override { liftBufferCacheSlots(getOperation()); }
};

struct EmitMojo : public impl::EmitMojoBase<EmitMojo> {
  using EmitMojoBase::EmitMojoBase;

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
          if (auto staticDim = MOSH::ShapeAttr::getStaticDim(dimAttr)) {
            newShape.push_back(dimAttr);
          } else {
            if (!dummyDimAttr) {
              dummyDimAttr =
                  KGEN::ParamDeclRefAttr::get("dyn", dimAttr.getType());
            }
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
    OpBuilder builder(ctx);
    MojoRegistry registry(moduleOp);
    MOGG::EmitMojoState state(registry, enableOpTraceComments,
                              enableComputeLambdas,
                              moggDebugDumpFusedTensorSpecs);
    state.moduleOp = moduleOp;

    // Import mojo stdlib packages and compiler directives.
    for (auto [package, as] : MOGG::kPackagesToImport) {
      if (as.empty())
        state.builder.import(package);
      else
        state.builder.importAs(package, as);
    }

    // Import mojo packages for the custom kernel libraries.
    for (const auto &kernelLibraryPath : kernelLibraryPaths) {
      auto moduleName = std::filesystem::path(kernelLibraryPath).stem();
      state.builder.import(moduleName.string());
    }

    // Each emitted kernel call uses the kernel's fully-qualified path
    // (`<module>.<Struct>.<method>`). The library imports above only bind the
    // top-level package, not the submodule the kernel lives in, so under
    // submodule-visibility gating that path isn't navigable. Import each
    // kernel's module explicitly. A `KernelDeclOp`'s `sym_name` is the struct's
    // fully-resolved name (`<module>.<Struct>`), so the module is everything
    // before the final segment.
    llvm::StringSet<> importedKernelModules;
    for (auto kernelDecl : moduleOp.getOps<MOGG::KernelDeclOp>()) {
      auto [moduleName, structName] =
          llvm::StringRef(kernelDecl.getSymName()).rsplit('.');
      if (structName.empty())
        continue;
      if (importedKernelModules.insert(moduleName).second)
        state.builder.import(moduleName.str());
    }

    state.builder.newline();

    // Emit mojo for all the kernel stubs that exist in the current module.
    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::KernelStubOp>())) {
      if (kernelStub.emitMojo(state).failed()) {
        signalPassFailure();
        return;
      }
    }

    // Append any deferred struct definitions (fusion structs emitted by
    // BindLambdaOp). Mojo processes all module-scope declarations before
    // compiling function bodies, so ordering doesn't matter.
    auto mojoText = state.builder.str() + state.structDefsBuilder.str();

    // Compute a base file name for the dump from the names of non-subgraph
    // ModelOps (user-visible model names) joined by "+", or "graph" when there
    // are none. Mirrors the `<names>.mo.mlir` stem pipeline_to_mo_graph.py
    // writes, so the .mojo dump lands beside the source graph's stages.
    std::string baseFileName;
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      if (!modelOp.getIsSubgraph()) {
        if (!baseFileName.empty())
          baseFileName += "+";
        baseFileName += modelOp.getSymName().str();
      }
    }
    if (baseFileName.empty())
      baseFileName = "graph";

    // Save the emitted Mojo code to a file if an IR dump directory is set
    // (via `MODULAR_DEBUG=ir-output-dir=<dir>` or the legacy `max.temps_dir`).
    if (auto outFile = openIntermediateTextFile(baseFileName, ".mojo")) {
      outFile->os() << mojoText;
      outFile->keep();
    }

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

    // The device-info mapping is only needed through Mojo emission (e.g. to
    // resolve per-device `tile_based_fusion`); strip it now so it doesn't
    // surface in downstream formats (e.g. MEF).
    moduleOp->removeAttr(MO::DeviceInfoMapping::kAttrName);
  }

  // This pass emits Mojo text. [optional] phases exist strictly as compilation
  // time optimizations.
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // [optional]
    // Strip device IDs (not the labels like cpu or gpu, just the numbers) and
    // dynamic shape parameters because those mean nothing when emitting Mojo
    // and thus they should not be an obstacle to deduplication.
    stripDeviceIdsAndDynamicShapes();

    // [optional] Lift inline `index.constant` ops feeding
    // `mgp.buffer.get_cached`'s slot operand out of kernel bodies and into
    // kernel-stub function arguments, so that kernels that previously
    // differed only in those constant slot indices become structurally
    // identical and `MergeIdenticalKernelStubs` (below) can collapse them.
    // See `liftBufferCacheSlots` for the full rationale.
    liftBufferCacheSlots(moduleOp);

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
    // Merge identical kernel stubs using their hashes.
    pm.addPass(MOGG::createMergeIdenticalKernelStubs());

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
    pm.addPass(MOGG::createAddTryExceptBlocks());

    if (pm.run(moduleOp).failed())
      return signalPassFailure();

    emitMojoText();
  }
};

} // namespace

} // namespace M::MOGG
