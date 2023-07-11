//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "Cache/CachePasses/CachePasses.h"
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/InitAllDialects.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      MO::MODialect, MOSH::MOSHDialect, MIP::MIPDialect, MOGG::MOGGDialect,
      // The dialects needed for KGEN.
      KGEN::KGENDialect, KGEN::POP::POPDialect, DebugInfo::DebugInfoDialect,
      Cache::CacheDialect, HLCF::HLCFDialect, mlir::index::IndexDialect,
      mlir::LLVM::LLVMDialect>();

  // The elaborator requires LLVM lowering to run the generated functions.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}
void M::MO::registerMoToMgpPipeline() {
  mlir::PassPipelineRegistration<MoToMgpOptions>(
      "mo-to-mgp", "Lower MO to MGP.", buildMoToMgpPipeline);
}

void M::MO::registerPasses() {
  MO::registerShapeInferencePipeline();
  MO::registerMoToMgpPipeline();
  MO::registerMOToMOGGPasses();
  MO::registerMOTransformsPasses();
  MGP::registerMOToPrimitives();
  Cache::registerDeflateConstants();
  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCE();
}

void M::MO::buildMoToMgpPipeline(mlir::OpPassManager &pm,
                                 const MoToMgpOptions &opts) {
  // Create the options for MoToPrimitives.
  MGP::MOToPrimitivesOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (opts.useShapeInference) {
    buildShapeInferencePipeline(pm);
    pm.addPass(MO::createSymbolicOptimization());
    pm.addPass(MO::createPropagateShapes());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
  pm.addPass(MO::createDeparameterizeMoOps());

  if (opts.usePatterns) {
    pm.addPass(MOGG::createPatternFusion());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  MO::MOToMOGGOptions moToMoggOpts;
  moToMoggOpts.useJit = opts.useJit;
  // If jiting simplify broadcast patterns to a single op that we can jit.
  if (opts.useJit) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addPass(MO::createMOToMOGG(moToMoggOpts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.useJit) {
    // Fuse connected elementwise kernels.
    if (opts.fuseJitKernels) {
      pm.addPass(MOGG::createFuseElementwise());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());

      // Fuse broadcasts.
      pm.addPass(MOGG::createFuseBroadcasts());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
      pm.addPass(MOGG::createFuseConstants());
    }

    if (opts.performTrivialKernelOpt) {
      // Mark trivial kernels as being syncronous.
      pm.addPass(MOGG::createMarkTrivialKernels());

      // Soft fuse trivial kernels where possible.
      if (opts.fuseJitKernels) {
        pm.addPass(MOGG::createInlineLambdas());
        pm.addPass(MOGG::createSoftFuse());
      }
    }

    // Add explicit chain ready's to the end of fully sync kenrels.
    pm.addPass(MOGG::createAddExplicitChainReady());

    // Add a cleanup pass to catch e.g duplicated constants.
    pm.addPass(mlir::createCSEPass());

    // If we are jitting then we need to compile the jit kernels as well.
    MOGG::JitCompileKernelsOptions jitOpts;
    jitOpts.useSearch = opts.useKgenSearch;
    jitOpts.minimumCpuAlignment = moToPrimOpts.minimumCpuAlignment;
    pm.addPass(MOGG::createJitCompileKernels(jitOpts));
  }

  pm.addPass(MGP::createMOToPrimitives(moToPrimOpts));

  if (opts.cacheBuffersAcrossExecutes)
    pm.addPass(MGP::createReallocBuffersPass());

  if (opts.cacheDir.hasValue() && !opts.cacheDir.empty()) {
    Cache::DeflateConstantsOptions deflateOpts;
    deflateOpts.cacheDir = opts.cacheDir;
    pm.addPass(Cache::createDeflateConstants(deflateOpts));

    MGP::LookupCachedConstantsPassOptions lookupOpts;
    lookupOpts.cacheDir = opts.cacheDir;
    pm.addPass(MGP::createLookupCachedConstantsPass(lookupOpts));
  }

  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createRemoveRedundantTensorExtractsPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(MGP::createConcatInPlacePass());
  pm.addPass(MGP::createDeferAllocsPass());
  pm.addPass(MGP::createExecInvariantCodeMotionPass());
  pm.addPass(MGP::createStripBufferAttributesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm) {
  pm.addPass(MO::createResolveUnknownParameters());
  pm.addPass(MO::createShapeInferenceMaterialization());
  // Convert most scalar <--> tensor reshapes into rebind, which will be merged
  // during staticization. Otherwise staticization must look through those
  // reshapes for shape inputs.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MO::createStaticizeShapes());
  pm.addPass(MO::createInlineShapeFunctions());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
