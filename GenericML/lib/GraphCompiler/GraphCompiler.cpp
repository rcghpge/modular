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
#include "GenericML/GraphCompiler/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/MGPDialect/MGPTransforms.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/HLCFDialect/HLCFDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOGG::MOGGDialect,
                  // The dialects needed for KGEN.
                  KGEN::KGENDialect, DebugInfo::DebugInfoDialect,
                  Cache::CacheDialect, HLCF::HLCFDialect,
                  mlir::index::IndexDialect, mlir::LLVM::LLVMDialect>();

  // The elaborator requires LLVM lowering to run the generated functions.
  mlir::registerLLVMDialectTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

void M::MO::registerPasses() {
  MO::registerShapeInferencePipeline();
  MO::registerMOToMOGG();
  MO::registerMOTransformsPasses();
  MGP::registerMOToPrimitives();
  Cache::registerDeflateConstants();
  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
}

void M::MO::buildMoToMGPPipeline(mlir::PassManager &pm,
                                 const PassOptions &opts) {
  pm.addPass(mlir::createCSEPass());

  MO::MOToMOGGOptions moToMoggOpts;
  moToMoggOpts.useJit = opts.useJit;
  pm.addPass(MO::createMOToMOGG(moToMoggOpts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.useJit) {
    // Fuse connected elementwise kernels.
    if (opts.fuseJitKernels) {
      pm.addPass(MOGG::createFuseElementwise());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }

    // If we are jitting then we need to compile the jit kernels as well.
    MOGG::JitCompileKernelsOptions jitOpts;
    jitOpts.useSearch = opts.useKgenSearch;
    pm.addPass(MOGG::createJitCompileKernels(jitOpts));
  }

  MGP::MOToPrimitivesOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;
  pm.addPass(MGP::createMOToPrimitives(moToPrimOpts));

  if (!opts.cacheDir.empty()) {
    Cache::DeflateConstantsOptions deflateOpts;
    deflateOpts.cacheDir = opts.cacheDir;
    pm.addPass(Cache::createDeflateConstants(deflateOpts));

    MGP::LookupCachedConstantsPassOptions lookupOpts;
    lookupOpts.cacheDir = opts.cacheDir;
    pm.addPass(MGP::createLookupCachedConstantsPass(lookupOpts));
  }

  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createPeepholeTensorExtract());
  pm.addPass(MGP::createStripBufferAttributesPass());
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm) {
  pm.addPass(MO::createResolveUnknownParameters());
  pm.addPass(MO::createStaticizeShapes());
  pm.addPass(MO::createShapeInferenceMaterialization());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MO::createInlineShapeFunctions());
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}
