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
#include "GenericML/GraphCompiler/MOQDialect/MOQDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/Transforms/MOQDialect/MOQTransforms.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "Cache/CacheDialect/CacheDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOQ::MOQDialect, MOSH::MOSHDialect,
                  MIP::MIPDialect, MOGG::MOGGDialect>();
  M::registerAllKGENDialects(registry);
  M::registerKGENToLLVMTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

static Runtime *&getMoToMgpRuntime() {
  static Runtime *theRuntime = nullptr;
  return theRuntime;
}

void M::MO::registerMoToMgpRuntimeForTesting(Runtime &runtime) {
  getMoToMgpRuntime() = &runtime;
}

static void buildMoToMgpPipelineDefaultRuntime(mlir::OpPassManager &pm,
                                               const MoToMgpOptions &opts) {
  buildMoToMgpPipeline(pm, opts, getMoToMgpRuntime());
}

void M::MO::registerMoToMgpPipeline() {
  mlir::PassPipelineRegistration<MoToMgpOptions>(
      "mo-to-mgp", "Lower MO to MGP.", buildMoToMgpPipelineDefaultRuntime);
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

  MOQ::registerFixupMOQOpDTypesPass();
  MOQ::registerLegalizeMOQOpsToMOPass();
  MOQ::registerLowerQDQOperatorsToMOPass();
  MOQ::registerSimplifyQDQPatternsPass();
  MOQ::registerDynamicQuantizeQ4MatmulForModconPass();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCE();
  mlir::registerViewOpGraphPass();
}

static inline void
maybeApplyShapeInferencePipeline(mlir::OpPassManager &pm,
                                 const MoToMgpOptions &opts) {
  // Adds passes to perform shape inference and resulting optimizations if
  // relevant flags are on.
  if (!opts.useShapeInference)
    return;
  if (opts.constantFold)
    pm.addPass(MO::createConstantFold());
  buildShapeInferencePipeline(pm);
  pm.addPass(MO::createSymbolicOptimization());
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(MO::createAddFallbackShapeFunctions({opts.kernelLibraryPaths}));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MO::createSymbolicOptimization());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildMoToMgpPipeline(mlir::OpPassManager &pm,
                                 const MoToMgpOptions &opts,
                                 Runtime *optRuntime) {
  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

  pm.addPass(MO::createNameInputDimsPass({opts.dimNames}));

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (opts.lowerQDQOperators) {
    pm.addPass(MOQ::createLowerQDQOperatorsToMO());
    pm.addPass(MOQ::createFixupMOQOpDTypes());
    pm.addPass(MOQ::createSimplifyQDQPatterns());
  }
  pm.addPass(MOQ::createLegalizeMOQOpsToMO());

// TODO: This will be replaced with opts.genVizSpec flag once some issues in the
// option piplines are resolved.
#ifndef MODULAR_PRODUCTION
  if (std::getenv("GENERATE_VISUALIZATION") != nullptr)
    pm.addPass(MO::createGenVizSpec({opts.vizOutputDir}));
    // TODO: Add an early return here. When genVizSpec flag is added, there is
    // no reason to continue compilation after the needed file is generated.
#endif // !MODULAR_PRODUCTION

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  maybeApplyShapeInferencePipeline(pm, opts);
  if (opts.constantFold)
    pm.addPass(MO::createConstantFold());

  MO::AddDevicesPassOptions addDevicesOpts;
  addDevicesOpts.enableExtraDevices = opts.enableExtraDevices;
  addDevicesOpts.enableFakeCudaDevice = opts.enableFakeCudaDevice;
  addDevicesOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  pm.addPass(MO::createAddDevicesPass(addDevicesOpts, optRuntime));

  pm.addPass(MO::createHackyPatternFusionForModconPass());
  pm.addPass(MO::createAssignDevicesPass());
  pm.addPass(MO::createInferLayouts());

  if (opts.usePatterns) {
    pm.addPass(MOGG::createPatternFusion());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  // If jiting simplify broadcast patterns to a single op that we can jit.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFold());

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addPass(MO::createMOToMOGG({opts.kernelLibraryPaths}));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Fuse connected elementwise kernels.
  if (opts.fuseJitKernels) {
    pm.addPass(MOGG::createFuseElementwise());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Fuse broadcasts.
    pm.addPass(MOGG::createFuseBroadcasts());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // Fuse into big ops. I.E epilogue fusion into matmul.
    pm.addPass(MOGG::createFuseInOutLambdas());
    pm.addPass(mlir::createCSEPass());
  }

  if (opts.performTrivialKernelOpt) {
    // Mark trivial kernels as being syncronous.
    pm.addPass(MOGG::createMarkTrivialKernels());

    // Soft fuse trivial and non-host kernels where possible.
    if (opts.fuseJitKernels) {
      pm.addPass(MOGG::createInlineLambdas());
      MOGG::SoftFuseOptions softFuseOpts;
      softFuseOpts.nonHostKernelLimit = opts.softFusionNonHostKernelLimit;
      pm.addPass(MOGG::createSoftFuse(softFuseOpts));
    }
  }

  // Add explicit chain ready's to the end of fully sync kenrels.
  pm.addPass(MOGG::createAddExplicitChainReady());

  // Fuse constants last as they don't enable any extra fusions directly and
  // should be fused as late as possible to catch all cases of constants.
  if (opts.fuseJitKernels)
    pm.addPass(MOGG::createFuseConstants());

  // Add cleanup passes to catch e.g used/duplicated constants.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // If we are jitting then we need to compile the jit kernels as well.
  MOGG::JitCompileKernelsOptions jitOpts;
  jitOpts.useSearch = opts.useKgenSearch;
  jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  jitOpts.saveTempPrefix = opts.saveTempPrefix;
  jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
  jitOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  pm.addPass(MOGG::createJitCompileKernels(jitOpts));

  // Create the options for MoToPrimitives.
  MGP::MOToPrimitivesOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;
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
  pm.addPass(MGP::createDeferTensorCreateExecPass());
  pm.addPass(MGP::createDeferAllocsPass());
  pm.addPass(MGP::createExecInvariantCodeMotionPass());
  pm.addPass(MGP::createVerifyModelArgumentsPass());
  // TODO(#26193): Disabled due to combinatorial explosion.
  // pm.addPass(MGP::createPredicateAssertDependenciesPass());
  if (opts.enableMemoryPlanning)
    pm.addPass(MGP::createStaticMemoryPlannerPass());
  pm.addPass(MGP::createStripBufferAttributesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
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
