//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "Cache/CachePasses/CachePasses.h"
#include "GenericML/CUDATransforms/CUDATransforms.h"
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOQDialect/MOQDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/Addons/MONameInputDimsPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/ConstantFoldPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/Transforms/MOQDialect/MOQTransforms.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GenericML/Support/ForeignBufferInterface.h"
#include "GenericML/Support/ResourceBufferInterface.h"
#include "GraphRT/MIPDialect/MIPDialect.h"
#include "GraphRT/MLIRToMEF.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/ADT/ConcatenationTree.h"
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
  mlir::PassPipelineRegistration<ShapeInferenceOptions>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

static void buildMoToMgpPipelineDefaultRuntime(mlir::OpPassManager &pm,
                                               const MoToMgpOptions &opts) {
  // Requires a runtime to have already been established.
  buildMoToMgpPipeline(pm, opts, LLCL::Runtime::getCurrentRuntime());
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
  MO::registerNameInputDimsPass();
  MGP::registerMOToPrimitives();
  Cache::registerDeflateConstants();
  MGP::registerMGPTransformsPasses();
  CUDA::registerCUDATransformsPasses();
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
    pm.addPass(
        MO::createConstantFoldPass(opts.compiledFrameworkLabel.hasValue()
                                       ? opts.compiledFrameworkLabel.getValue()
                                       : ""));
  ShapeInferenceOptions shapeInferenceOpts;
  shapeInferenceOpts.isExperimentalMode = opts.useExperimentalShapeInference;
  shapeInferenceOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  buildShapeInferencePipeline(pm, shapeInferenceOpts);
  pm.addPass(MO::createSymbolicOptimization());
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(MO::createAddFallbackShapeFunctions({opts.kernelLibraryPaths}));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MO::createSymbolicOptimization());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

LogicalResult MO::lowerMOToMGP(ModuleOp module,
                               const MoToMgpOptions &moToMgpOptions,
                               Runtime &runtime) {
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);

  MO::buildMoToMgpPipeline(pm, moToMgpOptions, runtime);

  return pm.run(module);
}

void M::MO::buildMoToMgpPipeline(mlir::OpPassManager &pm,
                                 const MoToMgpOptions &opts, Runtime &runtime) {
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

  if (opts.genVizSpec)
    pm.addPass(MO::createGenVizSpec({opts.vizOutputDir}));
  // TODO: Add an early return here. When genVizSpec flag is added, there is
  // no reason to continue compilation after the needed file is generated.

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  maybeApplyShapeInferencePipeline(pm, opts);

  std::string_view compiledFrameworkLabel;
  if (opts.compiledFrameworkLabel.hasValue())
    compiledFrameworkLabel = opts.compiledFrameworkLabel;

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel));

  MO::AddDevicesPassOptions addDevicesOpts;
  addDevicesOpts.enableExtraDevices = opts.enableExtraDevices;
  addDevicesOpts.enableFakeCudaDevice = opts.enableFakeCudaDevice;
  addDevicesOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  pm.addPass(MO::createAddDevicesPass(addDevicesOpts, runtime));
  pm.addPass(MO::createInferLayouts());

  if (opts.usePatterns) {
    pm.addPass(MOGG::createPatternFusion());
    // Layout optimizations in `PatternFusion` might add new shape function
    // invocation for the inserted layout transformations, and in most (if not
    // all) cases we can fold them away.
    if (opts.useExperimentalShapeInference) {
      pm.addPass(MO::createSymbolicizeFallbackShapeFunctions(
          {opts.kernelLibraryPaths}));
      pm.addPass(MO::createPropagateShapes());
    }
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
  pm.addPass(MO::createAssignDevicesPass());

  // If jiting simplify broadcast patterns to a single op that we can jit.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel));

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addPass(MO::createMOToMOGG({opts.kernelLibraryPaths}));
  pm.addPass(MOGG::createSimplifyKernelOperands());
  pm.addPass(MOGG::createSimplifyKGENParameters());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // Fuse connected elementwise kernels.
  if (opts.fuseJitKernels) {
    pm.addPass(MOGG::createFuseElementwise());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Fuse broadcasts.
    pm.addPass(MOGG::createFuseBroadcasts());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Fuse into big ops. I.E epilogue fusion into matmul.
    pm.addPass(MOGG::createFuseInOutLambdas());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(MOGG::createSimplifyKernelOperands());
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

  // Fuse constants last as they don't enable any extra fusions directly and
  // should be fused as late as possible to catch all cases of constants.
  if (opts.fuseJitKernels)
    pm.addPass(MOGG::createFuseConstants());

  // Add cleanup passes to catch e.g used/duplicated constants.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MOGG::createSimplifyKernelOperands());
  pm.addPass(MOGG::createInlineKGENParameters());

  // If we are jitting then we need to compile the jit kernels as well.
  MOGG::JitCompileKernelsOptions jitOpts;
  jitOpts.useSearch = opts.useKgenSearch;
  jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  jitOpts.saveTempPrefix = opts.saveTempPrefix;
  jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
  jitOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
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
    pm.addPass(Cache::createDeflateConstantsPass(deflateOpts, runtime));

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
  if (opts.minPinnedHostBufferSize > 0) {
    CUDA::CUDAPinHostBuffersPassOptions pinOpts;
    pinOpts.minPinnedHostBufferSize = opts.minPinnedHostBufferSize;
    pm.addPass(CUDA::createCUDAPinHostBuffersPass(pinOpts));
  }
  if (opts.enableMemoryPlanning)
    pm.addPass(MGP::createStaticMemoryPlannerPass());
  pm.addPass(MGP::createPredicateAssertDependenciesPass());
  pm.addPass(MGP::createStripBufferAttributesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm,
                                        const ShapeInferenceOptions &opts) {
  pm.addPass(MO::createResolveUnknownParameters());
  if (opts.isExperimentalMode) {
    // these will handle ops with simple shape functions like
    // mo.matmul, mo.concat, mo.gather, mo.range, etc.
    AddFallbackShapeFunctionsOptions passOpts;
    passOpts.restricted = true;
    passOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
    pm.addPass(MO::createAddFallbackShapeFunctions(passOpts));
    pm.addPass(
        MO::createSymbolicizeFallbackShapeFunctions({opts.kernelLibraryPaths}));
  }
  // this handles this rest
  pm.addPass(MO::createShapeInferenceMaterialization());

  // Convert most scalar <--> tensor reshapes into rebind, which will be merged
  // during staticization. Otherwise staticization must look through those
  // reshapes for shape inputs.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MO::createStaticizeShapes());

  // Staticization might open up new opportunities for symbolicizing fallback
  // shape functions. This is simialr to why we run symbolic shape function
  // materialization during staticization.
  if (opts.isExperimentalMode)
    pm.addPass(
        MO::createSymbolicizeFallbackShapeFunctions({opts.kernelLibraryPaths}));
  pm.addPass(MO::createInlineShapeFunctions());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

ErrorOr<BufferRef> M::MO::convertMGPToMEF(ModuleOp module,
                                          bool enableMEFMetadata) {
  // Some MGP-level ops may need to be lightly rewritten before encoding
  // into MEF. For example, the device-generic ops are renamed to capture
  // their device label so the GraphRT executor can dispatch to a
  // device-appropriate primitive.
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);
  pm.addPass(MGP::createPrepareDevicePrimitivesPass());
  if (failed(pm.run(module)))
    return Error("failed to prepare MGP model for conversion to MEF.");

  // Translate to MEF.
  ConcatenationTree result = translateMLIRToMEF(module, enableMEFMetadata);
  if (result.getSize() == 0)
    return Error("failed to convert model to MEF.");

  // Serialize the ConcatenationTree to the result buffer.
  M::WriteableBufferRef outputBuffer = WriteableBuffer::get(
      0, maxAttributeAlignment().value(), result.getSize());
  result.traverse([&](ArrayRef<uint8_t> data) {
    outputBuffer->write(reinterpret_cast<const char *>(data.data()),
                        data.size());
  });

  return outputBuffer;
}
