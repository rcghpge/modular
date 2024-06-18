//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/CUDADialect/CUDADialect.h"
#include "GenericML/CUDATransforms/CUDATransforms.h"
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOQDialect/MOQDialect.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/Addons/MONameInputDimsPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/ConstantFoldPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/Transforms/MOQDialect/MOQTransforms.h"
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GenericML/Support/BlobLifetimeManager.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
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
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "mlir/IR/AsmState.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOQ::MOQDialect, MOSH::MOSHDialect,
                  MIP::MIPDialect, MOGG::MOGGDialect, MGP::MGPDialect,
                  GRT::GRTDialect, CUDA::CUDADialect, RMO::RMODialect>();
  M::registerAllKGENDialects(registry);
  M::registerKGENToLLVMTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<ShapeInferenceOptions>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipelineDefaultOpts);
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
  MO::registerNameInputDimsPass();
  MGP::registerMOToPrimitives();
  MGP::registerLowerMOGGRegionCompiles();
  MGP::registerLowerPrimitives();
  MGP::registerLowerMOGGABI();
  MGP::registerFuseGenericCompiledRegions();
  MGP::registerRemoveRedundantOperands();
  MGP::registerRemoveNonLowerableResults();
  MGP::registerMGPTransformsPasses();
  CUDA::registerCUDATransformsPasses();
  MOGG::registerPasses();

  MOQ::registerFixupMOQOpDTypesPass();
  MOQ::registerLegalizeMOQOpsToMOPass();
  MOQ::registerLowerQDQOperatorsToMOPass();
  MOQ::registerSimplifyQDQPatternsPass();
  MOQ::registerDynamicQuantizeQ4MatmulForModconPass();

  RMO::registerLegalizeRMOOpsPass();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCE();
  mlir::registerViewOpGraphPass();
}

LogicalResult MO::lowerMOToMGP(ModuleOp module,
                               const MoToMgpOptions &moToMgpOptions) {
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);

  MO::buildMoToMgpPipeline(pm, moToMgpOptions);

  return pm.run(module);
}

void M::MO::buildMoToMgpPipeline(mlir::OpPassManager &pm,
                                 const MoToMgpOptions &opts) {
  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

  pm.addPass(MO::createNameInputDimsPass({opts.dimNames}));

  // RMO pipeline:
  ResolveUnknownParametersOptions resolveUnknownParametersOpts;
  resolveUnknownParametersOpts.resolveRMOOnly = true;
  pm.addPass(MO::createResolveUnknownParameters(resolveUnknownParametersOpts));
  pm.addPass(RMO::createLegalizeRMOOps());

  // MO pipeline:
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (opts.lowerQDQOperators) {
    pm.addPass(MOQ::createLowerQDQOperatorsToMO());
    pm.addPass(MOQ::createFixupMOQOpDTypes());
    pm.addPass(MOQ::createSimplifyQDQPatterns());
  }
  pm.addPass(MOQ::createLegalizeMOQOpsToMO());

  if (opts.genVizSpec)
    pm.addPass(MO::createGenVizSpec({opts.vizOutputPath}));
  // TODO(#34452): Add an early return here. When genVizSpec flag is added,
  // there is no reason to continue compilation after the needed file is
  // generated.

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.useShapeInference) {
    ShapeInferenceOptions shapeInferenceOpts;
    shapeInferenceOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;
    shapeInferenceOpts.constantFold = opts.constantFold;
    shapeInferenceOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
    buildShapeInferencePipeline(pm, shapeInferenceOpts, opts);

    // Run symbolic optimizations after shape inference to avoid complicating
    // tracing shape pipeline.
    SymbolicOptimizationOptions symbolicOptimizationOpts;
    symbolicOptimizationOpts.symbolicFfastMath = opts.symbolicFfastMath;
    symbolicOptimizationOpts.symbolicNoUndefinedMath =
        opts.symbolicNoUndefinedMath;
    pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));

    // Add fallback shape functions for any remaining unhandled ops.
    // NOTE: https://github.com/modularml/modular/issues/36956 means we need
    // this to handle mo.arg_nonzero.
    pm.addPass(MO::createAddFallbackShapeFunctions({opts.kernelLibraryPaths}));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }

  std::string_view compiledFrameworkLabel;
  if (opts.compiledFrameworkLabel.hasValue())
    compiledFrameworkLabel = opts.compiledFrameworkLabel;

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, opts));

  MO::AddDevicesPassOptions addDevicesOpts;
  addDevicesOpts.enableExtraDevices = opts.enableExtraDevices;
  addDevicesOpts.enableFakeCudaDevice = opts.enableFakeCudaDevice;
  addDevicesOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  pm.addPass(MO::createAddDevicesPass(addDevicesOpts));
  pm.addPass(MO::createInferLayouts());

  if (opts.usePatterns) {
    MOGG::PatternFusionOptions patternFusionOpts;
    patternFusionOpts.mhaAllowUnsafeBCast =
        opts.patternFusionMHAAllowUnsafeBCasts;
    patternFusionOpts.mhaAllowInternalReshape =
        opts.patternFusionMHAAllowInternalReshape;
    patternFusionOpts.enableMHAPattern = opts.patternFusionMHAEnable;
    pm.addPass(MOGG::createPatternFusion(patternFusionOpts));

    // Layout optimizations in `PatternFusion` might add new shape function
    // invocation for the inserted layout transformations, and in most (if not
    // all) cases we can fold them away.
    SymbolicizeFallbackShapeFunctionsOptions symbolizationOpts{
        opts.kernelLibraryPaths, /*reducedMode=*/true};
    pm.addPass(MO::createSymbolicizeFallbackShapeFunctions(symbolizationOpts));
    pm.addPass(MO::createPropagateShapes());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
  }
  pm.addPass(MO::createAssignDevicesPass());

  // If jiting simplify broadcast patterns to a single op that we can jit.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, opts));

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addPass(MO::createMOToMOGG({opts.kernelLibraryPaths, opts.forceSync}));
  pm.addPass(MOGG::createSimplifyKernelOperands());
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
    // Mark trivial kernels as being synchronous.
    pm.addPass(MOGG::createMarkTrivialKernels());

    // Soft fuse trivial and non-host kernels where possible.
    if (opts.fuseJitKernels) {
      pm.addPass(MOGG::createInlineLambdas());
      MOGG::SoftFuseOptions softFuseOpts;
      softFuseOpts.nonHostKernelLimit = opts.softFusionNonHostKernelLimit;
      pm.addPass(MOGG::createSoftFuse(softFuseOpts));

      MOGG::KernelSequenceFuseOptions sequenceFusionOpts;
      sequenceFusionOpts.nonHostKernelLimit = opts.softFusionNonHostKernelLimit;
      sequenceFusionOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
      pm.addPass(MOGG::createKernelSequenceFuse(sequenceFusionOpts));

      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createCSEPass());
    }
  }

  // Fuse constants last as they don't enable any extra fusions directly and
  // should be fused as late as possible to catch all cases of constants.
  if (opts.fuseJitKernels)
    pm.addPass(MOGG::createFuseConstants());

  // Add cleanup passes to catch e.g used/duplicated constants.
  pm.addPass(MOGG::createStackAllocateTemporaries());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MOGG::createSimplifyKernelOperands());

  MOGG::TranslateMOGGToKGENOptions translationOpts;
  translationOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  translationOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  pm.addPass(MOGG::createTranslateMOGGToKGEN(translationOpts));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  // If we are jitting then we need to compile the jit kernels as well.
  MOGG::JitCompileKernelsOptions jitOpts;
  jitOpts.useSearch = opts.useKgenSearch;
  jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  jitOpts.saveTempPrefix = opts.saveTempPrefix;
  jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
  jitOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
  jitOpts.skipJitCompileMiddle = opts.skipJitCompileMiddle;
  pm.addPass(MOGG::createJitCompileKernels(jitOpts));

  // Create the options for MoToPrimitives.
  MGP::MOToPrimitivesOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;
  pm.addPass(MOGG::createInlineLambdas());
  pm.addPass(MGP::createMOToPrimitives(moToPrimOpts));

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
  if (opts.enableMemoryPlanning) {
    // Canonicalization folds calculation of buffer size into a form the
    // memory planner understands.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createStaticMemoryPlannerPass());
  }
  pm.addPass(MGP::createPredicateAssertDependenciesPass());
  pm.addPass(MGP::createStripBufferAttributesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createAddDeviceContextProfilingPass());
  if (!opts.saveTempPrefix.empty())
    jitOpts.saveTempPrefix = opts.saveTempPrefix + ".region";
  MGP::LowerPrimitivesOptions lowerPrimOpts;
  lowerPrimOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  pm.addPass(MGP::createLowerPrimitives(lowerPrimOpts));
  MGP::FuseGenericCompiledRegionsOptions fuseOpts;
  fuseOpts.enableFusion = opts.enableCompiledRegionsFusion;
  pm.addPass(MGP::createFuseGenericCompiledRegions(fuseOpts));
  // FIXME: Remove the `DisableRemoveRedundantOperands` knob after fixing the
  // runtime error.
  if (!opts.DisableRemoveRedundantOperands)
    pm.addPass(MGP::createRemoveRedundantOperands());
  pm.addPass(MGP::createRemoveNonLowerableResults());

  MGP::LowerMOGGABIOptions abiOpts;
  abiOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  pm.addPass(MGP::createLowerMOGGABI(abiOpts));
  pm.addPass(MOGG::createJitCompileKernels(jitOpts));
  pm.addPass(MGP::createLowerMOGGRegionCompiles());
  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildShapeInferencePipelineDefaultOpts(
    mlir::OpPassManager &pm, const ShapeInferenceOptions &opts) {
  MoToMgpOptions moToMgpOptions;
  moToMgpOptions.kernelLibraryPaths = opts.kernelLibraryPaths;
  moToMgpOptions.compiledFrameworkLabel = opts.compiledFrameworkLabel;
  buildShapeInferencePipeline(pm, opts, moToMgpOptions);
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm,
                                        const ShapeInferenceOptions &opts,
                                        const MoToMgpOptions &moToMgpOptions) {
  pm.addPass(MO::createResolveUnknownParameters());
  AddFallbackShapeFunctionsOptions passOpts;
  passOpts.restricted = true;
  passOpts.kernelLibraryPaths = opts.kernelLibraryPaths;
  SymbolicizeFallbackShapeFunctionsOptions symbolizationOpts{
      opts.kernelLibraryPaths, /*reducedMode=*/true};
  // Adding fallback shape function calls move the parameter declaration from
  // most MO ops (execpt for rebind and #36956) into the shape function call
  // ops, so that downstream passes won't need to worry about them when rewrting
  // the MO ops.
  pm.addPass(MO::createAddFallbackShapeFunctions(passOpts));

  // This canonicalizer pass serves the following purposes:
  // 1. Convert static reshapes like `[D0] -> [1]` into rebind, otherwise
  //    staticization must alsp look through those reshapes for shape inputs.
  // 2. Materialize parameter definition for `mo.rebind` that still declare
  //    output parameters.
  pm.addPass(mlir::createCanonicalizerPass());
  // Staticization should always run first because it might open up new
  // opportunities for shape function symbolicization.
  pm.addPass(MO::createStaticizeShapes());
  pm.addPass(MO::createSymbolicizeFallbackShapeFunctions(symbolizationOpts));
  // Propagation always goes last as it simplifies and propagates all the
  // symbolic expressions we obtained from staticization and symbolicization.
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // TODO(GRA-560): There may be more opportunities in staticization exposed by
  // propagation of shapes which is why it's repeated here.
  pm.addPass(MO::createStaticizeShapes());
  pm.addPass(MO::createPropagateShapes());

  // Re-run the simplification part of shape inference (i.e., excluding
  // `ResolveUnknownParameters` and `AddFallbackShapeFunctions`)

  // We run constant folding here because sometimes not all constant ops
  // have static output shape (and constant folding will conservatively not fold
  // those), e.g., `mo.range : ... -> !mo.tensor<[?], si32>`. Constant folding
  // is needed because Staticization currently only works on a limited set
  // of shape computation encoded in MO.
  if (opts.constantFold) {
    pm.addPass(
        MO::createConstantFoldPass(opts.compiledFrameworkLabel.hasValue()
                                       ? opts.compiledFrameworkLabel.getValue()
                                       : "",
                                   moToMgpOptions));
  }
  // The rest follows the ordering from above, also see comments there for why.
  pm.addPass(MO::createStaticizeShapes());
  pm.addPass(MO::createSymbolicizeFallbackShapeFunctions(symbolizationOpts));
  pm.addPass(MO::createPropagateShapes());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

using BlobMap = DenseMap<const uint8_t *, mlir::AsmResourceBlob *>;

static BlobMap getDenseResources(ModuleOp module) {
  BlobMap result;
  for (Operation &op : module.getOps()) {
    if (auto modelOp = dyn_cast<MGP::ModelOp>(op)) {
      modelOp.walk([&](MGP::BufferConstantOp constOp) {
        auto elementsAttr = constOp.getBValue();
        if (auto resourceAttr =
                dyn_cast<DenseResourceElementsAttr>(elementsAttr)) {
          // Map the underlying data to the ASM Blob for later memory management
          mlir::AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
          if (blob->getDeleter())
            result[blob->getDataAs<uint8_t>().data()] = blob;
        }
      });
    }
  }

  return result;
}

ErrorOr<BufferRef> M::MO::convertMGPToMEF(ModuleOp module,
                                          bool enableMEFMetadata) {
  OwningOpRef<ModuleOp> moduleToRewrite = module.clone();
  // Some MGP-level ops may need to be lightly rewritten before encoding
  // into MEF. For example, the device-generic ops are renamed to capture
  // their device label so the GraphRT executor can dispatch to a
  // device-appropriate primitive.
  BlobMap resources = getDenseResources(*moduleToRewrite);
  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module->getContext(), mlir::OpPassManager::Nesting::Implicit);
  pm.addPass(MGP::createPrepareDevicePrimitivesPass());
  if (failed(pm.run(*moduleToRewrite)))
    return Error("failed to prepare MGP model for conversion to MEF.");
  // Translate to MEF.
  ConcatenationTree result =
      translateMLIRToMEF(*moduleToRewrite, enableMEFMetadata);
  if (result.getSize() == 0)
    return Error("failed to convert model to MEF.");
  // Serialize the ConcatenationTree to the result buffer.
  M::WriteableBufferRef outputBuffer = WriteableBuffer::get(
      0, maxAttributeAlignment().value(), result.getSize());
  result.traverse([&](ArrayRef<uint8_t> data) {
    outputBuffer->write(reinterpret_cast<const char *>(data.data()),
                        data.size());
    if (resources.count(data.data())) {
      if (auto blob = resources[data.data()]; blob->getDeleter()) {
        // Take ownership of the blob and immediately delete blobs that own
        // their data.
        auto ref = M::GML::BlobLifetimeManager(std::move(blob->getDeleter()),
                                               blob->getData(),
                                               blob->getDataAlignment());
        blob->getDeleter() = nullptr;
      }
    }
  });

  return outputBuffer;
}
