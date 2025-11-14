//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "AsyncRT/Runtime/Runtime.h"
#include "GenericML/GraphCompiler/Conversion/MOGGToKGEN.h"
#include "GenericML/GraphCompiler/Conversion/MOToMOGG.h"
#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/MODialect/MODialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/Analysis/MojoLibraryAnalysis.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GenericML/GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GenericML/GraphCompiler/Passes.h"
#include "GenericML/GraphCompiler/RMODialect/RMODialect.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/Addons/MONameInputDimsPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/ConstantFoldPass.h"
#include "GenericML/GraphCompiler/Transforms/MODialect/MOTransforms.h"
#include "GenericML/GraphCompiler/Transforms/RMODialect/RMOTransforms.h"
#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPTransforms/MGPTransforms.h"
#include "GenericML/Support/BlobLifetimeManager.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MLIRToMEF.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Compiler/PassTimingUtils.h"

#include "GenericML/MGPDialect/MGPDialect.h"
#include "GraphRT/GRTDialect/GRTDialect.h"
#include "GraphRT/MEFDialect/MEFDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOSH::MOSHDialect, MOGG::MOGGDialect,
                  MGP::MGPDialect, GRT::GRTDialect, RMO::RMODialect>();
  M::registerAllKGENDialects(registry);
  M::registerKGENToLLVMTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

void M::MO::registerMoToMgpPipeline() {
  mlir::PassPipelineRegistration<MoToMgpOptions>(
      "mo-to-mgp", "Lower MO to MGP.",
      [](mlir::OpPassManager &pm, const MoToMgpOptions &opts) {
        buildMoToMgpPipeline(pm, opts);
      });
}

void M::MO::registerPasses() {
  MO::registerShapeInferencePipeline();
  MO::registerMoToMgpPipeline();
  MO::registerMOToMOGGPasses();
  MO::registerMOTransformsPasses();
  MO::registerNameInputDimsPass();
  MGP::registerCoalesceAsyncRefsPass();
  MGP::registerMOToPrimitives();
  MGP::registerLowerAsyncRefs();
  MGP::registerEmitMojo();
  MGP::registerDumpMojo();
  MGP::registerLowerMOGGRegionCompiles();
  MGP::registerLowerPrimitives();
  MGP::registerLowerMOGGABI();
  MGP::registerFuseGenericCompiledRegions();
  MGP::registerPostABIFuseGenericRegions();
  MGP::registerRemoveRedundantOperands();
  MGP::registerLegalizeAsyncRefDeallocations();
  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();
  MOGG::registerMOGGToKGENPasses();

  RMO::registerLegalizeRMOOpsPass();

  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCE();
  mlir::registerViewOpGraphPass();

  KGEN::registerVerifyParametersPass();
}

LogicalResult
MO::lowerMOToMGP(ModuleOp module, const MoToMgpOptions &moToMgpOptions,
                 std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {

  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);

  if (!moToMgpOptions.mlirTimingOutDir.empty()) {
    const char *pipelineName = "MOToMGP Pipeline";
    auto enabledTiming = M::configureMLIRPassTimingJSONOutput(
        pm, moToMgpOptions.mlirTimingOutDir.getValue(), pipelineName);
    if (enabledTiming.isError()) {
      llvm::errs() << enabledTiming.getError() << "\n";
      return failure();
    }
  }

  MO::buildMoToMgpPipeline(pm, moToMgpOptions, std::move(libraryAnalysis));
  return pm.run(module);
}

void M::MO::buildMoToMgpPipeline(
    mlir::OpPassManager &pm, const MoToMgpOptions &opts,
    std::shared_ptr<MOGG::MojoLibraryAnalysis> libraryAnalysis) {
  pm.addPass(KGEN::createVerifyParameters());

  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

  pm.addPass(MO::createNameInputDimsPass(
      {llvm::to_vector_of<std::string>(opts.dimNames)}));

  // RMO pipeline:
  pm.addPass(RMO::createLegalizeRMOOps());

  // MO pipeline:
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createCanonicalizeChainSequences());

  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  // Label for constant fold.
  std::string_view compiledFrameworkLabel;
  if (opts.compiledFrameworkLabel.hasValue())
    compiledFrameworkLabel = opts.compiledFrameworkLabel;

  // Run shape inference and constant fold.
  if (opts.useShapeInference)
    buildShapeInferencePipeline(pm);

  // TODO(GEX-886): Remove dependency of constant folding for symbolic
  // optimization
  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, opts));

  pm.addPass(MO::createHoistConstantSubgraphsIntoParent());
  pm.addPass(mlir::createCSEPass());

  // Run symbolic operator set domain-specific optimizations
  SymbolicOptimizationOptions symbolicOptimizationOpts;
  symbolicOptimizationOpts.symbolicFfastMath = opts.symbolicFfastMath;
  symbolicOptimizationOpts.symbolicNoUndefinedMath =
      opts.symbolicNoUndefinedMath;

  // Add fallback shape functions for any remaining unhandled ops.
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  pm.addPass(MO::createMergeDuplicateShapeMaterializations());
  pm.addPass(MO::createAddFallbackShapeFunctions({kernelLibraryPaths},
                                                 libraryAnalysis));
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, opts));

  pm.addPass(MO::createInferLayouts());
  if (opts.usePatterns) {
    MOGG::PatternFusionOptions patternFusionOpts;
    patternFusionOpts.mhaAllowUnsafeBCast =
        opts.patternFusionMHAAllowUnsafeBCasts;
    patternFusionOpts.mhaAllowInternalReshape =
        opts.patternFusionMHAAllowInternalReshape;
    patternFusionOpts.enableMHAPattern = opts.patternFusionMHAEnable;
    patternFusionOpts.targetDevice = opts.defaultDeviceLabel;
    patternFusionOpts.targetDeviceId = opts.defaultDeviceId;
    pm.addPass(MOGG::createPatternFusion(patternFusionOpts));

    InvokeShapePackingFunctionsOptions shapeFuncOptions;
    shapeFuncOptions.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;

    pm.addPass(MO::createInvokeShapePackingFunctions(shapeFuncOptions));
    pm.addPass(MO::createPropagateShapes());
    pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
    pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  }

  MO::ValidateDevicesPassOptions validateDevicesPassOpt;
  validateDevicesPassOpt.defaultDeviceTargetTriple =
      opts.defaultDeviceTargetTriple;
  validateDevicesPassOpt.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  pm.addPass(MO::createValidateDevicesPass(validateDevicesPassOpt));

  pm.addPass(MO::createDecomposeViewOps());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, opts));

  if (opts.fuseJitKernels) {
    // Only run this when kernel fusion is enabled as otherwise we might
    // pessimize model execution by introducing extraneous copies.
    pm.addPass(MO::createElideRedundantMutableOps());
  }

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(MO::createMOToMOGG({kernelLibraryPaths, opts.forceSync},
                                libraryAnalysis));
  pm.addPass(MOGG::createSimplifyKernelOperands());

  // Fuse connected elementwise kernels.
  if (opts.fuseJitKernels) {
    pm.addPass(MOGG::createFuseAffineViews());
    MOGG::FuseElementwiseOptions fuseElementwiseOptions;
    // To dump dot graph, uncomment below:
    // fuseElementwiseOptions.dumpDotGraph = true;
    pm.addPass(MOGG::createFuseElementwise(fuseElementwiseOptions));
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Fuse views.
    pm.addPass(MOGG::createFuseViews());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Fuse into big ops. I.E epilogue fusion into matmul.
    pm.addPass(MOGG::createFuseProloguesEpilogues());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Fuse constants last as they don't enable any extra fusions directly and
    // should be fused as late as possible to catch all cases of constants.
    pm.addPass(MOGG::createFuseConstants());
  }

  if (opts.fuseMutableLoadStore)
    pm.addPass(MOGG::createFuseMutableLoads());

  // Convert output lambdas to compute output lambdas when possible.
  pm.addPass(MOGG::createMOGGToKGEN(
      {kernelLibraryPaths, opts.forceSync, opts.enableComputeLambdas},
      libraryAnalysis));
  pm.addPass(MOGG::createSimplifyKernelOperands());

  // Add cleanup passes to catch e.g used/duplicated constants.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MOGG::createSimplifyKernelOperands());

  // Create the options for MoToPrimitives.
  MGP::MOToPrimitivesOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;
  pm.addPass(MGP::createMOToPrimitives(moToPrimOpts));

  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(MGP::createConcatInPlacePass());
  // Note: LowerAlignmentsPass must be before Dynamic MemoryPlanning or it can
  // erroneously pick up slices introduced by memory planning
  pm.addPass(MGP::createLowerAlignmentsPass());
  pm.addPass(MGP::createExecInvariantCodeMotionPass());

  // Run simple buffer reuse pass after moving code to `init()`
  // As buffer reuse can prevent hoisting of some code up.
  pm.addPass(MGP::createSimpleBufferReusePass());

  pm.addPass(MGP::createDynamicMemoryPlanningPass());
  pm.addPass(MGP::createDeferAllocsPass());
  pm.addPass(MGP::createVerifyModelArgumentsPass());
  pm.addPass(MGP::createPredicateAssertDependenciesPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  MGP::HoistHarnessOptions harnessOpts;
  harnessOpts.kernelLibraryPaths = kernelLibraryPaths;
  harnessOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  pm.addPass(MGP::createHoistHarness(harnessOpts, libraryAnalysis));

  // Before any sort of fusion, use location information to create better
  // runtime assert messages.
  pm.addPass(MGP::createAddStackTracesToAssertsPass());

  pm.addPass(MGP::createGlobalizeSlotIndicesPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  MGP::LowerPrimitivesOptions lowerPrimOpts;
  lowerPrimOpts.kernelLibraryPaths = kernelLibraryPaths;
  pm.addPass(MGP::createLowerPrimitives(lowerPrimOpts, libraryAnalysis));
  pm.addPass(MGP::createFuseGenericCompiledRegions());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createRemoveRedundantOperands());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(MGP::createSimplifyExtendLifetimesPass());

  pm.addPass(MGP::createLowerMOGGABI());

  if (opts.enablePostABIRegionFusion && opts.fuseJitKernels) {
    pm.addPass(MGP::createPostABIFuseGenericRegions());
    pm.addPass(MGP::createRemoveRedundantOperands());
    // NOTE: We might be able to remove the CSE pass after modeling async value
    // correctly.
    pm.addPass(mlir::createCSEPass());
    // TODO: Run cleanup per MOGG::GenericCompiledRegionOp for pass
    // multi-threading. For now, we simply use a module pass, as the generic
    // region might be enclosed inside a grt.if/while/... operations, making the
    // pass nest harder to find. We should re-exam all passes, many of them can
    // be triggered more locally.
    pm.addPass(MGP::createLegalizeAsyncRefDeallocations());
  }
  // Lower async pack/unpack intrinsic.
  MGP::LowerAsyncRefsOptions lowerAsyncopt;
  lowerAsyncopt.kernelLibraryPaths = kernelLibraryPaths;
  pm.addPass(MGP::createLowerAsyncRefs(lowerAsyncopt, libraryAnalysis));
  pm.addPass(MGP::createCoalesceAsyncRefs());

  MGP::EmitMojoOptions emitMojoOptions;
  emitMojoOptions.kernelLibraryPaths = kernelLibraryPaths;
  emitMojoOptions.dumpMojoText = opts.dumpMojoText;
  emitMojoOptions.saveTempPrefix = opts.saveTempPrefix;
  emitMojoOptions.enableOpTraceComments = opts.enableOpTraceComments;
  emitMojoOptions.enableComputeLambdas = opts.enableComputeLambdas;
  emitMojoOptions.moggDebugDumpFusedTensorSpecs =
      opts.moggDebugDumpFusedTensorSpecs;
  pm.addPass(MGP::createEmitMojo(emitMojoOptions));

  // If we are jitting then we need to compile the jit kernels as well.
  MOGG::JitCompileKernelsOptions jitOpts;
  jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  jitOpts.saveTempPrefix = opts.saveTempPrefix;
  jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
  jitOpts.kernelLibraryPaths = kernelLibraryPaths;
  jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
  jitOpts.debugKGENStubs = opts.debugKGENStubs;

  pm.addPass(MOGG::createJitCompileKernels(jitOpts, libraryAnalysis));
  pm.addPass(MGP::createLowerMOGGRegionCompiles());
  pm.addPass(MGP::createInlineMGPCallsPass());
  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void M::MO::buildShapeInferencePipeline(mlir::OpPassManager &pm) {
  // This canonicalizer pass serves the following purposes:
  // 1. Convert static reshapes like `[D0] -> [1]` into rebind, otherwise
  //    staticization must alsp look through those reshapes for shape inputs.
  // 2. Materialize parameter definition for `mo.rebind` that still declare
  //    output parameters.
  pm.addNestedPass<MO::GraphOp>(mlir::createCanonicalizerPass());

  // Propagation does 3 things:
  // 1. Run calculations of certain constant ops in parameter domain
  // 2. Materializes shape functions using parameter domain values
  // 3. Simplifies parameter binding using substitution and constraint solving.
  pm.addNestedPass<MO::GraphOp>(MO::createPropagateShapes());
  pm.addNestedPass<MO::GraphOp>(mlir::createCSEPass());
  pm.addNestedPass<MO::GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(KGEN::createVerifyParameters());
}

using BlobMap = DenseMap<const uint8_t *, mlir::AsmResourceBlob *>;

static BlobMap getDenseResources(ModuleOp module) {
  BlobMap result;
  for (Operation &op : module.getOps()) {
    if (auto modelOp = dyn_cast<MGP::ModelOp>(op)) {
      modelOp.walk([&](MGP::BufferConstantOp constOp) {
        auto elementsAttr = constOp.getAValue();
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
  FormatScopedDiagnosticHandler diagnostics(moduleToRewrite->getContext());
  if (failed(pm.run(*moduleToRewrite)))
    return Error(Twine("Failed to prepare MGP model for conversion to MEF. ") +
                 diagnostics.formatMessage());
  // Translate to MEF.
  ConcatenationTree result =
      translateMLIRToMEF(*moduleToRewrite, enableMEFMetadata);
  if (result.getSize() == 0) {
    return Error(Twine("Failed to convert model to MEF. ") +
                 diagnostics.formatMessage());
  }
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
