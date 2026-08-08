//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "GraphCompiler/Conversion/MAPToMOGG/MAPToMOGG.h"
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/Conversion/MOToMOGG/MOToMOGG.h"
#include "GraphCompiler/Conversion/PackageVariadicsAndScalars/PackageVariadicsAndScalars.h"
#include "GraphCompiler/Driver/Common/Utils.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Transforms/MAPTransforms.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/Transforms/ConstantFoldPass.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/Passes.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h"
#include "GraphCompiler/Support/BlobLifetimeManager.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/Diagnostics/FormatScopedDiagnosticHandler.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

// All the KGEN dependent passes
#include "KGEN/ToolCommon/InitAllDialects.h"
#include "Support/Compiler/PassTimingUtils.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace MO;

void M::MO::loadDialects(mlir::DialectRegistry &registry) {
  registry.insert<MO::MODialect, MOSH::MOSHDialect, MOGG::MOGGDialect,
                  MGP::MGPDialect, GRT::GRTDialect, RMO::RMODialect,
                  MAP::MAPDialect>();
  M::registerAllKGENDialects(registry);
  M::registerKGENToLLVMTranslation(registry);
}

void M::MO::registerShapeInferencePipeline() {
  mlir::PassPipelineRegistration<>(
      "mo-shape-inference", "Run the end-to-end shape inference pipeline.",
      buildShapeInferencePipeline);
}

void M::MO::buildJitPipeline(mlir::OpPassManager &pm) {
  pm.addPass(MOGG::createLowerMOGGRegionCompiles());
  pm.addPass(MGP::createInlineMGPCallsPass());
  pm.addPass(MGP::createFixupOverhangsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(MGP::createOutlineModelsToCallables());
}

namespace {
static void gcPipelineImpl(mlir::OpPassManager &pm,
                           const GCPipelineOptions &opts, bool skipJit,
                           bool dumpIntermediates = false);
} // namespace

void M::MO::registerGCPipeline() {
  mlir::PassPipelineRegistration<GCPipelineOptions>(
      "gc-pipeline", "Full graph-compiler pipeline (MO → MGP → JIT).",
      [](mlir::OpPassManager &pm, const GCPipelineOptions &opts) {
        // Enable dumps here so buildGCPipeline stays dump-free for cache keys.
        gcPipelineImpl(pm, opts, /*skipJit=*/false,
                       /*dumpIntermediates=*/!getMaxIRDumpDir().empty());
      });
}

void M::MO::registerGCPipelineThroughEmitMojo() {
  mlir::PassPipelineRegistration<GCPipelineOptions>(
      "gc-pipeline-pre-jit",
      "Graph compiler through emit-mojo (MO → MGP → emit-mojo; no JIT).",
      [](mlir::OpPassManager &pm, const GCPipelineOptions &opts) {
        gcPipelineImpl(pm, opts, /*skipJit=*/true,
                       /*dumpIntermediates=*/!getMaxIRDumpDir().empty());
      });
}

void M::MO::registerMoPipeline() {
  mlir::PassPipelineRegistration<MoPipelineOptions>(
      "mo-pipeline",
      "Passes from graph input up to (but not including) mo-to-mogg.",
      [](mlir::OpPassManager &pm, const MoPipelineOptions &opts) {
        buildMoPipeline(pm, opts);
      });
}

void M::MO::registerMapPipeline() {
  mlir::PassPipelineRegistration<MapPipelineOptions>(
      "map-pipeline",
      "Passes between mo-to-map and mogg-to-mgp (both excluded).",
      [](mlir::OpPassManager &pm, const MapPipelineOptions &opts) {
        buildMapPipeline(pm, opts);
      });
}

void M::MO::registerMoggPipeline() {
  mlir::PassPipelineRegistration<MoggPipelineOptions>(
      "mogg-pipeline",
      "Passes between mo-to-mogg and mogg-to-mgp (both excluded).",
      [](mlir::OpPassManager &pm, const MoggPipelineOptions &opts) {
        buildMoggPipeline(pm, opts);
      });
}

void M::MO::registerMgpPipeline() {
  mlir::PassPipelineRegistration<MgpPipelineOptions>(
      "mgp-pipeline", "Passes between mogg-to-mgp and jit (both excluded).",
      [](mlir::OpPassManager &pm, const MgpPipelineOptions &opts) {
        buildMgpPipeline(pm, opts);
      });
}

void M::MO::registerJitPipeline() {
  mlir::PassPipelineRegistration<>(
      "jit-pipeline", "Passes that run after the jit pass.",
      [](mlir::OpPassManager &pm) { buildJitPipeline(pm); });
}

void M::MO::registerPasses() {
  MO::registerShapeInferencePipeline();
  MO::registerGCPipeline();
  MO::registerGCPipelineThroughEmitMojo();
  MO::registerMoPipeline();
  MO::registerMapPipeline();
  MO::registerMoggPipeline();
  MO::registerMgpPipeline();
  MO::registerJitPipeline();
  MO::registerMOToMOGGPasses();
  MO::registerMOToMAPPasses();
  MO::registerMOTransformsPasses();
  MAP::registerMAPTransformsPasses();
  MAP::registerMAPToMOGGPasses();
  MOGG::registerPackageVariadicsAndScalarsPasses();
  MOGG::registerOutlineKernelsIntoFunctionsPass();
  MGP::registerMOGGToMGP();

  MOGG::registerMergeIdenticalKernelStubs();
  MOGG::registerAddTryExceptBlocksPass();
  MOGG::registerLookupConstantResources();
  MGP::registerRaiseBackIndexConstant();
  MOGG::registerOutlineKernelsPass();
  MOGG::registerEmitMojo();
  MOGG::registerDumpMojo();
  MOGG::registerLoadMojo();
  MOGG::registerLowerMOGGRegionCompiles();
  MGP::registerLowerPrimitives();
  MOGG::registerSequenceFusion();

  MGP::registerMGPTransformsPasses();
  MOGG::registerPasses();

  RMO::registerLegalizeRMOOpsPass();

  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerReconcileUnrealizedCastsPassPass();
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerViewOpGraphPass();

  KGEN::registerVerifyParametersPass();
}

LogicalResult MO::runGcPipeline(ModuleOp module,
                                const GCPipelineOptions &gcPipelineOptions) {

  mlir::PassManager pm = mlir::PassManager::on<ModuleOp>(
      module.getContext(), mlir::OpPassManager::Nesting::Implicit);

  if (!gcPipelineOptions.mlirTimingOutDir.empty()) {
    const char *pipelineName = "GC Pipeline";
    auto enabledTiming = M::configureMLIRPassTimingJSONOutput(
        pm, gcPipelineOptions.mlirTimingOutDir.getValue(), pipelineName);
    if (enabledTiming.isError()) {
      llvm::errs() << enabledTiming.getError() << "\n";
      return failure();
    }
  }

  MO::buildGCPipeline(pm, gcPipelineOptions);
  return pm.run(module);
}

// ---------------------------------------------------------------------------
// mo-pipeline
// ---------------------------------------------------------------------------

void M::MO::buildMoPipeline(mlir::OpPassManager &pm,
                            const MoPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // --- import-packages (first pass: populate mogg.kernel.decl ops) ---
  // Skipped when `kernelDeclsPrePopulated` is set: the caller (Python compile
  // path, constant folding) has already seeded the module with decl ops and
  // the `mogg.opaque_type_mapping` attribute via `MOGG::copyKernelDecls`.
  // `mogg-rewrite-opaque-types` immediately below still runs, so user
  // `!mo.opaque<"simple">` references on freshly-cloned ops get rewritten.
  if (!opts.kernelDeclsPrePopulated) {
    pm.addPass(MOGG::createImportPackages(
        {.mojoPkgPaths = kernelLibraryPaths, .litMlirPaths = {}}));
    pm.addPass(MOGG::createVerifyKernels());
  }

  pm.addPass(MOGG::createRewriteOpaqueTypes());
  pm.addPass(KGEN::createVerifyParameters());

  if (opts.debugPrintAllTensors)
    pm.addPass(MO::createDebugPrintAllTensorsPass());

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

  std::optional<std::string> cacheDir;
  if (opts.cacheDir.hasValue())
    cacheDir = opts.cacheDir;

  // Run shape inference and constant fold.
  if (opts.useShapeInference)
    buildShapeInferencePipeline(pm);

  // TODO(GEX-886): Remove dependency of constant folding for symbolic
  // optimization
  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  pm.addPass(MO::createHoistConstantSubgraphsIntoParent());
  pm.addNestedPass<MO::GraphOp>(mlir::createCSEPass());

  // Run symbolic operator set domain-specific optimizations
  SymbolicOptimizationOptions symbolicOptimizationOpts;
  symbolicOptimizationOpts.symbolicFfastMath = opts.symbolicFfastMath;
  symbolicOptimizationOpts.symbolicNoUndefinedMath =
      opts.symbolicNoUndefinedMath;

  // Add fallback shape functions for any remaining unhandled ops.
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addPass(MO::createMergeDuplicateShapeMaterializations());
  pm.addPass(MO::createAddFallbackShapeFunctions());
  pm.addPass(MO::createSymbolicOptimization(symbolicOptimizationOpts));
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

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

  // Wrap per-device consumers of an mo.tensor.unbundle in a fresh
  // mo.parallel, then canonicalize (so bundle(unbundle(X)...) collapses
  // to X), then merge the resulting adjacent mo.parallels into their
  // upstream producers where SSA equality holds.  Together this absorbs
  // per-device consumer chains back into the producing parallel so that
  // MOGG's epilogue fusion sees them as a single kernel body.
  pm.addPass(MO::createCreateParallelFromUnbundle());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createMergeAdjacentParallels());

  if (opts.constantFold)
    pm.addPass(MO::createConstantFoldPass(compiledFrameworkLabel, cacheDir,
                                          kernelLibraryPaths));

  if (opts.fuseJitKernels) {
    // Only run this when kernel fusion is enabled as otherwise we might
    // pessimize model execution by introducing extraneous copies.
    pm.addPass(MO::createElideRedundantMutableOps());
  }

  pm.addPass(MO::createHoistParameterExpressions());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addPass(MO::createRemoveUnusedOps());
}

// ---------------------------------------------------------------------------
// map-pipeline
// ---------------------------------------------------------------------------

void M::MO::buildMapPipeline(mlir::OpPassManager &pm,
                             const MapPipelineOptions &opts) {
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addPass(MAP::createFuseKernels());
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());

  pm.addPass(MAP::createOutlineStructuredKernelBodies());
  pm.addNestedPass<GraphOp>(MAP::createLoopInvariantViewMotion());
}

// ---------------------------------------------------------------------------
// mogg-pipeline
// ---------------------------------------------------------------------------

void M::MO::buildMoggPipeline(mlir::OpPassManager &pm,
                              const MoggPipelineOptions &opts) {
  // Finalize the advanced-fusion MOGG codegen form: package variadics/
  // scalars, then outline each mogg._kernel into a mogg.kernel.stub.
  pm.addPass(MOGG::createPackageVariadicsAndScalars());
  pm.addPass(MOGG::createOutlineKernelsIntoFunctions());

  pm.addPass(MOGG::createSimplifyKernelOperands());

  if (opts.fuseJitKernels) {
    MOGG::FuseElementwiseOptions fuseElementwiseOptions;
    // To dump dot graph, uncomment below:
    // fuseElementwiseOptions.dumpDotGraph = true;
    pm.addPass(MOGG::createFuseElementwise(fuseElementwiseOptions));
    pm.addPass(MOGG::createSimplifyKernelOperands());

    pm.addPass(MOGG::createFuseViews());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    pm.addPass(MOGG::createFusePrologues());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // FuseEpilogues also runs affine view fusion internally until convergence,
    // handling multi-hop cases like: matmul -> add -> affine_view -> add.
    pm.addPass(MOGG::createFuseEpilogues());
    pm.addPass(MOGG::createSimplifyKernelOperands());

    // Inline fully-broadcast constant scalars into kernel lambdas (elementwise,
    // prologue, and epilogue) so they are emitted as literals rather than
    // zero-strided loads. Runs before FuseConstants so the broadcast source is
    // still a constant kernel operand.
    pm.addPass(MOGG::createInlineBroadcastConstantScalars());

    pm.addPass(MOGG::createFuseConstants());
    if (opts.fuseMutableLoadStore)
      pm.addPass(MOGG::createFuseMutableLoads());
  }

  pm.addPass(MOGG::createSimplifyKernelOperands());

  // Add cleanup passes to catch e.g used/duplicated constants.
  pm.addNestedPass<GraphOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<GraphOp>(mlir::createCSEPass());
  pm.addPass(MOGG::createSimplifyKernelOperands());
}

// ---------------------------------------------------------------------------
// mgp-pipeline
// ---------------------------------------------------------------------------

void M::MO::buildMgpPipeline(mlir::OpPassManager &pm,
                             const MgpPipelineOptions &opts) {
  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
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
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  // Before any sort of fusion, attach Python stack traces to ops and asserts.
  pm.addPass(MGP::createAttachPythonStackTraces());

  pm.addPass(MGP::createGlobalizeSlotIndicesPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  MGP::LowerPrimitivesOptions lowerPrimOpts;
  lowerPrimOpts.kernelLibraryPaths = kernelLibraryPaths;
  lowerPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  pm.addPass(MGP::createLowerPrimitives(lowerPrimOpts));

  { // Device-graph synthesis path
    pm.addPass(MGP::createStageDeviceGraphsPass());
    pm.addPass(MGP::createComputeDeviceGraphDependenciesPass());
    pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
    pm.addPass(MGP::createGenerateBuildGraphStubsPass());
    pm.addPass(MGP::createFinalizeDeviceGraphsPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }

  pm.addPass(MOGG::createSequenceFusion());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addPass(MGP::createSimplifyChainsPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<MGP::ModelOp>(mlir::createCSEPass());

  pm.addPass(MGP::createSimplifyExtendLifetimesPass());

  pm.addPass(MOGG::createOutlineKernels());
  pm.addPass(MOGG::createLookupConstantResources());

  MOGG::EmitMojoOptions emitMojoOptions;
  emitMojoOptions.kernelLibraryPaths = kernelLibraryPaths;
  emitMojoOptions.enableOpTraceComments = opts.enableOpTraceComments;
  emitMojoOptions.enableComputeLambdas = opts.enableComputeLambdas;
  emitMojoOptions.moggDebugDumpFusedTensorSpecs =
      opts.moggDebugDumpFusedTensorSpecs;
  pm.addPass(MOGG::createEmitMojo(emitMojoOptions));

  if (opts.dumpMojoText)
    pm.addPass(MOGG::createDumpMojo());
}

namespace {

/// Unified gc-pipeline implementation.
///
/// Builds all stage option structs from `opts` once, then appends every stage's
/// passes to `pm`. When `dumpIntermediates` is set, a DumpStagePass is
/// interleaved before the first stage and after each one, so running `pm`
/// writes the per-stage IR. `pm` is only populated here; the caller runs it.
static void gcPipelineImpl(mlir::OpPassManager &pm,
                           const GCPipelineOptions &opts, bool skipJit,
                           bool dumpIntermediates) {

  auto kernelLibraryPaths =
      llvm::to_vector_of<std::string>(opts.kernelLibraryPaths);

  // Shared stem for intermediate dumps. The first DumpStagePass (the "mo" front
  // dump below) fills it from the user-visible (non-subgraph) model names while
  // the GraphOps still exist; later stage dumps reuse it.
  auto dumpBaseName = std::make_shared<std::string>();

  // Dump the MO input before any pipeline stages run.
  if (dumpIntermediates)
    pm.addPass(MO::createDumpStagePass("mo", dumpBaseName));

  // --- mo-pipeline options ---
  MoPipelineOptions moOpts;
  moOpts.debugPrintAllTensors = opts.debugPrintAllTensors;
  moOpts.useShapeInference = opts.useShapeInference;
  moOpts.constantFold = opts.constantFold;
  if (opts.compiledFrameworkLabel.hasValue())
    moOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel.getValue();
  if (opts.cacheDir.hasValue())
    moOpts.cacheDir = opts.cacheDir.getValue();
  moOpts.symbolicFfastMath = opts.symbolicFfastMath;
  moOpts.symbolicNoUndefinedMath = opts.symbolicNoUndefinedMath;
  moOpts.kernelLibraryPaths = kernelLibraryPaths;
  moOpts.kernelDeclsPrePopulated = opts.kernelDeclsPrePopulated;
  moOpts.usePatterns = opts.usePatterns;
  moOpts.patternFusionMHAEnable = opts.patternFusionMHAEnable;
  moOpts.patternFusionMHAAllowUnsafeBCasts =
      opts.patternFusionMHAAllowUnsafeBCasts;
  moOpts.patternFusionMHAAllowInternalReshape =
      opts.patternFusionMHAAllowInternalReshape;
  moOpts.sanityCheckShapeFuncs = opts.sanityCheckShapeFuncs;
  moOpts.defaultDeviceLabel = opts.defaultDeviceLabel;
  moOpts.defaultDeviceId = opts.defaultDeviceId;
  moOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
  moOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
  moOpts.fuseJitKernels = opts.fuseJitKernels;

  // --- map-pipeline options ---
  MapPipelineOptions mapOpts;

  // --- mogg-pipeline options ---
  MoggPipelineOptions moggOpts;
  moggOpts.fuseJitKernels = opts.fuseJitKernels;
  moggOpts.fuseMutableLoadStore = opts.fuseMutableLoadStore;

  // --- mogg-to-mgp options ---
  MGP::MOGGToMGPOptions moToPrimOpts;
  moToPrimOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  if (opts.compiledFrameworkLabel.hasValue())
    moToPrimOpts.compiledFrameworkLabel = opts.compiledFrameworkLabel;

  // --- mgp-pipeline options ---
  MgpPipelineOptions mgpOpts;
  mgpOpts.kernelLibraryPaths = kernelLibraryPaths;
  mgpOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
  mgpOpts.enableOpTraceComments = opts.enableOpTraceComments;
  mgpOpts.enableComputeLambdas = opts.enableComputeLambdas;
  mgpOpts.moggDebugDumpFusedTensorSpecs = opts.moggDebugDumpFusedTensorSpecs;
  mgpOpts.dumpMojoText = opts.dumpMojoText;

  // --- jit options ---
  MOGG::JitCompileKernelsOptions jitOpts;
  if (!skipJit) {
    jitOpts.minimumCpuAlignment = opts.minimumCpuAlignment;
    jitOpts.saveTempPrefix = opts.saveTempPrefix;
    jitOpts.createMOGGReproducers = opts.createMOGGReproducers;
    jitOpts.kernelLibraryPaths = kernelLibraryPaths;
    jitOpts.timeMOGGCompile = opts.timeMOGGCompile;
    jitOpts.debugKGENStubs = opts.debugKGENStubs;
    // Carry the explicit accelerator arch/triple through to kernel codegen so
    // it cross-compiles for the requested GPU instead of probing the host.
    // These are the same options fed to ValidateDevicesPass above.
    jitOpts.defaultDeviceTargetArch = opts.defaultDeviceTargetArch;
    jitOpts.defaultDeviceTargetTriple = opts.defaultDeviceTargetTriple;
  }

  // Appends one pipeline stage's passes to `pm`, then (when dumping) a
  // DumpStagePass that writes the stage's IR under `dumpSuffix`.
  auto buildStage = [&](auto &&build, const std::string &dumpSuffix) {
    build(pm);
    if (dumpIntermediates && !dumpSuffix.empty())
      pm.addPass(MO::createDumpStagePass(dumpSuffix, dumpBaseName));
  };

  // --- mo-pipeline ---
  buildStage([&](auto &p) { buildMoPipeline(p, moOpts); }, "mo-pre-mogg");

  // --- mo-to-map ---
  buildStage([&](auto &p) { p.addPass(MO::createMOToMAP()); }, "mo-to-map");

  // --- map-pre-mogg: elementwise/prologue/epilogue fusion over MAP kernels,
  // then MAP-dialect prep before the MAP -> MOGG conversion ---
  buildStage([&](auto &p) { buildMapPipeline(p, mapOpts); }, "map-pre-mogg");

  // --- map-to-mogg: the MAP -> MOGG conversion ---
  buildStage([&](auto &p) { p.addPass(MAP::createMAPToMOGG()); },
             "map-to-mogg");

  // --- mo-to-mogg ---
  buildStage(
      [&](auto &p) {
        p.addPass(MO::createMOToMOGG({kernelLibraryPaths, opts.forceSync}));
      },
      "mogg");

  // --- mogg-pipeline (+ optional nan-check) ---
  // Insert NaN/Inf checks at compiled region boundaries. Runs after all fusion
  // passes so checks sit between already-fused compiled regions. The pass
  // creates mo.custom ops and lowers them to MOGG kernels in-place.
  buildStage(
      [&](auto &p) {
        buildMoggPipeline(p, moggOpts);
        if (opts.nanCheck)
          p.addPass(MO::createNanCheckPass(opts.nanCheckStride));
        // Python tracebacks live in OpaqueLocs, which cannot be
        // printed; convert them to FusedLocs so the dump carries the
        // call sites. Limit to dumpIntermediates because the conversion
        // strips the OpaqueLocs that the assert handling in
        // buildMgpPipeline reads.
        if (dumpIntermediates)
          p.addPass(MGP::createAttachPythonStackTraces());
      },
      "mogg-pre-mgp");

  // --- mogg-to-mgp ---
  buildStage([&](auto &p) { p.addPass(MGP::createMOGGToMGP(moToPrimOpts)); },
             "mgp-pre-pipeline");

  // --- mgp-pipeline + jit + jit-pipeline ---
  buildStage(
      [&](auto &p) {
        buildMgpPipeline(p, mgpOpts);
        if (!skipJit) {
          p.addPass(MOGG::createJitCompileKernels(jitOpts));
          p.addPass(MOGG::createStripKernelDecls());
          buildJitPipeline(p);
        }
      },
      "mgp");
}

} // namespace

// ---------------------------------------------------------------------------
// gc-pipeline  (full pipeline: mo-pipeline + mo-to-mogg + mogg-pipeline +
//               mogg-to-mgp + mgp-pipeline + jit + jit-pipeline)
// gc-pipeline-pre-jit: same through mgp-pipeline (emit-mojo); no jit
// ---------------------------------------------------------------------------

void M::MO::buildGCPipeline(mlir::OpPassManager &pm,
                            const GCPipelineOptions &opts) {
  gcPipelineImpl(pm, opts, /*skipJit=*/false);
}

void M::MO::buildGCPipelineThroughEmitMojo(mlir::OpPassManager &pm,
                                           const GCPipelineOptions &opts) {
  gcPipelineImpl(pm, opts, /*skipJit=*/true);
}

LogicalResult M::MO::runGCPipeline(ModuleOp moduleOp,
                                   const GCPipelineOptions &opts,
                                   bool dumpIntermediates) {
  auto pm = mlir::PassManager::on<ModuleOp>(
      moduleOp.getContext(), mlir::OpPassManager::Nesting::Implicit);
  // gcPipelineImpl populates `pm`; when dumping, it interleaves DumpStagePasses
  // that write each stage's IR as `pm` runs.
  gcPipelineImpl(pm, opts, /*skipJit=*/false, dumpIntermediates);
  return pm.run(moduleOp);
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
        auto elementsAttr = constOp.getValue();
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
