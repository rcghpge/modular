//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_PASSES_H
#define GENERICML_MO_GRAPHCOMPILER_PASSES_H

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/GraphCompilerCLOptions.h"
#include "LLCL/ForwardDecls.h"

namespace M::MO {

/// Collects all the options which influence buildMoToMgpPipeline.
struct MoToMgpOptions : public mlir::PassPipelineOptions<MoToMgpOptions> {
  Option<std::string> compiledFrameworkLabel{
      *this, "compiled-framework-label",
      llvm::cl::desc(
          "Framework label to compile to (used in mgp.model framework tag).")};
  Option<size_t> minimumCpuAlignment{
      *this, "minimum-cpu-alignment",
      llvm::cl::desc("Alignment for all CPU-allocated buffers."),
      llvm::cl::init(kPreferredMemoryAlignment)};
  Option<bool> useKgenSearch{
      *this, "use-kgen-search",
      llvm::cl::desc("When jitting kernels should we enable the KGEN search."),
      llvm::cl::init(false)};
  Option<bool> fuseJitKernels{
      *this, "fuse-jit-kernels",
      llvm::cl::desc(
          "Fusing kernels is on by default but will do nothing if useJit is "
          "off. If use jit is on then we will fuse incoming jit kernels."),
      llvm::cl::init(true)};
  Option<bool> usePatterns{
      *this, "use-patterns",
      llvm::cl::desc("Lower to hand-fused kernels using pattern matching."),
      llvm::cl::init(true)};
  Option<std::string> cacheDir{
      *this, "cache-dir",
      llvm::cl::desc("Path used by the CAS subsystem to establish its backing "
                     "store. If non-empty then models will be rewritten to "
                     "outline dense_attributes to the given CAS instance.")};
  ListOption<std::string> kernelLibraryPaths{
      *this, "kernel-library-paths",
      llvm::cl::desc(
          "Paths to mojo packages containing mojo kernel definitions used to "
          "replace MO ops.")};
  Option<bool> useShapeInference{
      *this, "mo-shape-inference",
      llvm::cl::desc("Run shape inference on the input before other "
                     "optimizations and lowering."),
      llvm::cl::init(true)};
  Option<bool> cacheBuffersAcrossExecutes{
      *this, "cache-buffers-across-executes",
      llvm::cl::desc(
          "If true, use realloc instead of alloc for all buffers, "
          "thus sharing the cost of buffer allocation across "
          "calls to execute at the cost of higher memory high water mark."),
      llvm::cl::init(false)};
  Option<bool> performTrivialKernelOpt{
      *this, "trivial-kernel-opt",
      llvm::cl::desc(
          "If true we will promote trivial kernels to synchronous kernels"
          "which then can be soft-fused together. This is the default but"
          "is available as a flag for tests."),
      llvm::cl::init(true)};
  Option<std::string> saveTempPrefix{
      *this, "save-temp-prefix",
      llvm::cl::desc(
          "The prefix to use when saving temporary mogg files. Setting this"
          "option to any value will toggle on mogg.saveTemps which will save"
          "assembly/IR to files this prefix."),
      llvm::cl::init("")};
  Option<bool> createMOGGReproducers{
      *this, "create-mogg-reproducers",
      llvm::cl::desc(
          "Prints isolated reproducers of each compiled mogg kernel in"
          "a newly created mo graph."),
      llvm::cl::init(false)};
  Option<bool> constantFold{
      *this, "constant-fold",
      llvm::cl::desc("Evaluate parts of the program that can be evaluated at "
                     "compile time."),
      llvm::cl::init(true)};
  Option<bool> lowerQDQOperators{
      *this, "lower-quantized-operators",
      llvm::cl::desc("Dispatch to quantized operations around Q-DQ regions."),
      llvm::cl::init(true)};
  Option<bool> enableExtraDevices{
      *this, "enable-extra-devices",
      llvm::cl::desc("Include all host devices instead of just the CPU during "
                     "compilation."),
      llvm::cl::init(false)};
  Option<std::string> defaultDeviceLabel{
      *this, "default-device-label",
      llvm::cl::desc(
          "If non-empty, assign as many ops as possible to the device "
          "with the given label (and device id zero)."),
      llvm::cl::init("")};
  Option<bool> enableMemoryPlanning{
      *this, "enable-memory-planning",
      llvm::cl::desc("If true, statically plan as many buffer allocations as "
                     "possible."),
      llvm::cl::init(false)};
  Option<bool> genVizSpec{
      *this, "gen-viz-spec",
      llvm::cl::desc(
          "If true, generates a visualization spec compatible with Netron."),
      llvm::cl::init(false)};
  Option<std::string> vizOutputDir{
      *this, "viz-output-dir",
      llvm::cl::desc(
          "If genVizSpec, can be used to specify the output directory for "
          "the visualization spec."),
      llvm::cl::init("")};
};

// Registers a runtime for use by the MO-to-MGP pipeline when invoked via the
// MlirOptMain testing driver. The main entry point should call this before
// MlirOptMain if required.
//
// FOR TESTING ONLY: Production compilation paths should call
// buildMoToMgpPipeline directly with the appropriate options and runtime
// arguments.
void registerMoToMgpRuntimeForTesting(Runtime &runtime);

// Registers the MO-to-MGP pipeline for use by testing tools.
void registerMoToMgpPipeline();

// Creates the canonical MO-to-MGP pipeline for the given options. If a runtime
// is provided it may be used by passes to derive runtime context information
// and run compilation sub-tasks.
void buildMoToMgpPipeline(mlir::OpPassManager &pm, const MoToMgpOptions &opts,
                          Runtime *optRuntime = nullptr);

// Registers the end-to-end shape inference pipeline.
void registerShapeInferencePipeline();

// Create the end-to-end shape inference pipeline.
void buildShapeInferencePipeline(mlir::OpPassManager &pm);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
