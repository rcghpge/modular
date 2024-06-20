//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_PASSES_H
#define GENERICML_MO_GRAPHCOMPILER_PASSES_H

#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "LLCL/ForwardDecls.h"
#include "Support/AlignedAlloc.h"
#include "mlir/Pass/PassOptions.h"

namespace M::MO {

/// Collects mlir options which influence buildMoToMgpPipeline.
///
/// See also GraphCompilerCLOptions for the corresponding mt command line
/// options.
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
  Option<bool> symbolicFfastMath{
      *this, "symbolic-ffast-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed (1) to operate under the "
          "finite math assumption, (2) re-order floating point operations and "
          "(3) apply reciprocal optimizations"),
      llvm::cl::init(true)};
  Option<bool> symbolicNoUndefinedMath{
      *this, "symbolic-no-undefined-math",
      llvm::cl::desc(
          "If the symbolic optimizer is allowed to assume that there will be no"
          "undefined math operations such as divisions by 0"),
      llvm::cl::init(true)};
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
  Option<bool> enableFakeCudaDevice{
      *this, "enable-fake-cuda-device",
      llvm::cl::desc(
          "If true, compile assuming a single CUDA GPU is available. "
          "For testing only."),
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
  Option<bool> enableCompiledRegionsFusion{
      *this, "enable-compiled-regions-fusion",
      llvm::cl::desc("If false, will disable compiled region fusion"),
      llvm::cl::init(false)};
  Option<bool> skipJitCompileMiddle{
      *this, "skip-jit-compile-middle",
      llvm::cl::desc(
          "If false, will use the jit compile in the middle code "
          "path for mogg kernels. Otherwise, lower to the compiled region"),
      llvm::cl::init(false)};
  Option<bool> genVizSpec{
      *this, "gen-viz-spec",
      llvm::cl::desc(
          "If true, generates a visualization spec compatible with Netron."),
      llvm::cl::init(false)};
  Option<std::string> vizOutputPath{
      *this, "viz-output-path",
      llvm::cl::desc(
          "If genVizSpec, can be used to specify the output path for "
          "the visualization spec."),
      llvm::cl::init("")};
  Option<bool> debugPrintAllTensors{
      *this, "debug-print-all-tensors",
      llvm::cl::desc("If enabled, print all intermediate tensors. The usual "
                     "'result-output-style', 'print-precision' and "
                     "'binary-output-dir' flags may be used to control the "
                     "printed format and saved location."),
      llvm::cl::init(false)};
  ListOption<std::string> dimNames{
      *this, "dim-names",
      llvm::cl::desc("Named dimensions for each input of the MO graph")};
  Option<size_t> softFusionNonHostKernelLimit{
      *this, "soft-fusion-non-host-kernel-limit",
      llvm::cl::desc(
          "Maximum number of non-host kernels to place in a single soft fused "
          "kernel."),
      llvm::cl::init(3)};
  Option<size_t> minPinnedHostBufferSize{
      *this, "min-pinned-host-buffer-size",
      llvm::cl::desc(
          "If non-zero, replace host-side allocations with pinned allocations "
          "when the buffer is at least this size in bytes, and is the source "
          "or target of a host/device memory transfer."),
      llvm::cl::init(0)};
  Option<bool> timeMOGGCompile{
      *this, "time-mogg-compile",
      llvm::cl::desc("Turns on timing for the KGEN passes invoked by mogg"),
      llvm::cl::init(false)};
  Option<bool> patternFusionMHAEnable{
      *this, "mha-enable",
      llvm::cl::desc("Enables multi-head attention (mha) pattern fusion."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowUnsafeBCasts{
      *this, "mha-allow-unsafe-bcast",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> patternFusionMHAAllowInternalReshape{
      *this, "mha-allow-internal-reshape",
      llvm::cl::desc("If the mha pattern is enabled, allows fusing potentially "
                     "unsafe bcasts with a runtime guard."),
      llvm::cl::init(true)};
  Option<bool> forceSync{*this, "force-sync",
                         llvm::cl::desc("Make all kernels synchronous."),
                         llvm::cl::init(false)};
};

struct ShapeInferenceOptions
    : public mlir::PassPipelineOptions<ShapeInferenceOptions> {
  ListOption<std::string> kernelLibraryPaths{
      *this, "kernel-library-paths",
      llvm::cl::desc(
          "Paths to mojo packages containing mojo shape function definitions")};
  Option<bool> constantFold{
      *this, "constant-fold",
      llvm::cl::desc("For use with the extended pipeline. Whether to use "
                     "constant-folding."),
      llvm::cl::init(true)};
  Option<std::string> compiledFrameworkLabel{
      *this, "compiled-framework-label",
      llvm::cl::desc(
          "For use with the extended pipeline / constant-folding. Framework "
          "label to compile to (used in mgp.model framework tag).")};
};

/// Registers the MO-to-MGP pipeline for use by testing tools.
/// The passes will use the current runtime, which must be established by the
/// tool.
void registerMoToMgpPipeline();

/// Creates the canonical MO-to-MGP pipeline for the given options.
void buildMoToMgpPipeline(mlir::OpPassManager &pm, const MoToMgpOptions &opts);

/// Registers the end-to-end shape inference pipeline for use by testing tools.
void registerShapeInferencePipeline();

// Create the end-to-end shape inference pipeline.
void buildShapeInferencePipeline(mlir::OpPassManager &pm,
                                 const ShapeInferenceOptions &opts,
                                 const MoToMgpOptions &moToMgpOptions);
void buildShapeInferencePipelineDefaultOpts(mlir::OpPassManager &pm,
                                            const ShapeInferenceOptions &opts);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
