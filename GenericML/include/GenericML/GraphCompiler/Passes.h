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
  Option<bool> useJit{*this, "use-jit",
                      llvm::cl::desc("Should we compile kernels on the fly, if "
                                     "not will use the C++ shims."),
                      llvm::cl::init(true)};
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
  Option<std::string> extraLibPaths{
      *this, "extra-lib-paths",
      llvm::cl::desc("Extra library paths to load kernel sources from.")};
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
  Option<bool> includeHostTargetInfo{
      *this, "include-host-target-info",
      llvm::cl::desc(
          "If true, the generated model will capture the TargetInfoAttr of the "
          "compile-time host within the generated mgp.model. A runtime check "
          "will then be made to ensure the runtime host satisfies the "
          "assumptions made at compile time."),
      llvm::cl::init(true)};
  Option<bool> constantFold{
      *this, "constant-fold",
      llvm::cl::desc("Evaluate parts of the program that can be evaluated at "
                     "compile time."),
      llvm::cl::init(false)}; // TODO: enable by default #16888
};

// Registers the end-to-end shape inference pipeline.
void registerMoToMgpPipeline();

// Create the canonical MO pipeline for the given options.
void buildMoToMgpPipeline(mlir::OpPassManager &pm, const MoToMgpOptions &opts);

// Registers the end-to-end shape inference pipeline.
void registerShapeInferencePipeline();

// Create the end-to-end shape inference pipeline.
void buildShapeInferencePipeline(mlir::OpPassManager &pm);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
