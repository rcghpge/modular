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

/// Collects all the options which influence buildMoToMGPPipeline.
struct PassOptions {
  PassOptions() = default;

  /// Framework label to compile to (used in mgp.model framework tag).
  std::string compiledFrameworkLabel;

  /// Alignment for all CPU-allocated buffers.
  size_t minimumCpuAlignment = kPreferredMemoryAlignment;

  /// Should we compile kernels on the fly, if not will use the C++ shims.
  bool useJit = true;

  /// When jitting kernels should we enable the KGEN search.
  bool useKgenSearch = false;

  /// Fusing kernels is on by default but will do nothing if useJit is off. If
  /// use jit is on then we will fuse incoming jit kernels.
  bool fuseJitKernels = true;

  /// Lower to hand-fused kernels using pattern matching
  bool usePatterns = true;

  /// Path used by the CAS subsystem to establish its backing store.
  /// If non-empty then models will be rewritten to outline dense_attributes
  /// to the given CAS instance.
  std::string cacheDir;
};

// Create the canonical MO pipeline for the given options.
void buildMoToMGPPipeline(mlir::PassManager &pm, const PassOptions &opts);

// Registers the end-to-end shape inference pipeline.
void registerShapeInferencePipeline();

// Create the end-to-end shape inference pipeline.
void buildShapeInferencePipeline(mlir::OpPassManager &pm);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
