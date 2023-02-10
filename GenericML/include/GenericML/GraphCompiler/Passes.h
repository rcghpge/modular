//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MO_GRAPHCOMPILER_PASSES_H
#define GENERICML_MO_GRAPHCOMPILER_PASSES_H

#include "GenericML/GraphCompiler/Conversion/MOToPrimitives.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"

namespace M::MO {

struct PassOptions {
  PassOptions()
      : moToPrimOpts(), useJit(false), useKgenSearch(false),
        fuseJitKernels(true) {}

  /// Options to propagate to MoToPrimitives.
  MGP::MOToPrimitivesOptions moToPrimOpts;

  /// Should we compile kernels on the fly, if not will use the C++ shims.
  bool useJit;

  /// When jitting kernels should we enable the KGEN search.
  bool useKgenSearch;

  /// Fusing kernels is on by default but will do nothing if useJit is off. If
  /// use jit is on then we will fuse incoming jit kernels.
  bool fuseJitKernels;
};

// Create the canonical MO pipeline for the given options.
void buildMoToMGPPipeline(mlir::PassManager &pm, const PassOptions &opts);

} // namespace M::MO

#endif // GENERICML_MO_GRAPHCOMPILER_PASSES_H
