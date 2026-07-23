//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def FuseKernels : Pass<"map-fuse-kernels", "mlir::ModuleOp"> {
  let summary = "Fuse MAP kernels by inlining producers into consumers.";
  let description = [{
    Runs the advanced fusion patterns over the MAP dialect, reducing the number
    of `map` kernels by inlining producers into consumers.

    This is scaffolding gated on `M::MO::useAdvancedFusion()`; it currently does
    nothing.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
