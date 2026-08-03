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

    Gated on `M::MO::useAdvancedFusion()`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def OutlineStructuredKernelBodies
    : Pass<"map-outline-structured-kernel-bodies", "mlir::ModuleOp"> {
  let summary =
      "Outline structured MAP kernel bodies into `map.iter.opaque` regions.";
  let description = [{
    Wraps each structured MAP loop (`map.iter.foreach`, and future reduce /
    rowwise) in a `map.iter.opaque` region so sequential setup (e.g. strided-view
    lifting) can live outside the structured loop body. After outlining, nothing
    inspects the inside of the `map.iter.opaque` again.

    This is scaffolding gated on `M::MO::useAdvancedFusion()`; it currently does
    nothing.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
