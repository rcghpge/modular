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
    lifting) can live outside the structured loop body. The loop's own shape and
    body move into a nested `map.iter.foreach.body`, which reaches the wrapper's
    inputs/outputs by SSA dominance rather than by operand or block argument.
    After outlining, nothing inspects the inside of the `map.iter.opaque` again.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
