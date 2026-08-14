//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def FuseKernels : Pass<"map-fuse-kernels", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels by inlining producers into consumers.";
  let description = [{
    Runs the advanced fusion patterns over the MAP dialect, reducing the number
    of `map` kernels by inlining producers into consumers.

    Anchored on `mo.graph` (which is `IsolatedFromAbove` and always a direct
    child of the module), so the pass never walks the imported kernel decls or
    the insides of kernels it already visited, and independent graphs run in
    parallel.
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

def LoopInvariantViewMotion : Pass<"map-loop-invariant-view-motion", "M::MAP::IterOpaqueOp"> {
  let summary =
      "Hoist chains of index-transform ops out of a structured loop body.";
  let description = [{
    Runs after `OutlineStructuredKernelBodies`, once per `map.iter.opaque`.
    Scoping to a single `map.iter.opaque` (rather than the whole module)
    relies on it being `IsolatedFromAbove`, and lets independent instances run
    in parallel.

    For each `map.load` inside a nested `map.iter.foreach.body`, walks
    backward from its index operand through any chain of index-transform ops
    (ops implementing `MAP::IndexTransformOpInterface`; today:
    `map.index.slice`) back to the loop's own (untransformed) index. When
    such a chain exists, hoists it, hop by hop, into a chain of "stride" ops
    (today: `map.index.slice.stride`) placed in the enclosing
    `map.iter.opaque`'s body -- outside the loop, so each transform runs once
    per kernel invocation rather than once per index -- and rewrites the load
    to read the fully-hoisted tensor with the chain's outermost
    (untransformed) index directly.

    Loads with no such chain (already indexing directly with the loop's own
    index) are left untouched. Multiple loads that hoist the exact same chain
    (SSA-identical, not just structurally similar) share one hoisted op.

    Invariant enforced at the end of the pass: no index-transform op survives
    inside any `map.iter.foreach.body`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
  ];
}

def FuseMutableLoads : Pass<"map-fuse-mutable-loads", "MO::GraphOp"> {
  let summary = "'Fuse' mo.mutable.load ops with map.iter.foreach ops.";
  let description = [{
    Will perform 'fusion' of a mo.mutable.load op with a map.iter.foreach op.
    This amounts to:
      1. Replacing the value semantic tensor created by the mutable.load op
         with the buffer semantic tensor
      2. Making the input chain of the load op an input of the
         map.iter.foreach op
      3. Adding the output chain of the mutable.load op to the output of the
         map.iter.foreach op
  }];
  let dependentDialects = ["M::MAP::MAPDialect"];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
