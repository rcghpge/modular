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

    Repeats epilogue, elementwise, view, then prologue fusion until a round
    fuses nothing, since folding one link of a chain can expose the next.
    Reports the total number of kernels fused as a pass statistic.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numKernelsFused", "num-kernels-fused", "Number of kernels fused">
  ];
}

def FuseEpilogue : Pass<"map-fuse-epilogue", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via EpilogueFuser only. For testing only.";
  let description = [{
    Runs `EpilogueFuser` in isolation over `mo.graph`, skipping the other
    fusers `FuseKernels` also runs. Lets a lit test exercise epilogue fusion
    without depending on the other fusers having already run. Not part of
    `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseReshapeEpilogue : Pass<"map-fuse-reshape-epilogue", "MO::GraphOp"> {
  let summary =
      "Fuse MAP kernels via ReshapeEpilogueFuser only. For testing only.";
  let description = [{
    Runs `ReshapeEpilogueFuser` in isolation over `mo.graph`, skipping the
    other fusers `FuseKernels` also runs. Lets a lit test exercise
    reshape-into-epilogue fusion without depending on the other fusers having
    already run. Not part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseElementwise : Pass<"map-fuse-elementwise", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via ElementwiseFuser only. For testing only.";
  let description = [{
    Runs `ElementwiseFuser` in isolation over `mo.graph`, skipping the other
    fusers `FuseKernels` also runs. Lets a lit test exercise elementwise
    fusion without depending on the other fusers having already run. Not
    part of `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseViews : Pass<"map-fuse-views", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via ViewFuser only. For testing only.";
  let description = [{
    Runs `ViewFuser` in isolation over `mo.graph`, skipping the other fusers
    `FuseKernels` also runs. Lets a lit test exercise view fusion without
    depending on the other fusers having already run. Not part of
    `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numKernelsFused", "num-kernels-fused", "Number of kernels fused">
  ];
}

def FusePrologue : Pass<"map-fuse-prologue", "MO::GraphOp"> {
  let summary = "Fuse MAP kernels via PrologueFuser only. For testing only.";
  let description = [{
    Runs `PrologueFuser` in isolation over `mo.graph`, skipping the other
    fusers `FuseKernels` also runs. Lets a lit test exercise prologue fusion
    without depending on the other fusers having already run. Not part of
    `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseInput : Pass<"map-fuse-input", "MO::GraphOp"> {
  let summary = "Unconstrained generic input fusion. For testing only.";
  let description = [{
    Runs generic input fusion over `mo.graph` with no wrapper constraint --
    every eligible producer/consumer edge fuses, subject only to the shared
    correctness constraints. Lets a lit test exercise the generic directly,
    unconstrained by the elementwise/view/prologue narrowing. Not part of
    `buildMapPipeline`.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
  ];
}

def FuseOutput : Pass<"map-fuse-output", "MO::GraphOp"> {
  let summary = "Unconstrained generic output fusion. For testing only.";
  let description = [{
    Runs generic output fusion over `mo.graph` with no wrapper constraint --
    every eligible epilogue edge fuses, subject only to the shared correctness
    constraints. Lets a lit test exercise the generic directly, unconstrained by
    the epilogue/reshape-epilogue narrowing. Not part of `buildMapPipeline`.
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

def RaiseBackContiguousViews
    : Pass<"map-raise-back-contiguous-views", "MO::GraphOp"> {
  let summary = "Raise unfused slice/reshape view kernels back to MO/MOGG ops.";
  let description = [{
    Runs after `FuseKernels` reaches a fixed point, before
    `OutlineStructuredKernelBodies`. Finds `map.iter.foreach` view kernels
    that never got fused into anything and raises them back into ops the
    pre-advanced-fusion pipeline already knows how to lower without a copy: a
    contiguous slice becomes `mogg.slice` (a zero-copy buffer slice, later
    lowered by `MOGGToMGP` to `mgp.tensor.slice`), and a reshape becomes
    `mo.static.reshape` (which `MOGGToMGP` turns into a non-copying
    primitive). Scope mirrors the legacy `mogg-fuse-views` pass's own
    raise-back step: slice and reshape only, not broadcast/transpose.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MOGG::MOGGDialect",
  ];

  let statistics = [
    Statistic<"numSlicesRaised", "num-slices-raised",
              "Number of contiguous slice kernels raised back">,
    Statistic<"numReshapesRaised", "num-reshapes-raised",
              "Number of reshape kernels raised back">
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

def AllocateAccumulatingBuffers
    : Pass<"map-allocate-accumulating-buffers", "MO::GraphOp"> {
  let summary = "Give every fused-output epilogue a real store pointer.";
  let description = [{
    Runs after `FuseKernels`. Sets each `map.iter.epilogue`'s `$intermediateTensor`
    so `MAPToMOGG` can always hand the lowered epilogue a real base pointer
    instead of a dangling one:

    - Normally the pointer source is the tensor the epilogue stores into.
    - For a mixed-precision accumulating kernel (`containsMatmulOrConvOrConcat`
      whose epilogue changes dtype), the kernel must accumulate in the pre-cast
      dtype before the lambda casts and stores. A fresh accumulation buffer of
      that dtype is allocated as an extra kernel output and used as the pointer
      source; the store target stays the real (post-cast) output.

    No buffer is allocated for a non-accumulating kernel -- it only needs a
    valid pointer, which the store target already provides.
  }];
  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MO::MODialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTRANSFORMS_TD
