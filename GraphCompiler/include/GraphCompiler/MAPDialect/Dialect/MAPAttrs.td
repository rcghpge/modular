//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for attributes in this dialect.
class MAP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MAPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def MAP_ViewSignatureAttr : MAP_Attr<"ViewSignature", "view.signature"> {
  let summary = "Input/output shape signature shared by all MAP views";
  let description = [{
    The static input/output shapes of a view op (e.g. a slice), reusing
    `MOSH::ShapeAttr` (the same representation `map.iter.foreach`'s own
    `shape` attribute uses) for each side. Every view kind embeds one of these
    for its in/out shapes rather than redeclaring its own pair of fields.
    Printed as `in -> out`, e.g.:

    ```mlir
    #map.view.signature<[?, 64] -> [?, 32]>
    ```
  }];

  let parameters = (ins
    "MOSH::ShapeAttr":$in,
    "MOSH::ShapeAttr":$out
  );

  // Custom parse/print: `in`/`out` reuse `MOSH::parseShapeAttr`/
  // `printShapeTypedAttr` (the same helpers `map.iter.foreach`'s custom
  // parser uses for its own `shape` attr), joined by an arrow.
  let hasCustomAssemblyFormat = 1;
}

def MAP_SliceSpecAttr : MAP_Attr<"SliceSpec", "index.slice.spec"> {
  let summary = "Static shape and per-dimension bounds for a slice view";
  let description = [{
    Compile-time-known shape/bounds metadata for a `mo.slice` view lowered
    through the MAP dialect. `signature` is the slice's input/output shape
    signature (same rank — slice never changes rank). `starts`/`stops`/
    `steps` carry one entry per dimension: the literal value where that
    dimension's bound is a compile-time constant (e.g. the `mo.slice` operand
    was a `mo.constant`), or `?` (`ShapedType::kDynamic`) where it isn't and
    must be read at runtime from the corresponding
    `map.index.slice`/`map.index.slice.stride` tensor operand at that
    dimension instead — the same `?`-for-unknown convention MLIR shaped
    types already use (e.g. `tensor<?x4xf32>`).

    Example:
    ```mlir
    #sspec = #map.index.slice.spec<[?, 64] -> [?, 32], starts = [?, 32], stops = [?, 64], steps = [1, 1]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature,
    "mlir::DenseI64ArrayAttr":$starts,
    "mlir::DenseI64ArrayAttr":$stops,
    "mlir::DenseI64ArrayAttr":$steps
  );

  // Custom parse/print (not a declarative `assemblyFormat`): `signature`
  // embeds `MAP_ViewSignatureAttr`'s own `in -> out` form directly (same
  // inlining convention `MOSH::ShapeAttr` uses elsewhere), and
  // `starts`/`stops`/`steps` print `?` for `ShapedType::kDynamic` entries
  // instead of the raw sentinel a plain `DenseI64ArrayAttr` would print.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins
      "MOSH::ShapeAttr":$in,
      "MOSH::ShapeAttr":$out,
      "llvm::ArrayRef<int64_t>":$starts,
      "llvm::ArrayRef<int64_t>":$stops,
      CArg<"llvm::ArrayRef<int64_t>", "{}">:$steps
    ), [{
      // Canonicalize steps when omitted: infer per-dimension from
      // (stop - start) / out (1 for the common contiguous case), falling
      // back to dynamic wherever that isn't statically determinable.
      llvm::SmallVector<int64_t> inferredSteps;
      if (steps.empty()) {
        inferredSteps = MAP::inferSliceSteps(out, starts, stops);
        steps = inferredSteps;
      }
      MLIRContext *ctx = in.getContext();
      return $_get(ctx, MAP::ViewSignatureAttr::get(ctx, in, out),
                   mlir::DenseI64ArrayAttr::get(ctx, starts),
                   mlir::DenseI64ArrayAttr::get(ctx, stops),
                   mlir::DenseI64ArrayAttr::get(ctx, steps));
    }]>
  ];
}

def MAP_BroadcastSpecAttr : MAP_Attr<"BroadcastSpec", "index.broadcast.spec"> {
  let summary = "Static input/output shape signature for a broadcast view";
  let description = [{
    Compile-time-known shape metadata for a `mo.static.broadcast_to` view
    lowered through the MAP dialect. `signature` is the broadcast's
    input/output shape signature -- unlike slice, broadcast can add rank
    (numpy semantics): `in` aligns with `out`'s trailing dimensions, and
    every leading `out` dimension `in` doesn't have is replicated. An aligned
    dimension where `in`'s size is 1 is also replicated; deciding this for a
    dimension that isn't statically known either way is deferred to the
    primitive `MAPToMOGG` lowers into, not resolved by this attribute.

    Example:
    ```mlir
    #bspec = #map.index.broadcast.spec<[1, 64] -> [8, 64]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature
  );

  // Custom parse/print: `signature` embeds `MAP_ViewSignatureAttr`'s own
  // `in -> out` form directly, same inlining convention `MAP_SliceSpecAttr`
  // uses for its own `signature` field.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def MAP_ReshapeSpecAttr : MAP_Attr<"ReshapeSpec", "index.reshape.spec"> {
  let summary = "Static input/output shape signature for a reshape view";
  let description = [{
    Compile-time-known shape metadata for a `mo.static.reshape` view lowered
    through the MAP dialect. `signature` is the reshape's input/output shape
    signature -- unlike slice/transpose, rank can change in either direction
    (reshape only preserves the total element count, not the rank), and
    unlike broadcast/slice/transpose, the projection has no other per-view
    parameters: it is a pure function of `signature`'s in/out shapes alone.

    Example:
    ```mlir
    #rspec = #map.index.reshape.spec<[2, 3] -> [3, 2]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature
  );

  // Custom parse/print: `signature` embeds `MAP_ViewSignatureAttr`'s own
  // `in -> out` form directly, same inlining convention `MAP_SliceSpecAttr`
  // uses for its own `signature` field.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

def MAP_TransposeSpecAttr : MAP_Attr<"TransposeSpec", "index.transpose.spec"> {
  let summary = "Static shape signature and permutation for a transpose view";
  let description = [{
    Compile-time-known shape/permutation metadata for a `mo.transpose` view
    lowered through the MAP dialect. `signature` is the transpose's
    input/output shape signature (same rank -- transpose never changes rank,
    only reorders dimensions). `perm` is the permutation applied to `in`'s
    dimensions to produce `out`: `out.dim[i] == in.dim[perm[i]]`, matching
    `mo.transpose`'s own `perm` operand. Only reachable once `perm` is fully
    known at compile time (a `mo.constant`); a `mo.transpose` whose `perm`
    isn't a compile-time constant doesn't lower through this path at all.

    Example:
    ```mlir
    #tspec = #map.index.transpose.spec<[2, 3] -> [3, 2], perm = [1, 0]>
    ```
  }];

  let parameters = (ins
    "MAP::ViewSignatureAttr":$signature,
    "mlir::DenseI64ArrayAttr":$perm
  );

  // Custom parse/print: `signature` embeds `MAP_ViewSignatureAttr`'s own
  // `in -> out` form directly, same inlining convention `MAP_SliceSpecAttr`
  // uses for its own `signature` field.
  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD
