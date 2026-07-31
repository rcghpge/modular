//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "GraphCompiler/MAPDialect/Dialect/MAPTypes.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/MOGGDialect/MOGGAttrs.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MAP_Op<string mnemonic, list<Trait> traits = []>
    : Op<MAPDialect, mnemonic, traits>;

// Type constraint matching the `map.index<N>` type.
def MAP_AnyIndex : Type<
    CPred<"::llvm::isa<::M::MAP::IndexType>($_self)">,
    "map.index type">;

//===----------------------------------------------------------------------===//
// map.iter.opaque
//===----------------------------------------------------------------------===//

def MAP_IterOpaqueOp : MAP_Op<"iter.opaque", [
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  ]> {
  let summary = "An opaque (hand-written) kernel with fusible load/store paths.";
  let description = [{
    Wraps a call to an opaque Mojo kernel (e.g. a hand-tuned GEMM) that exposes
    named load/store callbacks. The body holds the load/store callbacks
    (`map.iter.prologue` / `map.iter.epilogue`) plus the `map.call.opaque` that
    invokes the kernel; it is not a sequence of computation ops.

    The `inputs` are the kernel's tensor operands. The `outputs` are the kernel
    results; each becomes a block argument of the body so the callbacks and the
    `map.call.opaque` can address it, and the op result of the same type
    carries the filled tensor out. When every result is produced this way (the
    common case today), the body has no terminator: the output block arguments
    themselves are the results, the same destination-passing-style convention
    `mogg._kernel` uses. A trailing `map.yield` is only needed if some
    `map.call.opaque` in the body produces an actual (non-DPS) return value
    that isn't already one of the output block arguments.

    ```mlir
    %mm = map.iter.opaque
        inputs (%A : !mo.tensor<[M, K], bf16>, %B : !mo.tensor<[K, N], bf16>)
        outputs (%out : !mo.tensor<[M, N], bf16>) {
      %epi = map.iter.epilogue (%val : bf16, %idx : !map.index<2>) capturing (%out) {
        map.store %out[%idx] = %val : !mo.tensor<[M, N], bf16>
      } : !mo.tensor<[M, N], bf16>
      map.call.opaque @builtin_kernels.linalg.Matmul[transpose_b = false]
        inputs (%A, %B) outputs (%epi) : (...) -> (...)
    }
    ```

    `fusionInfo` records the fusion-tree provenance of the kernel (a
    `mogg.fusion` attribute), taking over the role the attribute of the same
    name played on `mogg.kernel`. It is typed as `AnyAttr` here so the `map`
    dialect need not depend on `mogg`.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<AnyAttr>:$fusionInfo
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    /// The subrange of body entry-block arguments that are the DPS outputs:
    /// the trailing `getNumResults()` arguments (block arguments are ordered
    /// [inputs..., outputs...], outputs last).
    ::mlir::Block::BlockArgListType getOutputBlockArgs();

    /// The subrange of body entry-block arguments shadowing `$inputs` --
    /// whatever precedes the output block args. Today `$inputs` are not
    /// themselves shadowed, so this is always empty; anchoring the split on
    /// the (always known) result count rather than the input count keeps this
    /// correct either way.
    ::mlir::Block::BlockArgListType getInputBlockArgs();
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// map.yield
//===----------------------------------------------------------------------===//

def MAP_YieldOp : MAP_Op<"yield", [Terminator]> {
  let summary = "Terminates a map.iter.opaque body, yielding its results.";
  let description = [{
    An optional terminator of a `map.iter.opaque` body, needed only when some
    `map.call.opaque` produces a return value that is not already one of the
    body's output block arguments (the DPS case, the common one today, has no
    terminator at all).
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// map.iter.epilogue
//===----------------------------------------------------------------------===//

def MAP_IterEpilogueOp : MAP_Op<"iter.epilogue", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  ]> {
  let summary = "A store-path callback inlined into an opaque kernel's output.";
  let description = [{
    An SSA callback the enclosing kernel's `map.call` consumes on its `outputs`:
    it is handed a value and an index and stores (or further transforms then
    stores) it. The leading block arguments are the value and the index; any
    remaining block arguments are the captured operands (the tensors the body
    stores into), which are the op's actual operands. The region is isolated
    from above; the `capturing (...)` clause is print/parse sugar over those
    operands.
  }];

  let arguments = (ins Variadic<AnyType>:$captures);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// map.iter.prologue
//===----------------------------------------------------------------------===//

def MAP_IterPrologueOp : MAP_Op<"iter.prologue", [
    IsolatedFromAbove,
    SingleBlock,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  ]> {
  let summary = "A load-path callback inlined into an opaque kernel's input.";
  let description = [{
    An SSA callback the enclosing kernel's `map.call` consumes on its `inputs`:
    it is handed an index and produces the value the kernel should read at
    that index, as if it had loaded the bound tensor directly. The leading
    block argument is the index; any remaining block arguments are the
    captured operands (the tensors the body reads from), which are the op's
    actual operands. The region is isolated from above; the `capturing (...)`
    clause is print/parse sugar over those operands. Terminated by
    `map.yield`.
  }];

  let arguments = (ins Variadic<AnyType>:$captures);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// map.call
//===----------------------------------------------------------------------===//

def MAP_CallOpaqueOp : MAP_Op<"call.opaque", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ]> {
  let summary = "Invoke an opaque Mojo kernel by symbol.";
  let description = [{
    Calls the kernel declared under `callee` (a `mogg.kernel.decl` symbol, e.g.
    `@builtin_kernels.linalg.Matmul`) with the given `inputs` and `outputs`. The
    kernel parameters (e.g. `packed_b`, `transpose_b`) are carried in brackets
    immediately after the callee symbol, e.g.
    `@...Matmul[packed_b = false, transpose_b = false]`.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    DictionaryAttr:$params,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $callee `[` custom<KernelParamsBody>($params) `]`
    `inputs` `(` $inputs `)` `outputs` `(` $outputs `)`
    attr-dict `:` functional-type($inputs, $outputs)
  }];
}

//===----------------------------------------------------------------------===//
// map.iter.foreach
//===----------------------------------------------------------------------===//

def MAP_IterForeachOp : MAP_Op<"iter.foreach", [
  IsolatedFromAbove,
  SingleBlock,
  NoTerminator,
  DeclareOpInterfaceMethods<MO_HasDeviceInterface>,
]> {
  let summary = "Structured iteration over a tensor shape";
  let description = [{
    Iterates over every point in an N-dimensional shape, executing the body
    region once per index. The shape is a `MOSH::ShapeAttr` and may contain
    static integer dims, named symbolic parameters (D0, D1, …), or parametric
    expressions.

    Inputs are op operands that are also shadowed as body block arguments
    (same convention as `mogg.kernel`). Outputs are extra body block arguments
    with no corresponding op operand; their types match the op results. The
    index is a single extra block argument of type `map.index<N>`.

    Block arguments are ordered: [input_args..., output_args..., index_arg].

    Example:
    ```mlir
    %result = map.iter.foreach<[8, 16, 128]>
        inputs  (%x: !mo.tensor<[8, 16, 128], f32>,
                 %y: !mo.tensor<[8, 16, 128], f32>)
        outputs (%out: !mo.tensor<[8, 16, 128], f32>)
        index   (%idx: !map.index<3>)
        -> !mo.tensor<[8, 16, 128], f32>
    { ... }
    ```
  }];

  let arguments = (ins
    MOSH_ShapeAttr:$shape,
    Variadic<MO_Tensor>:$inputs,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo
  );

  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.load
//===----------------------------------------------------------------------===//

def MAP_LoadOp : MAP_Op<"load", [
  TypesMatchWith<"result type matches tensor element type",
                 "tensor", "result",
                 "::llvm::cast<::M::MO::TensorType>($_self).getElementType()">,
  TypesMatchWith<"index type matches tensor rank",
                 "tensor", "index",
                 "::M::MAP::IndexType::get($_self.getContext(), "
                 "::llvm::cast<::M::MO::TensorType>($_self).getRank())">,
  Pure
]> {
  let summary = "Load a value from a tensor at an N-D index";
  let description = [{
    Loads the value at an N-dimensional index from a tensor that is a block
    argument of the enclosing `map.iter.foreach`. Logically a single scalar
    element per index; the actual lowering may load a wider vector or tile
    instead, depending on the target hardware and how the surrounding loop
    is vectorized/tiled. The tensor type is spelled out in the assembly; the
    result type is inferred as the tensor's element type (same convention as
    `memref.load`).

    Example:
    ```mlir
    %xv = map.load %x[%idx] : !mo.tensor<[8, 16, 128], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$tensor, MAP_AnyIndex:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $tensor `[` $index `]` `:` qualified(type($tensor)) attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.store
//===----------------------------------------------------------------------===//

def MAP_StoreOp : MAP_Op<"store", [
  TypesMatchWith<"value type matches tensor element type",
                 "tensor", "value",
                 "::llvm::cast<::M::MO::TensorType>($_self).getElementType()">,
  TypesMatchWith<"index type matches tensor rank",
                 "tensor", "index",
                 "::M::MAP::IndexType::get($_self.getContext(), "
                 "::llvm::cast<::M::MO::TensorType>($_self).getRank())">
]> {
  let summary = "Store a value into a tensor at an N-D index";
  let description = [{
    Stores a value at an N-dimensional index into a tensor that is an output
    block argument of the enclosing `map.iter.foreach`. Logically a single
    scalar element per index; the actual lowering may store a wider vector
    or tile instead, depending on the target hardware and how the
    surrounding loop is vectorized/tiled. The tensor type is spelled out in
    the assembly; the scalar type is inferred from the tensor's element type
    (same convention as `memref.store`).

    Example:
    ```mlir
    map.store %out[%idx] = %outv : !mo.tensor<[8, 16, 128], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$tensor, MAP_AnyIndex:$index, AnyType:$value);

  let assemblyFormat = [{ $tensor `[` $index `]` `=` $value `:` qualified(type($tensor)) attr-dict }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// map.call.elementwise
//===----------------------------------------------------------------------===//

def MAP_CallElementwiseOp : MAP_Op<"call.elementwise", [
  HasParent<"IterForeachOp">,
  Pure,
]> {
  let summary = "Call a named pointwise kernel on the current index and scalar arguments";
  let description = [{
    Calls a named pointwise kernel identified by a flat symbol reference,
    given the enclosing `map.iter.foreach`'s current index and the scalar
    operands, and returns a single scalar result. All operands other than
    the index, and the result, must be scalar (non-tensor) types. Only valid
    inside a `map.iter.foreach` body.

    Example:
    ```mlir
    %outv = map.call.elementwise @builtin_kernels.elementwise.Add(%idx, %xv, %yv)
                : !map.index<3>, f32, f32 -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, MAP_AnyIndex:$index,
                       Variadic<AnyType>:$args);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $callee `(` $index `,` $args `)` attr-dict `:` type($index) `,` type($args) `->` type($result)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD
