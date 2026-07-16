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

include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "GraphCompiler/GraphCompiler/MAPDialect/Dialect/MAPTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MAP_Op<string mnemonic, list<Trait> traits = []>
    : Op<MAPDialect, mnemonic, traits>;

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
    (`map.iter.prologue` / `map.iter.epilogue`) plus the `map.call` that invokes
    the kernel; it is not a sequence of computation ops.

    The `inputs` are the kernel's tensor operands. The `outputs` are the kernel
    results; each becomes a block argument of the body so the callbacks and the
    `map.call` can address it, and the op result of the same type carries the
    filled tensor out.

    ```mlir
    %mm = map.iter.opaque inputs (%A, %B) outputs (%out : !mo.tensor<[M, N], bf16>) {
      %epi = map.iter.epilogue (%val : bf16, %idx : !map.index<2>) capturing (%out) {
        map.iter.store %out[%idx] = %val : !mo.tensor<[M, N], bf16>, !map.index<2>, bf16
      } : !mo.tensor<[M, N], bf16>
      map.call @builtin_kernels.linalg.Matmul params = {...}
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

  let hasCustomAssemblyFormat = 1;
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
// map.call
//===----------------------------------------------------------------------===//

def MAP_CallOp : MAP_Op<"call", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ]> {
  let summary = "Invoke an opaque Mojo kernel by symbol.";
  let description = [{
    Calls the kernel declared under `callee` (a `mogg.kernel.decl` symbol, e.g.
    `@builtin_kernels.linalg.Matmul`) with the given `inputs` and `outputs`. The
    `params` dictionary maps kernel parameters (e.g. `packed_b`, `transpose_b`)
    to their values.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    DictionaryAttr:$params,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $callee `params` `=` $params
    `inputs` `(` $inputs `)` `outputs` `(` $outputs `)`
    attr-dict `:` functional-type($inputs, $outputs)
  }];
}

//===----------------------------------------------------------------------===//
// map.iter.store
//===----------------------------------------------------------------------===//

def MAP_IterStoreOp : MAP_Op<"iter.store"> {
  let summary = "Store a value into a tensor at an index.";
  let description = [{
    Stores `value` into `dest` at `index`. Used in a store-path callback
    (`map.iter.epilogue`).
  }];

  let arguments = (ins
    AnyType:$dest,
    AnyType:$index,
    AnyType:$value
  );

  let assemblyFormat = [{
    $dest `[` $index `]` `=` $value attr-dict `:`
    type($dest) `,` type($index) `,` type($value)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPOPS_TD
