//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MAP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MAPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MAP_Index : MAP_Type<"Index", "index"> {
  let summary = "A statically-ranked index into an iteration space.";
  let description = [{
    Represents an index into a `map.iter` kernel's iteration space, carrying its
    static rank. This is the `imap` (index-space) index type; per the MAP naming
    convention it lives in the `map` dialect.

    Example:
    ```mlir
    !map.index<2>    // a rank-2 index
    ```
  }];

  let parameters = (ins "uint64_t":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD
