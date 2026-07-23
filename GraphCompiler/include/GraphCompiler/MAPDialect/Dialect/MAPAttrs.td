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
include "mlir/IR/AttrTypeBase.td"

// Base class for attributes in this dialect.
class MAP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MAPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// Currently no attributes.

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPATTRS_TD
