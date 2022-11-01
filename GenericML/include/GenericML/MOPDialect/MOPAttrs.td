//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPATTRS_TD
#define GENERICML_MOPDIALECT_MOPATTRS_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Base class for attributes in this dialect.
class MOP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// Currently no attributes.

#endif  // GENERICML_MOPDIALECT_MOPATTRS_TD
