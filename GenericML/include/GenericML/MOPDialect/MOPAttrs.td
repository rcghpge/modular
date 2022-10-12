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

/// An attribute in the Modular dialect.
class MOPAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MOPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif  // GENERICML_MOPDIALECT_MOPATTRS_TD
