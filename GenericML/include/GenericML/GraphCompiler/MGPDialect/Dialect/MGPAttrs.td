//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPATTRS_TD
#define GENERICML_MGPDIALECT_MGPATTRS_TD

include "GenericML/GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

// Base class for attributes in this dialect.
class MGP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MGPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// Currently no attributes.

#endif  // GENERICML_MGPDIALECT_MGPATTRS_TD
