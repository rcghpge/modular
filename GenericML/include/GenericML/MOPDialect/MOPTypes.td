//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPTYPES_TD
#define GENERICML_MOPDIALECT_MOPTYPES_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"

// Base class for other typedefs. Provides dialect-specific defaults.
class MOPType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif  // GENERICML_MOPDIALECT_MOPTYPES_TD