//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD

include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"

// Base class for types in this dialect.
class MOGG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOGGDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
