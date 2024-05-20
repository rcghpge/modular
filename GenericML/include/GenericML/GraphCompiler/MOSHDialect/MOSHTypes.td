//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MOSH dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD

include "GenericML/GraphCompiler/MOSHDialect/MOSHDialect.td"
include "KGEN/KGENDialect/KGENTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MOSH_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOSHDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MOSH_ShapeType : MOSH_Type<"Shape", "ape",
  [DeclareTypeInterfaceMethods<ParameterTypeInterface>]>;

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD
