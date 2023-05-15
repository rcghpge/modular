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

/// The type used to represent explicit control dependencies.
/// TODO: this is currently in the MOSH dialect because shape functions can use
/// it. We should find a better place for it.
def MOSH_ChainType : MOSH_Type<"Chain", "chain">;

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHTYPES_TD
