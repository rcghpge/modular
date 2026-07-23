//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD

include "GraphCompiler/RMODialect/RMODialect.td"
include "Support/Compiler/AttrBase.td"
include "mlir/IR/AttrTypeBase.td"

class RMOAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RMODialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOATTRS_TD
