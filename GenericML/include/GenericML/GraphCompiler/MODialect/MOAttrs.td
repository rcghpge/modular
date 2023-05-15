//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD

include "KGEN/KGENDialect/KGENAttrs.td"

/// Helps declare optional parameter declaration arguments.
def MO_ParamDecls : DefaultValuedAttr<KGEN_ParamDeclArrayAttr,
                                      "ArrayRef<ParamDeclAttr>({})">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
