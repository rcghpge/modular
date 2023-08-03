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

include "GenericML/GraphCompiler/MODialect/MODialect.td"
include "KGEN/KGENDialect/KGENAttrs.td"

/// Helps declare optional parameter declaration arguments.
def MO_ParamDecls : DefaultValuedAttr<KGEN_ParamDeclArrayAttr,
                                      "ArrayRef<ParamDeclAttr>({})">;

class MOAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MODialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MO_LayoutAttr
//===----------------------------------------------------------------------===//

def MO_LayoutAttr : MOAttr<"Layout", "layout"> {
  let summary = "An attribute that represents tensor memory layout.";
  let description = [{
    This attribute holds the memory layout information for some tensor value. It
    can be used by graph compiler to
    1. reason about tensor layouts globally in a `mo.graph`
    2. choose the appropriate layout-specific kernels during lowering
  }];

  let parameters = (ins "mlir::StringAttr":$format);

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins "mlir::StringRef" : $formatStr)>,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_TD
