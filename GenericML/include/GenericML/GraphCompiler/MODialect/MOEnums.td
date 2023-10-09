//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD

include "GenericML/GraphCompiler/MODialect/MODialect.td"
include "mlir/IR/EnumAttr.td"

class MO_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::MO";
  let genSpecializedAttr = 0;
}

class MO_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

def MO_CoordinateTransformMode : MO_Enum<
  "CoordinateTransformMode", "mo.resize coordinate transform mode enum", [
    I32EnumAttrCase<"HalfPixel", 0, "half_pixel">,
    I32EnumAttrCase<"AlignCorners", 1, "align_corners">,
    I32EnumAttrCase<"Asymmetric", 2, "asymmetric">
  ]
>;

def MO_CoordinateTransformModeAttr : EnumAttr<MODialect, MO_CoordinateTransformMode, "coord_trans_mode">
{
  let cppNamespace = "::M::MO";
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD
