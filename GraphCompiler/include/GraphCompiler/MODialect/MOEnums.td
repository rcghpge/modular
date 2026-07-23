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

include "GraphCompiler/MODialect/MODialect.td"
include "mlir/IR/EnumAttr.td"

class MO_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::MO";
  let genSpecializedAttr = 0;
}

class MO_EnumCase<string sym, int val, string str = sym> :
  I32EnumAttrCase<sym, val, str>;

def MO_CoordinateTransformMode : MO_Enum<
  "CoordinateTransformMode", "Valid coordinate transformations for `mo.resize`", [
    I32EnumAttrCase<"HalfPixel", 0, "half_pixel">,
    I32EnumAttrCase<"AlignCorners", 1, "align_corners">,
    I32EnumAttrCase<"Asymmetric", 2, "asymmetric">,
    I32EnumAttrCase<"HalfPixel1D", 3, "half_pixel_1D">
  ]
>;

def MO_CoordinateTransformModeAttr : EnumAttr<MODialect, MO_CoordinateTransformMode, "coord_trans_mode">
{
  let summary = "coordinate transformations";
  let description = [{
    This attribute is used by `mo.resize`.
  }];

  let cppNamespace = "::M::MO";
}

def MO_IOKind : I32EnumAttr<"IOKind", "IO type of a mojo kernel operand", [
  I32EnumAttrCase<"Unknown", 32, "Unknown">, // Can't do -1.
  I32EnumAttrCase<"Output", 0, "Output">,
  I32EnumAttrCase<"Input", 1, "Input">,
  I32EnumAttrCase<"FusedInput", 2, "FusedInput">,
  I32EnumAttrCase<"FusedOutput", 3, "FusedOutput">,
  I32EnumAttrCase<"FusedComputeOutput", 31, "FusedComputeOutput">
]> {
  let cppNamespace = "::M::MO";
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOENUMS_TD
