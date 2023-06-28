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

//===----------------------------------------------------------------------===//
// MOGG basic kernel types.
//===----------------------------------------------------------------------===//

def MOGG_Simd : MOGG_Type<"Simd", "simd"> {
  let summary = "Represents a SIMD value, length will be determined by elaboration";
  let description = [{
    This represents an abstract virtual SIMD type. The size will be fixed based on the type and
    resolved during the KGEN elaboration.
  }];

  let parameters = (ins "DType" : $dtype);
  let assemblyFormat = [{
    `<` $dtype `>`
  }];
}

def MOGG_Indices : MOGG_Type<"Indices", "indices"> {
  let summary = "A type representing the indices into a tensor";
  let description = [{
    This is an abstraction above the Mojo indices type which is still
    somewhat TBD. Whatever that type is in mojo this represents it at the
    MOGG level. It is used as inputs/outputs of certain lambdas which then
    use these to access tensor data or return modified indices.
  }];
}

def MOGG_Shape : MOGG_Type<"Shape", "shape"> {
  let summary = "A type mapping onto a shape";
  let description = [{
    This is the mogg representation of a runtime shape value.
  }];

  let parameters = (ins "TypedAttr" : $shapeAttr);
  let assemblyFormat = [{
    `<` custom<ShapeParam>($shapeAttr) `>`
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
