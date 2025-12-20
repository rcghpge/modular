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
}

// TODO(GEX-2729): Ideally we'd reuse !mogg.shape but I've chosen not to
// (at the moment) for two reasons:
//
// 1. !mogg.shape is already used in the emit-mojo pipeline to
//    represent DimList values.
//
// 2. It currently doesn't carry around any notion of the rank of its result
//    (which we is information we need during lowering).
//
// In the future we should unify the two and also investigate why both
// IndexList and DimList are used in kernels.
def MOGG_ShapeFuncResult : MOGG_Type<"ShapeFuncResult", "shape_func_result"> {
  let summary = "A type representing the result of a shape function with known rank";
  let description = [{
    This represents the result of a shape function computation with a statically
    known rank. During emit-mojo lowering, this will be lowered to an IndexList
    of the specified rank.

    Examples:
    ```mlir
    !mogg.shape_func_result<3>     // Shape function result with rank 3
    !mogg.shape_func_result<5>     // Shape function result with rank 5
    ```
  }];

  // Keep this simple: concrete integer parameter so syntax is `<3>`.
  let parameters = (ins "uint64_t":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

def MOGG_Context : MOGG_Type<"Context", "context"> {
  let summary = "A type mapping onto a context";
  let description = [{
    This is the mogg representation of a runtime context value.
  }];
}

def MOGG_Error : MOGG_Type<"Error", "error"> {
  let summary = "A type mapping onto a mojo error";
  let description = [{
    This is the mogg representation of the runtime Mojo type `Error`.
    This type is used by throwing functions to return errors.
    Mojo kernels might throw errors that must be propagated to the graph runtime.
  }];
}

//===----------------------------------------------------------------------===//
// Scalar predicates
//===----------------------------------------------------------------------===//

def MOGG_TensorScalarOfAnyRankPred : And<[
  CPred<"llvm::isa<MO::TensorType>($_self)">,
  CPred<"cast<MO::TensorType>($_self).getNumElements().value_or(0) == 1">,
]>;

def MOGG_BufferScalarOfAnyRankPred : And<[
  CPred<"llvm::isa<MO::BufferType>($_self)">,
  CPred<"cast<MO::BufferType>($_self).toTensorType().getNumElements().value_or(0) == 1">,
]>;

def MOGG_ScalarOfAnyRankPred : Or<[MOGG_TensorScalarOfAnyRankPred, MOGG_BufferScalarOfAnyRankPred]>;

def MOGG_ScalarOfAnyRank : ConfinedType<
  AnyTypeOf<[MO_Tensor, MO_Buffer]>,
  [MOGG_ScalarOfAnyRankPred],
  "Has exactly 1 element"
>;

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
