//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD

include "mlir/IR/EnumAttr.td"

def MOGG_BindLambdaKind : I32EnumAttr<"BindLambdaKind", "lambda kind", [
  I32EnumAttrCase<"InputLambda", 0, "input_lambda">,
  I32EnumAttrCase<"OutputLambda", 1, "output_lambda">,
  I32EnumAttrCase<"ElemwiseLambda", 2, "elemwise_lambda">
]> {
  let cppNamespace = "::M::MOGG";
}

def MOGG_ABIKind : I32EnumAttr<"ABIKind", "ABI kind", [
  I32EnumAttrCase<"MOGG", 0, "MOGG">,
  I32EnumAttrCase<"MGP", 1, "MGP">,
]> {
  let cppNamespace = "::M::MOGG";
}

// Role a trait-bound tensor argument plays in the graph signature (the trait
// lane). Mirrors `TensorArgRole` in KernelDeclAdaptor.h and the role traits in
// `extensibility.tensor_arg_traits`. `mut` is not a separate case: an argument
// is mutable iff its role is not `input` (see D2 in the execution plan).
//
// Named `*Kind` to avoid colliding with the existing C++ `M::MOGG`
// classification enums `TensorArgRole` / `TensorArgAccess` in
// KernelDeclAdaptor.h, with which the case spellings are kept in lockstep
// (`tensorArgContractToAttr` maps between them).
def MOGG_TensorArgRole
    : I32EnumAttr<"TensorArgRoleKind", "tensor-argument role", [
        I32EnumAttrCase<"input", 0, "input">,
        I32EnumAttrCase<"output", 1, "output">,
        I32EnumAttrCase<"mutable_input", 2, "mutable_input">,
        I32EnumAttrCase<"none", 3, "none">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

// Element-access form a trait-bound tensor argument exposes -- the fusion
// access kind, or `none` for a plain dense argument. Mirrors
// `TensorArgAccess` in KernelDeclAdaptor.h. Only `none` is exercised in the
// dense slice; the full closed enum is defined for the fused paths (D3).
def MOGG_TensorArgAccess
    : I32EnumAttr<"TensorArgAccessKind", "tensor-argument access form", [
        I32EnumAttrCase<"none", 0, "none">,
        I32EnumAttrCase<"load", 1, "load">,
        I32EnumAttrCase<"store", 2, "store">,
        I32EnumAttrCase<"transform", 3, "transform">,
      ]> {
  let cppNamespace = "::M::MOGG";
}


#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
