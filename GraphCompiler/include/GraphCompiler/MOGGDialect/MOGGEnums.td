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
  // Advanced-fusion path: a `mogg.kernel.stub` outlined from a `mogg._kernel`
  // (see OutlineKernelsIntoFunctions). Labeled here so `EmitMojo` can branch on
  // it without touching the legacy MOGG/MGP ABI emission.
  I32EnumAttrCase<"MAP", 2, "MAP">,
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
// `none` is the no-role flavor carried by a shape-function tensor argument: it
// reads only metadata and is always read-only, so it has no io-role. Kept in
// lockstep with `TensorArgRole::none` in KernelDeclAdaptor.h.
def MOGG_TensorArgRole
    : I32EnumAttr<"TensorArgRoleKind", "tensor-argument role", [
        I32EnumAttrCase<"input", 0, "input">,
        I32EnumAttrCase<"output", 1, "output">,
        I32EnumAttrCase<"mutable_input", 2, "mutable_input">,
        I32EnumAttrCase<"none", 3, "none">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

// Element-access forms a trait-bound tensor argument exposes -- a SET of fusion
// access kinds, or the empty set `none` for a plain dense argument with no
// element access. Mirrors the access/capability bitmasks of `TensorArgContract`
// in KernelDeclAdaptor.h. A bit enum so a non-fused dense argument can carry
// several forms at once (e.g. `load | store` for in-place read-modify-write);
// a fused argument carries exactly one. The same bit-enum type is reused for
// the contract's capability set.
def MOGG_TensorArgAccess
    : I32BitEnumAttr<"TensorArgAccessKind", "tensor-argument access form set", [
        I32BitEnumAttrCaseNone<"none">,
        I32BitEnumAttrCaseBit<"load", 0, "load">,
        I32BitEnumAttrCaseBit<"store", 1, "store">,
        I32BitEnumAttrCaseBit<"transform", 2, "transform">,
      ]> {
  let cppNamespace = "::M::MOGG";
  let genSpecializedAttr = 0;
}


#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
