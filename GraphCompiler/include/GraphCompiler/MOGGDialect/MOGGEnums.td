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

// How a run of variadic tensor operands packed into a `mogg.tuple.create`
// must be materialized in Mojo. `ConformCallOperands` sets this from the
// callee's variadic parameter type, so `emitMojo` chooses the aggregate
// (`VariadicTensors` / `_FusedInputVariadicTensors` /
// `_FusedOutputVariadicTensors`, or a `TensorTuple` of trait arguments)
// without inspecting the producing ops.
def MOGG_VariadicABI
    : I32EnumAttr<"VariadicABI", "variadic tensor packing ABI", [
  I32EnumAttrCase<"Plain", 0, "plain">,
  I32EnumAttrCase<"FusedInput", 1, "fused_input">,
  I32EnumAttrCase<"FusedOutput", 2, "fused_output">,
  // A `TensorTuple` (`xs: TensorTuple[*Ts]`): the elements are already trait
  // arguments, and emit builds a `TensorTuple` holding them.
  I32EnumAttrCase<"TensorTuple", 3, "tensor_tuple">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_VariadicABIAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

def MOGG_ABIKind : I32EnumAttr<"ABIKind", "ABI kind", [
  I32EnumAttrCase<"MGP", 1, "MGP">,
  // A `mogg.kernel.stub` outlined from a `mogg.kernel` (advanced fusion), so
  // `EmitMojo` can branch on it separately from MGP emission.
  I32EnumAttrCase<"MOGG", 2, "MOGG">,
]> {
  let cppNamespace = "::M::MOGG";
  // Use `MOGG_ABIKindAttr` (MOGGAttrs.td) rather than a bare `i32`.
  let genSpecializedAttr = 0;
}

// Role a trait-bound tensor argument plays in the graph signature. Kept in
// lockstep with `TensorArgRole` in KernelDeclAdaptor.h (named `*Kind` to avoid
// colliding with it; `tensorArgContractToAttr` maps between them) and the role
// traits in `extensibility.tensor_arg_traits`. An argument is mutable iff its
// role is not `input`. `none` is for shape-function tensor arguments, which
// read only metadata.
def MOGG_TensorArgRole
    : I32EnumAttr<"TensorArgRoleKind", "tensor-argument role", [
        I32EnumAttrCase<"input", 0, "input">,
        I32EnumAttrCase<"output", 1, "output">,
        I32EnumAttrCase<"mutable_input", 2, "mutable_input">,
        I32EnumAttrCase<"none", 3, "none">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

// Set of element-access forms a trait-bound tensor argument exposes (`none`
// for a plain dense argument). Mirrors the access/capability bitmasks of
// `TensorArgContract` in KernelDeclAdaptor.h. A non-fused dense argument can
// carry several (e.g. `load | store` for read-modify-write); a fused argument
// carries exactly one.
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

// The deepest trait bound written on a shape-function tensor argument (see
// `TensorArgBound` in KernelDeclAdaptor.h). Execute-lane contracts always
// carry the default `unspecified`, so printing and the baked
// `tensor_arg_contracts` dict elide it.
def MOGG_TensorArgBound
    : I32EnumAttr<"TensorArgBoundKind", "tensor-argument written bound", [
        I32EnumAttrCase<"unspecified", 0, "unspecified">,
        I32EnumAttrCase<"tensor", 1, "tensor">,
        I32EnumAttrCase<"dense", 2, "dense">,
        I32EnumAttrCase<"tile", 3, "tile">,
      ]> {
  let cppNamespace = "::M::MOGG";
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
