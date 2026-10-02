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

include "GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/MODialect/MOTypes.td"

class MOGG_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOGGDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ProfileScopeRange
//===----------------------------------------------------------------------===//

def MOGG_ProfileScopeRange : MOGG_Type<"ProfileScopeRange", "profile_scope_range"> {
  let summary = "An opaque handle correlating a profile_scope.begin with its .end";
  let description = [{
    Represents the `std.runtime.tracing.Trace` variable in the emitted Mojo.
    Begin and end are correlated through SSA def-use, so ancestor and nested
    ranges can be open simultaneously without stack semantics.
  }];
}

//===----------------------------------------------------------------------===//
// MOGG basic kernel types.
//===----------------------------------------------------------------------===//

def MOGG_Shape : MOGG_Type<"Shape", "shape"> {
  let summary = "A type mapping onto a shape";
  let description = [{
    A runtime shape value.
  }];
}

// TODO(GEX-2729): Reuse !mogg.shape once it carries a rank; it already
// represents IndexList values in emit-mojo.
def MOGG_ShapeFuncResult : MOGG_Type<"ShapeFuncResult", "shape_func_result"> {
  let summary = "A type representing the result of a shape function with known rank";
  let description = [{
    Lowers to an IndexList of the given rank in emit-mojo.

    Example:
    ```mlir
    !mogg.shape_func_result<3>
    ```
  }];

  let parameters = (ins "uint64_t":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

def MOGG_Context : MOGG_Type<"Context", "context"> {
  let summary = "A type mapping onto a context";
  let description = [{
    A runtime context value.
  }];
}

def MOGG_TaskHandle : MOGG_Type<"TaskHandle", "task_handle"> {
  let summary = "Handle to an asynchronously dispatched task";
  let description = [{
    A Mojo task from `create_task`, emitted as a `(Task, Optional[Error])`
    pair. Produced by `mogg.call.stub.async`, consumed by `mogg.async.join`.
  }];
}

def MOGG_Error : MOGG_Type<"Error", "error"> {
  let summary = "A type mapping onto a mojo error";
  let description = [{
    The Mojo `Error` type, returned by throwing kernels so errors propagate to
    the graph runtime.
  }];
}

def MOGG_Async : MOGG_Type<"Async", "async"> {
  let summary = "An asynchronous wrapper around a value type";
  let description = [{
    Models an asynchronously computed value. `mogg.generic.compiled.region`
    body arguments are `!mogg.async<T>` because they are handles that must be
    unpacked (`mogg.async.unpack`) before use; `mogg.async.pack` is the
    inverse.

    Examples:
    ```mlir
    !mogg.async<!mgp.buffer>
    !mogg.async<!mef.chain>
    ```
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
}

// A `!mogg.async<T>` with `T` in `allowedTypes`. `ConfinedType` keeps the
// accessor typed as `TypedValue<MOGG::AsyncType>`.
class MOGG_AsyncOf<list<Type> allowedTypes> : ConfinedType<MOGG_Async,
  [Or<!foreach(allowedType, allowedTypes,
        SubstLeaves<"$_self",
                    "llvm::cast<MOGG::AsyncType>($_self).getElementType()",
                    allowedType.predicate>)>],
  "an !mogg.async of " #
    !interleave(!foreach(allowedType, allowedTypes, allowedType.summary),
                " or ")>;

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

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen types
//
// Opaque handles for the MAP -> MOGG -> MGP path: the creating op carries
// shape, dtype, and io-kind, not the type.
//===----------------------------------------------------------------------===//

def MOGG_Pointer : MOGG_Type<"Pointer", "pointer"> {
  let summary = "An opaque pointer to a tensor's backing data.";
  let description = [{
    Kernels take pointers plus dimensions rather than tensor objects; the body
    reconstructs the tensor with `mogg.tensor.create`.
  }];
}

def MOGG_Tensor : MOGG_Type<"Tensor", "tensor",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "An opaque codegen tensor handle.";
  let description = [{
    Shape, element type and io-kind live on the creating op. Mutability is in
    the type (`!mogg.tensor` vs `!mogg.buffer`) so `getKernelOperandKind()` can
    tell `Tensor` from `Buffer` without a defining op. Scalar-match rank/host
    validation is skipped (already validated upstream).
  }];
}

def MOGG_Buffer : MOGG_Type<"Buffer", "buffer",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "An opaque codegen mutable tensor (buffer) handle.";
  let description = [{
    The mutable counterpart of `!mogg.tensor`, for a `!mo.buffer` kernel
    operand (e.g. a collective's signal buffers). Requires `mut = true` on the
    creating op. Reports as `Buffer`, so operand inference matches it to a
    `MutableInput` argument without inspecting the creating op.
  }];
}

// Use wherever a `!mo.buffer` kernel operand may flow.
def MOGG_TensorLike : AnyTypeOf<[MOGG_Tensor, MOGG_Buffer]>;

def MOGG_Dim : MOGG_Type<"Dim", "dim"> {
  let summary = "An opaque runtime dimension value.";
  let description = [{
    Optionally carries the source dynamic-dimension parameter, printed as
    `!mogg.dim<batch>`. `MOGGToMGP` drops it, since a symbolic dimension would
    prevent kernel deduplication.
  }];
  let parameters = (ins OptionalParameter<"::M::KGEN::ParamDeclRefAttr">:$name);
  let assemblyFormat = "(`<` custom<DimName>($name)^ `>`)?";
}

def MOGG_Scalar : MOGG_Type<"Scalar", "scalar",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "An opaque codegen scalar handle.";
  let description = [{
    Produced by `mogg.scalar.create` from a `!mogg.tensor`. The creating op
    carries the element type.
  }];
}

def MOGG_Tuple : MOGG_Type<"Tuple", "tuple"> {
  let summary = "An opaque codegen tuple handle (a packed variadic).";
  let description = [{
    Variadic tensor operands packed by `mogg.tuple.create`, whose operands are
    the elements.
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
