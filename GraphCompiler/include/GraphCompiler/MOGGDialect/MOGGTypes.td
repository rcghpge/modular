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
//    represent IndexList values.
//
// 2. It currently doesn't carry around any notion of the rank of its result
//    (which we is information we need during lowering).
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

def MOGG_TaskHandle : MOGG_Type<"TaskHandle", "task_handle"> {
  let summary = "Handle to an asynchronously dispatched task";
  let description = [{
    Represents a Mojo-native async task dispatched via `create_task`.
    In emitted Mojo code, this maps to a `(Task, Optional[Error])` pair.
    Used by `mogg.call.stub.async` (produces) and `mogg.async.join` (consumes).
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

def MOGG_Async : MOGG_Type<"Async", "async"> {
  let summary = "An asynchronous wrapper around a value type";
  let description = [{
    This type wraps an arbitrary MLIR type to model an asynchronously computed
    value. Inside a `mogg.generic.compiled.region`, the body block arguments are
    `!mogg.async<T>` rather than `T` to reflect that they are not values but
    handles/pointers that must be unpacked before use. `mogg.async.unpack`
    projects an `!mogg.async<T>` to a `T`, and `mogg.async.pack` injects a `T`
    into an `!mogg.async<T>`.

    Examples:
    ```mlir
    !mogg.async<!mgp.buffer>
    !mogg.async<!mef.chain>
    ```
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    // Builds an `!mogg.async<T>` without an explicit context, reusing the
    // context of the wrapped element type.
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>,
  ];
}

// A `!mogg.async<T>` whose wrapped element type `T` is one of `allowedTypes`.
// Built on `MOGG_Async` via `ConfinedType` so the generated C++ accessor is a
// `TypedValue<MOGG::AsyncType>` (the base type's `cppType` is preserved); the
// `MOGG_Async` predicate already guarantees the value is an async type.
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
// The types below back the new MAP -> MOGG -> MGP lowering path. Everything
// ABOVE remains the legacy type set and is removed once advanced fusion
// (MAX_GC_USE_ADV_FUSION) is the default, so these are appended here for a clean
// eventual cut. They are opaque handles by design: the op that creates a value
// carries its shape / dtype / mutability / io-kind, not the type.
//===----------------------------------------------------------------------===//

def MOGG_Pointer : MOGG_Type<"Pointer", "pointer_"> {
  let summary = "An opaque pointer to a tensor's backing data.";
  let description = [{
    Codegen-dialect handle for a raw data pointer. Kernels in the new lowering
    path take pointers (plus dimensions) rather than rich tensor objects; the
    tensor is reconstructed inside the body via `mogg._tensor.create`.
  }];
}

def MOGG_Tensor : MOGG_Type<"Tensor", "tensor_",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "An opaque codegen tensor handle.";
  let description = [{
    A tensor value in the codegen dialect. Shape, element type, mutability and
    io-kind are carried by the op that creates it (`mogg._tensor.create` /
    `mogg._tensor.create.epilogue`), not by this type. Implements
    `MOKernelOperandTypeInterface` (as `Tensor`) so operand inference can run on
    a `mogg._call.opaque` — the rank/host validation of a scalar match is skipped
    for this opaque handle (already validated upstream).
  }];
}

def MOGG_Dim : MOGG_Type<"Dim", "dim_"> {
  let summary = "An opaque runtime dimension value.";
  let description = [{
    A single runtime dimension in the codegen dialect. Optionally carries the
    source dynamic-dimension parameter (a `#kgen.param.decl.ref`), printed bare
    as `!mogg.dim_<batch>`. `MOGGToMGP` drops the name — a symbolic dimension
    would prevent kernel deduplication, so dims become SSA value handles there.
  }];
  let parameters = (ins OptionalParameter<"::M::KGEN::ParamDeclRefAttr">:$name);
  let assemblyFormat = "(`<` custom<DimName>($name)^ `>`)?";
}

def MOGG_Scalar : MOGG_Type<"Scalar", "scalar_",
  [DeclareTypeInterfaceMethods<MO_KernelOperandTypeInterface>]> {
  let summary = "An opaque codegen scalar handle.";
  let description = [{
    A scalar kernel operand in the codegen dialect, produced by
    `mogg._scalar.create` from a `!mogg.tensor_` (the new-path counterpart of
    `mogg.as_scalar`). Element type is carried by the creating op, not the type.
  }];
}

def MOGG_Tuple : MOGG_Type<"Tuple", "tuple_"> {
  let summary = "An opaque codegen tuple handle (a packed variadic).";
  let description = [{
    A packed run of variadic tensor operands, produced by `mogg._tuple.create`
    (the new-path counterpart of `mogg.as_tuple`). The packed elements are the
    operands of the creating op.
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGTYPES_TD
