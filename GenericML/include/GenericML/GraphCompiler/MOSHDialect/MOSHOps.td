//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines shape operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHOPS_TD

include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class MOSH_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOSHDialect, mnemonic, traits>;

def MOSH_ParamToValueOp : MOSH_Op<"param.to_value", [Pure,
    DeclareOpInterfaceMethods<ParamOpInterface>]>,
    NoParamDecls {
  let summary = "create an SSA value from a parameter value";
  let description = [{
    The `mo.param.to_value` operation materializes the value of a parameter
    expression as an SSA value that may be used by other operations.
    Conceptually, it bridges the parameter value domain to the SSA value domain.

    Example:

    ```mlir
    %idx   = mo.param.to_value = <D0>
    %shape = mo.param.to_value: !mosh.ape = <[10, D1, 20]>
    ```

    Note that in the assembly format, we allow omitting output type annotation
    if it's `index`, for historical reasons.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `` custom<ParamToValueOpValue>($value, type($result)) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value),
              [{ build($_builder, $_state, value.getType(), value); }]>
  ];

  let hasVerifier = 1;
}

def MOShape_BroadcastOp : MOSH_Op<"broadcast", [Commutative, Pure]> {
  let summary = "Shape broadcasting operation";
  let description = [{
    Given two tensor shapes, return the shape that both broadcast to under the
    numpy rules. The semantics of this op are nearly identical to those of
    `mo.broadcast_shape` (see the documentation of that op for details), with
    the only difference that `mosh.broadcast` operates on `!mosh.ape` values
    instead of shape-like tensors.

    Example:

    ```mlir
    %sh1: !mosh.ape
    %sh2: !mosh.ape
    %sh = mosh.broadcast(%sh1, %sh2) : !mosh.ape
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input_x, MOSH_ShapeType:$input_y);
  let results = (outs MOSH_ShapeType:$shape);

  let assemblyFormat = "`(` operands `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOShape_ConcatOp : MOSH_Op<"concat", [Pure]> {
  let summary = "Concatenate a variadic number of shapes.";
  let description = [{
    Return a new shape by concatenating individual dimensions in the input
    shapes in order.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %sh2 : !mosh.ape
    %sh3 = mosh.concat(%sh0, %sh1, %sh2, %sh1)
    ```
  }];

  let arguments = (ins Variadic<MOSH_ShapeType>:$inputs);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $inputs `)` attr-dict";

  let builders = [
    /// Convenience method that works well with `rewriter.create(...)`
    OpBuilder<(ins
      "TypedValue<ShapeType>":$input1, "TypedValue<ShapeType>":$input2)>,
  ];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_EqualOp : MOSH_Op<"eq", [Pure, Commutative]> {
  let summary = "Check if two shapes are equal.";
  let description = [{
    Compare individual dimensions of the input shapes, and return `true` if they
    are all equal, or `false` otherwise.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %sh1 : !mosh.ape
    %b = mosh.equal(%sh0, %sh1)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input1, MOSH_ShapeType:$input2);
  let results = (outs I1:$result);

  let assemblyFormat = "`(` $input1 `,` $input2 `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    /// Return as parameter expression (the two shapes must have equal ranks).
    static TypedAttr asExpression(MOSH::ShapeAttr shape1,
                                  MOSH::ShapeAttr shape2);
  }];
}

def MOShape_GetDimOp : MOSH_Op<"get_dim", [Pure]> {
  let summary = "Operation to extract a dimension from a shape.";
  let description = [{
    Return the size of the specified dimension in the given shape.

    This operation supports dimensions specified by compile time constants only.
    The dimension can be negative in which case Python-like index semantics are
    used (e.g. -2 refers to the second to last dimension).

    Example:

    ```mlir
    %sh : !mosh.ape
    %dim : index
    %firstDim = mosh.get_dim(%sh)[%dim]
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input, Index:$dim);
  let results = (outs Index:$result);

  let assemblyFormat = "`(` $input `)` `` `[` $dim `]` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let extraClassDeclaration = [{
    /// Create a static call to apply(mosh.get_dim, inputShape, axis)
    static TypedAttr emitStaticCall(TypedAttr inputShape, int64_t axis);
  }];
}

def MOSH_GetRankOp : MOSH_Op<"get_rank", [Pure]> {
  let summary = "Operation to get the rank of a shape.";
  let description = [{
    Return the rank of the given shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %r = mosh.get_rank(%sh)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs Index:$result);

  let assemblyFormat = "`(` $input `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_NewOp : MOSH_Op<"new", [Pure]> {
  let summary = "Create a new shape from dimension values.";
  let description = [{
    Return a new shape whose dimensions equal to the given values in order.

    Example:

    ```mlir
    %d0 : index
    %d1 : index
    %sh0 = mosh.new(%d0, %d1)

    %d2 : index
    %sh1 = mosh.new(%d0, %d1, %d2)
    ```
  }];

  let arguments = (ins Variadic<Index>:$dims);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $dims `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOSH_NumElementsOp : MOSH_Op<"num_elements", [Pure]> {
  let summary = "Returns the number of elements in a tensor.";
  let description = [{
    The op computes the product of the dimensions in a shape.

    Example:

    ```mlir
    %sh : !mosh.ape
    %numEl = mosh.num_elements(%sh)
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs Index:$result);

  let assemblyFormat = "`(` $input `)` attr-dict";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def MOShape_SliceOp : MOSH_Op<"slice", [Pure]> {
  let summary = "Slice out a new shape from an existing shape.";
  let description = [{
    The slice supports start and end indices, and they must obey relationships
    that one would expect from Python-like slices. Note that if `end` is none,
    we emulate the pythonic `shape[start:]` semantics.

    Example:

    ```mlir
    %sh0 : !mosh.ape
    %zero = mosh.param.to_value = <0>
    %two = mosh.param.to_value = <-2>
    %negOne = mosh.param.to_value = <-2>
    %negTwo = mosh.param.to_value = <-2>

    %sh1 = mosh.slice(%sh0)[%zero, %two]      // first two dims
    %sh2 = mosh.slice(%sh0)[%negTwo]          // last two dims
    %sh3 = mosh.slice(%sh0)[%zero, %negTwo]   // all but last two dims
    %sh4 = mosh.slice(%sh0)[%negTwo, %negOne] // second to last dim
    ```
  }];

  let arguments = (ins
    MOSH_ShapeType:$input,
    Index:$start,
    Index:$end
  );
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $input `)` `` `[` $start `,` $end `]` attr-dict";

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOSHAPEOPS_TD
