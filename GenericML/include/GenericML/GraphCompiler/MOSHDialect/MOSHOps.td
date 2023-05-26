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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class MOSH_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOSHDialect, mnemonic, traits>;

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

  let extraClassDeclaration = [{
    /// Returns the forward shape function (building it in the given module if
    /// necessary) assuming both shapes are statically ranked, and the smaller
    /// rank is provided.
    static KGEN::FuncOp getForwardShapeFunc(Location loc, ModuleOp module,
      size_t smallerRank, mlir::RewriterBase &rewriter);
  }];
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
    %firstDim = mosh.get_dim(%sh)[0]
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input, AttrOfType<Index>:$dim);
  let results = (outs Index:$result);

  let builders = [
    OpBuilder<(ins "Value":$input, "int64_t":$dim)>,
  ];

  let assemblyFormat = [{
    `(` $input `)` `` `[` custom<IndexParamValue>($dim) `]` attr-dict
  }];
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Create a static call to apply(mosh.get_dim[axis], inputShape)
    static TypedAttr emitStaticCall(TypedAttr inputShape, int64_t axis);
    /// Helper to get the value of the dim attribute.
    std::optional<APInt> getDimValue();
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
    %zero = index.constant 0
    %two = index.constant -2
    %negOne = index.constant -2
    %negTwo = index.constant -2

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
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def MOSH_AssertOp : MOSH_Op<"assert"> {
  let summary = "Assert that a single boolean value is true.";
  let description = [{
    This op is special in the MOSH dialect since it does not take or return
    shapes. The argument is single index value and a !mosh.chain is returned.

    TODO: this is currently in the MOSH dialect because shape functions can use
    it, but we should find a proper place for it.

    Example:

    ```mlir
    %0 : index
    %1 : index
    %b1 = index.cmp eq(%0, %1)
    %ch1 = mosh.assert %b1
    %b2 = index.cmp lt(%0, %1)
    %ch2 = mosh.assert %b2, "something went wrong"
    ```
  }];

  let arguments = (ins I1:$cond, AttrOfType<KGEN_StringType>:$message);
  let results = (outs MOSH_ChainType:$chain);

  let builders = [
    OpBuilder<(ins "Value":$cond, "Twine":$message)>,
  ];

  let assemblyFormat = "$cond `,` custom<StringParam>($message) attr-dict";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Create a static call to apply(mosh.assert, cond, message)
    static TypedAttr emitStaticCall(TypedAttr cond, StringRef message);
  }];
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

def MOSH_GuardOp : MOSH_Op<"guard"> {
  let summary = "Signify control dependencies of the operands.";
  let description = [{
    Sometimes (although not often) the validity of values may depend on things
    that we can't (or don't want to) express in the dataflow. This op allows one
    to express explicit control dependencies, that later lowerings must convert
    to correct dependencies at runtime.

    TODO: this is currently in the MOSH dialect because shape functions can use
    it, but we should find a proper place for it.

    Example:

    ```mlir
    %sh0: !mosh.ape
    %cond0: i1
    %ch0 = mosh.assert %cond0
    mosh.guard[%ch0](%sh0)

    %sh1: !mosh.ape
    %cond1: i1
    %ch1 = mosh.assert %cond1
    mosh.guard[%ch1](%sh0, %sh1)
    ```
  }];

  let arguments = (ins MOSH_ChainType:$chain, Variadic<AnyType>:$values);
  let assemblyFormat = [{
    `[` $chain `]` ($values^ `:` type($values))? attr-dict
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOSHAPEOPS_TD
