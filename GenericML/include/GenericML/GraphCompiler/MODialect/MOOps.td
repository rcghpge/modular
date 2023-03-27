//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
  : Op<MODialect, mnemonic, traits>;

// Base class for ops that work with tensors in the MO dialect.
class MO_TensorOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [MO_TensorOpInterface] # traits>;

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def MO_AssertOp : MO_Op<"assert"> {
  let summary = "Assert that a single boolean value is true.";
  let description = [{
    This op is special in the MO dialect since it does not take or return MO
    tensors. The argument is single index value and a !mo.chain is returned.

    Example:

    ```mlir
    %0 : index
    %1 : index
    %b1 = index.cmp eq(%0, %1)
    %ch1 = mo.assert %b1
    %b2 = index.cmp lt(%0, %1)
    %ch2 = mo.assert %b2, "something went wrong"
    ```
  }];

  let arguments = (ins I1:$cond, OptionalAttr<StrAttr>:$message);
  let results = (outs MO_ChainType:$chain);

  let builders = [
    OpBuilder<(ins "Value":$cond)>,
    OpBuilder<(ins "Value":$cond, "StringRef":$message)>,
  ];

  let assemblyFormat = "$cond (`,` $message^)? attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

def MO_GuardOp : MO_Op<"guard"> {
  let summary = "Signify control dependencies of the operands.";
  let description = [{
    Sometimes (although not often) the validity of values may depend on things
    that we can't (or don't want to) express in the dataflow. This op allows one
    to express explicit control dependencies, that later lowerings must convert
    to correct dependencies at runtime.
    %sh0: !mo.shape
    %cond0: i1
    %ch0 = mo.assert %cond0
    mo.guard[%ch0](%sh0)

    %sh1: !mo.shape
    %cond1: i1
    %ch1 = mo.assert %cond1
    mo.guard[%ch1](%sh0, %sh1)
    ```
  }];

  let arguments = (ins MO_ChainType:$chain, Variadic<AnyType>:$values);
  let assemblyFormat = [{
    `[` $chain `]` ($values^ `:` type($values))? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_TensorOp<"graph",
  [HasParent<"ModuleOp">, FunctionOpInterface, Symbol, SymbolTable,
   IsolatedFromAbove, SingleBlockImplicitTerminator<"OutputOp">,
  DeclareOpInterfaceMethods<MO_Parameterization>,
  // needed to leverage KGEN parameter infra
  DeclInterface, FuncInterface
  // TODO : Re-enable TransformableGraphInterface
  // DeclareOpInterfaceMethods<TransformableGraphInterface>
]> {
  let summary = "Function that contains a MO graph with other ops mixed in";
  let arguments = (ins TypeAttrOf<KGEN_SignatureType>:$signature,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::front;
    using FunctionOpInterfaceTrait::end;
    Block *getBodyBlock() { return &getBody().front(); }
    OutputOp getOutputOp();

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// We aren't using always_inline functionality in MO.
    KGEN::AlwaysInlineLevel getAlwaysInlineLevel() {
      return KGEN::AlwaysInlineLevel::Disabled;
    }
  }];

  let builders = [
    /// Create a new mo.graph with an empty body.
    OpBuilder<(ins
      "StringRef":$name,
      "TypeRange":$inputTypes,
      "TypeRange":$resultTypes
    )>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MO_OutputOp : MO_TensorOp<"output",
    [Terminator, ReturnLike, HasParent<"MO::GraphOp">]> {
  let summary = "MO output operation";
  let description = [{
    The "mo.output" operation specifies the output parameters and values for a
    MO function. The operation takes variable number of operands and produces
    no results. The operand number and types must match the signature of
    the mo.graph that contains the operation (after substituting the bindings to
    signature output type). For example:

    Example:

    ```mlir
    mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
      mo.output %arg0 : !mo.tensor<?, f32>
    }

    mo.graph @with_params<D1 -> D2>(
        %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
      mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
    }
    ```

    Note that in the parameterized example, the output operands type annotation
    don't match the graph's output type textually, but they match after
    substituting the bindings into graph's output type.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
  );

  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : MO_TensorOp<"if",
          [SingleBlockImplicitTerminator<"MO::YieldOp">,
          RecursiveMemoryEffects, RecursivelySpeculatable]> {
  let summary = "if operation";
  let description = [{
    The "mo.if" operation takes an i1 condition, a 'then' body and an 'else'
    body. If the condition is true, the 'then' body is run and the "mo.if"
    operation returns the results of that region, otherwise the "else" body is
    run and its values are returned.

    The body regions must have the same signature, but they may take an
    arbitrary number of arguments and return values with the same types as the
    op return types, using a 'mo.yield' operation at the end of the
    region.

    Example:

    ```mlir
      %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
        %v1 = mo.add(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                               ) -> !mo.tensor<?, f32>
        mo.yield %v1 : !mo.tensor<?, f32>
      } else {
        %v2 = mo.sub(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                 ) -> !mo.tensor<?, f32>
        mo.yield %v2 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins MO_CondLike:$cond);
  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    `(` $cond `:` qualified(type($cond)) `)` (`->` type($results)^)?
    $thenRegion `else` $elseRegion attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def MO_YieldOp
    : MO_TensorOp<"yield", [Terminator, ReturnLike, Pure, HasParent<"IfOp">]> {
  let summary = "MO yield operation";
  let description = [{
    The "mo.yield" operation specifies the output values for MO blocks.
    The operation takes variable number of operands and produces
    no results.

    Example:

    ```mlir
    mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
      mo.yield %arg0 : !mo.tensor<?, f32>
    } else {
      mo.yield %arg1 : !mo.tensor<?, f32>
    }
    ```
  }];

  let arguments = (ins Variadic<MO_Tensor>:$operands);

  let assemblyFormat = [{
    `` attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def MO_RebindOp : MO_TensorOp<"rebind", [
    DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let summary = "tensor parameter rebind op";
  let description = [{
    Op to represent "rebinding" a value of `!mo.tensor<...>` type to another
    `!mo.tensor<...>`. Currently, this op can be ued to convert all attributes
    of `!mo.tensor` types, inlcuding shape, element type, and metadata.

    Examples:

    ```mlir
    %0 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, 1], f32>
    %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
    %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[], f32>
    %3 = mo.rebind<() -> K, M>(%0) :
      !mo.tensor<[N, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_TensorOp<"constant", [ConstantLike, Pure]> {
  let summary = "constant MO operation";
  let description = [{
    Op to store constant tensors in model.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>
    } : !mo.tensor<[2, 2], si32>
    ```
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[M_ArrayElementsAttr,
                  Builtin_DenseResourceElementsAttr,
                  M_AlignedBytesAttr]>]
    >:$value);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;

  let builders = [
    /// Create a new rank-1 tensor mo.constant of int32 value.
    OpBuilder<(ins "ArrayRef<int32_t>":$values)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int32 value.
    OpBuilder<(ins "int32_t":$value)>,
    /// Create a new rank-1 tensor mo.constant of int64 value.
    OpBuilder<(ins "ArrayRef<int64_t>":$values)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int64 value.
    OpBuilder<(ins "int64_t":$value)>,
    /// Create a new rank-1 tensor mo.constant of any float type.
    OpBuilder<(ins "ArrayRef<APFloat>":$values, "Type":$elementType)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of any float type.
    OpBuilder<(ins "APFloat":$value, "Type":$elementType )>,
  ];

}

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

class MO_ElementWiseOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure, MO_SameOperandsAndResultType]> {
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

class MO_ElementWiseUnaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseBinaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ComparisonOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure, MO_SameOperandsElementType,
                            MO_SameOperandsAndResultShape]> {
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class MO_ComparisonBinaryOp<string name, list<Trait> traits = []>
    : MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

def MO_AddOp : MO_ElementWiseBinaryOp<"add", [Commutative]> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasFolder = 1;
}

def MO_AbsOp : MO_ElementWiseUnaryOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_DivOp : MO_ElementWiseBinaryOp<"div"> {
  let summary = "Divide operation";
  let description = [{
    Returns x / y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ExpOp : MO_ElementWiseUnaryOp<"exp"> {
  let summary = "Exponential function";
  let description = [{
    Returns exp(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ErfOp : MO_ElementWiseUnaryOp<"erf"> {
  let summary = "Absolute value operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_GeluOp : MO_ElementWiseUnaryOp<"gelu"> {
  let summary = "Gaussian Error Linear Unit operation";
  let description = [{
    Returns GELU(x) = (x / 2) * (1 + erf(x/sqrt(2))), where erf denotes the
    (Gaussian) error function.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SigmoidOp : MO_ElementWiseUnaryOp<"sigmoid"> {
  let summary = "Logistic Sigmoid operation";
  let description = [{
    Returns $sigmoid(x) = \frac{1}{1+e^{-x}} = \fract{e^x}{e^x + 1} to compute
    the logistic sigmoid operations.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MaxOp : MO_ElementWiseBinaryOp<"max", [Commutative]> {
  let summary = "Maximum operation";
  let description = [{
    Returns max(x, y), where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MinOp : MO_ElementWiseBinaryOp<"min", [Commutative]> {
  let summary = "Minimum operation";
  let description = [{
    Returns min(x, y), where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MulOp : MO_ElementWiseBinaryOp<"mul", [Commutative]> {
  let summary = "Multiply operation";
  let description = [{
    Returns x * y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_PowOp : MO_TensorOp<"pow", [Pure, MO_SameOperandsAndResultShape]> {
  let summary = "Power operation";
  let description = [{
    Computes elementwise x ** y, where x is the base tensor and y is the
    exponent tensor.

    Example:

    ```mlir
    %x: !mo.tensor<[2, 3], f32>
    %y: !mo.tensor<[2, 3], f32>
    %res = mo.pow(
        %x: !mo.tensor<[2, 3], f32>,
        %y: !mo.tensor<[2, 3], f32>
    ) : !mo.tensor<[2, 3], f32>

    %x: !mo.tensor<[2, 3], f32>
    %y_int: !mo.tensor<[2, 3], si32>
    %res = mo.pow(
        %x: !mo.tensor<[2, 3], f32>,
        %y: !mo.tensor<[2, 3], si32>
    ) : !mo.tensor<[2, 3], f32>
    %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input_x `:` qualified(type($input_x))
    `,` $input_y `:` qualified(type($input_y)) `)` attr-dict
    `:` qualified(type($result))
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RsqrtOp : MO_ElementWiseUnaryOp<"rsqrt"> {
  let summary = "Inverse Square Root operation";
  let description = [{
    Returns 1/sqrt(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SqrtOp : MO_ElementWiseUnaryOp<"sqrt"> {
  let summary = "Square Root operation";
  let description = [{
    Return sqrt(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_CastOp : MO_TensorOp<"cast", [Pure, MO_SameOperandsAndResultShape]> {
  let summary = "Cast operation";
  let description = [{
    Return cast[dtype](x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], i32>
    %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
}

def MO_EqualOp : MO_ComparisonBinaryOp<"equal", [Commutative]> {
  let summary = "Equal operation";
  let description = [{
    Returns elementwise x == y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                   !mo.tensor<[2, 3], f32>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterOp : MO_ComparisonBinaryOp<"greater"> {
  let summary = "Greater operation";
  let description = [{
    Returns elementwise x > y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                     !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterEqualOp : MO_ComparisonBinaryOp<"greater_equal"> {
  let summary = "Greater Equal operation";
  let description = [{
    Returns elementwise x >= y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                           !mo.tensor<[2, 3], f32>
                                          ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let hasFolder = 1;
}

// TODO: Use an AllElementTypesMatch<["x", "y", "result"]> trait for MO_Tensor.
// TODO: Use an SameOperandsShape trait for MO_Tensor.
def MO_SelectOp : MO_TensorOp<"select", [Pure]>{
  let summary = "Select operation";
  let description = [{
    Return cond ? x : y (element-wise), where cond, x and y are input tensors.

    Example:

    ```mlir
    %cond: !mo.tensor<[2, 3], bool>
    %x: !mo.tensor<[2, 3], f32>
    %y: !mo.tensor<[2, 3], f32>
    %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$cond, MO_Tensor:$x, MO_Tensor:$y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MO_SliceOp : MO_TensorOp<"slice", [Pure]>{
  let summary = "Slice operation";
  let description = [{
    Return a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. `step` must contain nonzero elements.

    Example:
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesVectorIfRanked:$start,
                       MO_IndicesVectorIfRanked:$stop,
                       MO_IndicesVectorIfRanked:$step);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MO_SubOp : MO_ElementWiseBinaryOp<"sub"> {
  let summary = "Subtract operation";
  let description = [{
    Returns x - y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TanhOp : MO_ElementWiseUnaryOp<"tanh"> {
  let summary = "Tanh function";
  let description = [{
    Computes the hyperbolic tangent of the input tensor elements.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Softmax-like ops
//===----------------------------------------------------------------------===//

class MO_SoftmaxLikeOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [MO_SameOperandsAndResultType, Pure]> {
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_LogSoftmaxOp : MO_SoftmaxLikeOp<"logsoftmax"> {
  let summary = "Perfoms logsoftmax on tensor `input`";
  let description = [{
    Returns logsoftmax probability scores following:
     `log(softmax(input))`

    The scores are computed along the last axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SoftmaxOp : MO_SoftmaxLikeOp<"softmax"> {
  let summary = "Perfoms softmax on tensor `input`";
  let description = [{
    Returns softmax probability scores following:
     `exp(input) / sum(exp(input))`

    The scores are computed along the last axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.softmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

class MO_ReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
    name, traits # [Pure, MO_Reduction, MO_ReductionTraits]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_MeanOp : MO_ReductionOp<"mean"> {
  let summary = "Mean reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their mean value, and change that
    axis's dimension to 1 in the output shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.mean(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMulOp : MO_ReductionOp<"reduce.mul"> {
  let summary = "Product reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their product, and change that
    axis's dimension to 1 in the output shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.mul(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceAddOp : MO_ReductionOp<"reduce.add"> {
  let summary = "Sum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their sum, and change that axis's
    dimension to 1 in the output shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.add(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMinOp : MO_ReductionOp<"reduce.min"> {
  let summary = "Minimum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their minimum, and change that
    axis's dimension to 1 in the output shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.min(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMaxOp : MO_ReductionOp<"reduce.max"> {
  let summary = "Maximum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their maximum, and change that
    axis's dimension to 1 in the output shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.max(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Sliding window ops
//===----------------------------------------------------------------------===//

def MO_AvgPoolOp : MO_TensorOp<"avg_pool", [Pure, MO_SlidingWindow]>{
  let summary = "Average pooling operation";
  let description = [{
    Apply average pooling with the given filter shape, strides, and dilations.

    All hyperparameters (i.e. filter_shape, strides, dilations) must be of rank
    1, or unranked. If the input has static rank, all hyperparameters with
    static shape must have sizes the match the input rank. The input and output
    data tensors' ranks must match if statically known.

    This op currently only supports strides and dilations on the filter.

    TODO: Clarify how AvgPool should work with padding.

    Example:

    ```mlir
    %fs = mo.constant {
      value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %st = mo.constant {
      value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %di = mo.constant {
      value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %res = mo.avg_pool(%arg) [
        filter_shape = %fs, strides = %st, dilations = %di
    ] : (
      !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
    ) -> !mo.tensor<[9, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` `[` `filter_shape` `=` $filter_shape `,`
    `strides` `=` $strides `,` `dilations` `=` $dilations `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolOp : MO_TensorOp<"max_pool", [Pure, MO_SlidingWindow]>{
  let summary = "Max pooling operation";
  let description = [{
    Apply max pooling with the given filter shape, strides, and dilations.

    All hyperparameters (i.e. filter_shape, strides, dilations) must be of rank
    1, or unranked. If the input has static rank, all hyperparameters with
    static shape must have sizes the match the input rank. The input and output
    data tensors' ranks must match if statically known.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
    %fs = mo.constant {
      value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %st = mo.constant {
      value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %di = mo.constant {
      value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %res = mo.max_pool(%arg) [
        filter_shape = %fs, strides = %st, dilations = %di
    ] : (
      !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
    ) -> !mo.tensor<[9, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` `[` `filter_shape` `=` $filter_shape `,`
    `strides` `=` $strides `,` `dilations` `=` $dilations `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ConvOp : MO_TensorOp<"conv", [Pure, MO_SlidingWindow]>{
  let summary = "Convolution operation";
  let description = [{
    Compute the convolution of the input with the given filter, strides, and
    dilations.

    All hyperparameters (i.e. strides, dilations) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes the match the input rank. The input, output and filter
    tensors' ranks must match if statically known.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
    %st = mo.constant {
      value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %di = mo.constant {
      value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %res = mo.conv(%0, %1) [strides = %st, dilations = %di] : (
      !mo.tensor<[5, 5], f32>, !mo.tensor<[3, 3], f32>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>
    ) -> !mo.tensor<[2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `,` $filter `)` `[` `strides` `=` $strides `,`
    `dilations` `=` $dilations `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

class MO_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<mnemonic,
    traits # [DeclareOpInterfaceMethods<MO_Parameterization>,
              DeclareOpInterfaceMethods<MO_ParamDeclaration>]
> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BroadcastToOp : MO_ShapeModifierOp<"broadcast_to"> {
  let summary = "BroadcastTo Operation";
  let description = [{
    Broadcast the input tensor to the shape newShape.

    The broadcasting follows numpy semantics to determine validity of the
    two shapes.

    Example:

    ```mlir
    %from: !mo.tensor<[3], f32>
    %to = mo.constant {
      value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %res = mo.broadcast_to(%from, %to) : (
      !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasVerifier = 1;
}

def MO_ReshapeOp : MO_ShapeModifierOp<"reshape"> {
  let summary = "Reshape operation";
  let description = [{
    Return a tensor with the same underlying data, but different shape.

    Example:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %shape = mo.constant {
      value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
  }];

  let hasFolder = 1;
}

def MO_ShapeOfOp : MO_TensorOp<"shape_of", [
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>
]> {
  let summary = "ShapeOf Operation";
  let description = [{
    Get the shape of a tensor.

    Example:

    ```mlir
    // statically ranked
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %shape1 = mo.shapeof(%arg1) : !mo.tensor<[3], si64>

    // dynamically ranked
    %arg2: !mo.tensor<?, f32>
    %shape2 = mo.shapeof(%arg2) : !mo.tensor<[?], si64>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

def MO_BroadcastShapeOp : MO_TensorOp<"broadcast_shape", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
]> {
  let summary = "Broadcast Shape Operation";
  let description = [{
    Given two tensor shapes, return the shape that both broadcast to under
    the numpy rules:

    1. Make the shapes the same rank by adding 1 to the front
       of the shorter shape
    2. Ensure each dimension is equal, either by matching them or promoting a 1
       to a larger number from the other shape

    Example 1:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, shape1 will return [4, 2, 3]

    Example 2:

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, shape1 will return [10, 2, 5]
  }];

  let arguments = (ins MO_ShapeLike:$input_x,
                       MO_ShapeLike:$input_y,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

class MOStatic_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
    : MO_TensorOp<"static." # mnemonic, traits # [Pure]> {
  string cppNamespace = MODialect.cppNamespace # "::Static";

  let arguments = (ins MO_KnownTensor:$input);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];
  let hasFolder = 1;
}

def MOStatic_BroadcastToOp : MOStatic_ShapeModifierOp<"broadcast_to"> {
  let summary = "Broadcast operation for parametric shapes";
  let description = [{
    Broadcast the input tensor to the result tensor. The shape of the input and
    result tensors must not be unknown or contain unknown dimensions, but can be
    parametric. It is not recommended for frontend conversion code to rely on
    this op (use `mo.broadcast_to` with a constant shape-like tensor instead).

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
    %from: !mo.tensor<[3], f32>
    %res1 = mo.static.broadcast(%from)
      : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
    kgen.param.declare N = <...>
    %res2 = mo.static.broadcast(%from)
      : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
  }];

  let hasVerifier = 1;
}

def MOStatic_ReshapeOp : MOStatic_ShapeModifierOp<"reshape"> {
  let summary = "Reshape operation for parametric shapes";
  let description = [{
    Return a tensor with the same underlying data, but different shape. The
    shape of the input and result tensors must not be unknown or contain unknown
    dimensions, but can be parametric. It is not recommended for frontend
    conversion code to rely on this op (use `mo.reshape` with a constant
    shape-like tensor instead).

    Example:

    ```mlir
    %from: !mo.tensor<[2, 3], f32>
    %res = mo.static.reshape(%from)
      : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_ConcatOp : MO_TensorOp<"concat", [
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Concat Operation";
  let description = [{
    Concatenate the input tensors along a given dimension.

    Example:

    ```mlir
    %arg0: !mo.tensor<[2, 3], f32>
    %arg1: !mo.tensor<[2, 5], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
    ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
}

def MO_TransposeOp : MO_TensorOp<"transpose", [
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Transpose Operation";
  let description = [{
    Return a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Example:

    ```mlir
    %input : !mo.tensor<[2, 3], f32>
    %perm : !mo.tensor<[2], si64>
    %res = mo.transpose(%input, %perm) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

    %input : !mo.tensor<[?, 5, ?], f32>
    %perm : !mo.tensor<[3], si32>
    %res = mo.transpose(%input, %perm) : (
      !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
    ) -> !mo.tensor<[?, 5, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$perm,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_PadOp : MO_TensorOp<"pad", [Pure]> {
  let summary = "Pad Operation";
  let description = [{
    Pad the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assume input has rank `N`, the `paddings` tensor should have
    shape `(N, 2)`, where each row has the form `[before, after]`, indicating
    how many of `constant` to add before and after the contents of `input` in
    that dimension. The size of each dimension D of the padded output is:
    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
    %input: !mo.tensor<[2, 3], f32>
    %constant = mo.constant {
      value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
    %paddings = mo.constant {
      value = #M.dense_array<1, 0, 1, 1> : tensor<2x2xsi64>
    } : !mo.tensor<[2, 2], si64>
    %output   = mo.pad(%input, %paddings, %constant) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si64>, !mo.tensor<[], f32>
    ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  // TODO(#7987) support a padding_mode argument
  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_RankedScalarTensor:$constant);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def MO_GatherOp : MO_TensorOp<"gather", [
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Gather Operation";
  let description = [{
    Gather slices from input' axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    Example:

    ```mlir
    %input : !mo.tensor<[2, 2], f32>
    %indices: !mo.tensor<[2], si64>
    %axis = mo.constant {
      value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.gather(%input, %indices, %axis) : (
      !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// WhereOp
//===----------------------------------------------------------------------===//

def MO_WhereOp : MO_TensorOp<"where", [Pure]> {
  let summary = "Returns locations of nonzero values in a tensor";
  let description = [{
    Return a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<2x2xsi32>
    } : !mo.tensor<[3, 3], si32>
    %1 = mo.where(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

def MO_RangeOp : MO_TensorOp<"range", [
    MO_SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<MO_Parameterization>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>]> {
  let summary = "Create a sequence of numbers";
  let description = [{
    The sequence goes from `start` with increments of size `step` up to (but not
    including) `limit`. All arguments are mandatory and must have the same
    element type.

    Example:

    ```mlir
    %limit : !mo.tensor<[], f32>
    %start = mo.constant {
      value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
    %step = mo.constant {
      value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
    %res = mo.range(%start, %limit, %step) : (
      !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
    ) -> !mo.tensor<[?], f32>

    %startInt = mo.constant {
      value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
    %stepInt = mo.constant {
      value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
    %limitInt = mo.constant {
      value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
    %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
      !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
    ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarTensor:$start,
    MO_RankedScalarTensor:$limit,
    MO_RankedScalarTensor:$step,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
  private:
    /// Return the number of results given constant start/limit/step values.
    static ErrorOr<int64_t> getNumResElelements(
      int64_t start, int64_t limit, int64_t step);
  public:
  }];
}


//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//

class MO_DotBase<string name, list<Trait> traits = []> :
    MO_TensorOp<name, traits # [
    MO_SameOperandsAndResultElementType,
    MO_MatmulLike,
    DeclareOpInterfaceMethods<MO_Parameterization>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
]> {
  let description = [{
    Base class for all generic dot products (i.e. tensor contractions with two
    inputs), for example matmuls.
  }];

  let results = (outs MO_Tensor:$result);

  let hasVerifier = 1;
}

def MO_MatmulOp : MO_DotBase<"matmul"> {
  let summary = "Matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two 2D tensors.

    Example:

    ```mlir
    %lhs: ... !mo.tensor<[10, 20], f32>
    %rhs: ... !mo.tensor<[20, 5], f32>
    %res = mo.matmul(%lhs, %rhs) : (
      !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
    ) -> !mo.tensor<[10, 5], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_a, MO_Tensor:$input_b);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_BatchMatmulOp : MO_DotBase<"batch_matmul", [
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Batched matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.

    Example:

    ```mlir
    %lhs: ... !mo.tensor<[3, 4, 5], f32>
    %rhs: ... !mo.tensor<[3, 5, 6], f32>
    %res = mo.batch_matmul(%lhs, %rhs) :
      mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
    ) -> !mo.tensor<[3, 4, 6], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input_a,
    MO_Tensor:$input_b,
    MO_ParamDecls:$outputParamDecls);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
  private:
    /// Unique forward shape inference function name, set when materialized.
    static std::optional<StringRef> fwShapeFunc;
    /// Unique sideways shape inference function name, set when materialized.
    static std::optional<StringRef> sideShapeFunc;
  public:
  }];
}

//===----------------------------------------------------------------------===//
// Linear Algebra ops
//===----------------------------------------------------------------------===//

def MO_MatrixBandPartOp : MO_TensorOp<"matrix_band_part", [Pure]> {
  let summary = "Matrix Band Part operation";
  let description = [{
    Copy a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assume the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    Please explicitly note that with negative values, this kernel returns the
    entire lower or uppper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
    %arg: !mo.tensor<[3, 2, 3], f32>
    %num_lower = mo.constant {
      value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %num_upper = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.matrix_band_part(%arg, %num_lower, %num_upper) : (
      !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$num_lower,
    MO_RankedScalarIndexTensor:$num_upper);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
     `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
