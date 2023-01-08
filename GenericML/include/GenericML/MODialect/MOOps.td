//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOOPS_TD
#define GENERICML_MODIALECT_MOOPS_TD

include "GenericML/MODialect/MOInterfaces.td"
include "GenericML/MODialect/MOTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
    : Op<MODialect, mnemonic, [MO_MOOpInterface] # traits>;

class MO_ElementWiseOp<string name, list<Trait> traits = []>
    : MO_Op<name, traits # [Pure, SameOperandsAndResultType]> {
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

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_Op<"graph", [HasParent<"ModuleOp">, FunctionOpInterface,
                                 Symbol, SymbolTable, IsolatedFromAbove
  // TODO : Re-enable TransformableGraphInterface
  // DeclareOpInterfaceMethods<TransformableGraphInterface>
]> {
  let summary = "Function that contains a MO graph with other ops mixed in";
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getFunctionBody;
    using FunctionOpInterfaceTrait::front;
    using FunctionOpInterfaceTrait::end;
    Block *getBodyBlock() { return &getFunctionBody().front(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let builders = [
    /// Create a new mtfl.graph with an empty body.
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

def MO_OutputOp
    : MO_Op<"output", [Terminator, ReturnLike, HasParent<"MO::GraphOp">]> {
  let summary = "MO output operation";
  let description = [{
    The "mo.output" operation specifies the output values for a MO
    function. The operation takes variable number of operands and produces
    no results. The operand number and types must match the signature of
    the mo.graph that contains the operation. For example:

    ```mlir
      mo.graph @foo() : i32, f8 {
      ... mo.output e0, % 1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_Op<"constant", [ConstantLike, Pure]> {
  let summary = "constant MO operation";
  let description = [{
    Op to store constant tensors in model.
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[M_ArrayElementsAttr, Builtin_DenseResourceElementsAttr]>]
    >:$value);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def MO_AddOp : MO_ElementWiseBinaryOp<"add"> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.
  }];
}

def MO_AbsOp : MO_ElementWiseUnaryOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.
  }];
}

def MO_DivOp : MO_ElementWiseBinaryOp<"div"> {
  let summary = "Divide operation";
  let description = [{
    Returns x / y, where x and y are input tensors.
  }];
}

def MO_ExpOp : MO_ElementWiseUnaryOp<"exp"> {
  let summary = "Exponential function";
  let description = [{
    Returns exp(x), where x is the input tensor.
  }];
}

def MO_ErfOp : MO_ElementWiseUnaryOp<"erf"> {
  let summary = "Absolute value operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.
  }];
}

def MO_GeluOp : MO_ElementWiseUnaryOp<"gelu"> {
  let summary = " Gaussian Error Linear Unit operation";
  let description = [{
    Returns GELU(x) = (x / 2) * (1 + erf(x/sqrt(2))), where erf denotes the
    (Gaussian) error function.
  }];
}

def MO_MulOp : MO_ElementWiseBinaryOp<"mul"> {
  let summary = "Multiply operation";
  let description = [{
    Returns x * y, where x and y are input tensors.
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensor.
  }];
}

def MO_SoftmaxOp : MO_Op<"softmax", [SameOperandsAndResultType, Pure]> {
  let summary = "Perfoms softmax on tensor `input`";
  let description = [{
    Returns softmax probability scores following:
     `exp(input) / sum(exp(input))`

    Axis is a vector representing the axes to reduce.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_SubOp : MO_ElementWiseBinaryOp<"sub"> {
  let summary = "Subtract operation";
  let description = [{
    Returns x - y, where x and y are input tensors.
  }];
}

def MO_TanhOp : MO_ElementWiseUnaryOp<"tanh"> {
  let summary = "Tanh function";
  let description = [{
    Computes the hyperbolic tangent of the input tensor elements.
  }];
}

//===----------------------------------------------------------------------===//
// Sliding window ops
//===----------------------------------------------------------------------===//

/// Checks if a type has rank 1 if statically ranked.
def DimOneIfRankedTypePred : Or<[Neg<HasRankPred>, HasAnyRankOfPred<[1]>]>;

/// Checks if a MO_TensorType has SI64 elements.
def HasSI64ElementTypePred : SubstLeaves<
  "$_self", "cast<M::MO::TensorType>($_self).getElementType()", SI64.predicate>;

/// Confines the MO_TensorType so that it has SI64 elements, and rank 1 if the
/// rank is static. This is useful because the verifier for these constraints is
/// generated automatically.
def ShapeLikeIfRanked : ConfinedType<MO_Tensor,
    [DimOneIfRankedTypePred, HasSI64ElementTypePred],
    "dim 1 or unranked with signed 64 integer elements"> {
  let cppClassName = "::M::MO::TensorType";
}

def MO_MaxPoolOp : MO_Op<"max_pool", [Pure, MO_SlidingWindow]>{
  let summary = "Max pooling operation";
  let description = [{
    Apply max pooling with the given filter size and strides.

    All hyperparameters (i.e. filter_shape, strides, dilations) must be of rank
    1, or unranked. If the input has static rank, all hyperparameters with
    static shape must have sizes the match the input rank.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    ShapeLikeIfRanked:$filter_shape,
    ShapeLikeIfRanked:$strides,
    ShapeLikeIfRanked:$dilations);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` `[` `filter_shape` `=` $filter_shape `,`
    `strides` `=` $strides `,` `dilations` `=` $dilations `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

def MO_BroadcastToOp : MO_Op<"broadcast_to", [Pure]> {
  let summary = "BroadcastTo Operation";
  let description = [{
    Broadcast the input_x tensor to the shape input_shape.

    The Broadcasting follows numpy semantics to determine validity of the
    two shapes.
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_shape);
  let results = (outs MO_Tensor:$result);
}

def MO_ConcatOp : MO_Op<"concat", [Pure]> {
  let summary = "Concat Operation";
  let description = [{
    Concatenate the input tensors along a given dimension.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs, MO_Tensor:$axis);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $inputs `)` attr-dict `:`functional-type($inputs, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def MO_GatherOp : MO_Op<"gather", [Pure]> {
  let summary = "Gather Operation";
  let description = [{
    Gather slices from params' axis according to indices.
  }];

  let arguments = (ins MO_RankedNonScalarTensor:$params,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarTensor:$axis);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//

class MO_DotBase<string name, list<Trait> traits = []>
  : MO_Op<name, traits # [SameOperandsAndResultElementType]> {
  let arguments = (ins MO_Tensor:$input_a, MO_Tensor:$input_b);
  let results = (outs MO_Tensor:$result);

  let description = [{
    Base class for all generic dot products (i.e. tensor contractions with two
    inputs), for example matmuls.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MatmulOp : MO_DotBase<"matmul", [MO_MatmulLike]> {
  let summary = "Matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two 2D tensors.
  }];
}

def MO_BatchMatmulOp : MO_DotBase<"batch_matmul", [MO_MatmulLike]> {
  let summary = "Batched matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.
  }];
}

#endif // GENERICML_MODIALECT_MOOPS_TD
