//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD

include "KGEN/KGENDialect/KGENTypes.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/RMODialect/RMODialect.td"
include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOEnums.td"
include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for ops in this dialect.
class RMO_Op<string mnemonic, list<Trait> traits = []>
  : Op<RMODialect, mnemonic, traits # [RMO_TensorOpInterface, MO_ParamDeclaration]> {

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RMO Asserting Rebinds
//===----------------------------------------------------------------------===//
def RMO_RebindTensorShapeOp : RMO_Op<"rebind_tensor_shape", [RMO_TensorSameDTypeOperandsAndResults]> {
  let summary = "Rebind the input tensor's shape with runtime check.";
  let description = [{
    Unlike `mo.rebind` this also has the semantics of doing a runtime check
    that the given shapes are compatible.

    Right now this is modeled as a side-effect of the operation due to
    limitations of modeling assertions in our stack. In the future this will
    change and this operation may return either an error or a tensor.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ParamDecls:$outputParamDecls,
    DefaultValuedAttr<StrAttr, "\"RMO rebind shape check failed.\"">:$message
  );
  let results = (outs MO_Tensor:$output);

  let builders = [
    /// Inferred return type builders
    OpBuilder<(ins
      "TensorValue":$input,
      "ShapeAttr":$newShape,
      CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls
    )>,

    OpBuilder<(ins
      "TensorValue":$input,
      "ShapeAttr":$newShape,
      "StringRef":$message,
      CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls
    )>
  ];

  // If we can prove the rebind is rebinding to the same shape, become no-op
  let hasFolder = 1;
}


class RMO_PureBinaryElementWiseOp<string name, list<Trait> traits = []>
    : RMO_Op<name, traits # [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins
    MO_KnownTensor:$input_x,
    MO_KnownTensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                $cppClass::Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryElemwiseImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_AddOp : RMO_PureBinaryElementWiseOp<"add", [Commutative]> {
  let summary = "Flexible binary add op";
  let description = [{
    A flexible binary elementwise add operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_SubOp : RMO_PureBinaryElementWiseOp<"sub", []> {
  let summary = "Flexible binary sub op";
  let description = [{
    A flexible binary elementwise sub operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_MulOp : RMO_PureBinaryElementWiseOp<"mul", [Commutative]> {
  let summary = "Flexible binary mul op";
  let description = [{
    A flexible binary elementwise mul operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_DivOp : RMO_PureBinaryElementWiseOp<"div", []> {
  let summary = "Flexible binary div op";
  let description = [{
    A flexible binary elementwise div operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_ModOp : RMO_PureBinaryElementWiseOp<"mod", []> {
  let summary = "Flexible binary mod op";
  let description = [{
    A flexible binary elementwise mod operation with implicit broadcasting and implicit dtype promotion.
  }];
}

// TODO(GEX-568): investigate pow. It looks like the MO op allows for different input types without casting.
// For example, raising a float to the power of an integer. We update the lowering to avoid unnecessary casts.
def RMO_PowOp : RMO_PureBinaryElementWiseOp<"pow", []> {
  let summary = "Flexible binary pow op";
  let description = [{
    A flexible binary elementwise pow operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_MinOp : RMO_PureBinaryElementWiseOp<"min", [Commutative]> {
  let summary = "Flexible binary min op";
  let description = [{
    A flexible binary elementwise min operation with implicit broadcasting and implicit dtype promotion.
  }];
}

def RMO_MaxOp : RMO_PureBinaryElementWiseOp<"max", [Commutative]> {
  let summary = "Flexible binary max op";
  let description = [{
    A flexible binary elementwise max operation with implicit broadcasting and implicit dtype promotion.
  }];
}

class RMO_PureBinaryComparisonOp<string name, list<Trait> traits = []>
    : RMO_Op<name, traits # [Pure, InferTypeOpAdaptor, MO_SingleDevice]> {
  let arguments = (ins
    MO_KnownTensor:$input_x,
    MO_KnownTensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryComparisonImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_EqualOp : RMO_PureBinaryComparisonOp<"equal", [Commutative]> {
  let summary = "Flexible equality comaparison op";
  let description = [{
    A flexible binary elementwise equality comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_GreaterOp : RMO_PureBinaryComparisonOp<"greater"> {
  let summary = "Flexible greater than comparison op";
  let description = [{
    A flexible binary elementwise greater than comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_GreaterEqualOp : RMO_PureBinaryComparisonOp<"greater_equal"> {
  let summary = "Flexible greater than or equal to comparison op";
  let description = [{
    A flexible binary elementwise greater than or equal to comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_NotEqualOp : RMO_PureBinaryComparisonOp<"not_equal", [Commutative]> {
  let summary = "Flexible not equal to op";
  let description = [{
    A flexible binary elementwise not equal comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

class RMO_PureBinaryLogicalOp<string name, list<Trait> traits = []>
    : RMO_Op<name, traits # [Pure, InferTypeOpAdaptor, MO_SingleDevice]> {
  let arguments = (ins
    MO_BooleanMask:$input_x,
    MO_BooleanMask:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context,
                                std::optional<Location> location,
                                Adaptor adaptor,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryLogicalImpl(
          context, $cppClass::getOperationName(), location,
          cast<MO::TensorValue>(adaptor.getInputX()),
          cast<MO::TensorValue>(adaptor.getInputY()),
          inferredReturnTypes);
    }
  }];
}

def RMO_AndOp : RMO_PureBinaryLogicalOp<"and", [Commutative]> {
  let summary = "Flexible and op";
  let description = [{
    A flexible binary elementwise and comparison operation with implicit broadcasting.

    Returns elementwise `x & y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.and(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_OrOp : RMO_PureBinaryLogicalOp<"or", [Commutative]> {
  let summary = "Flexible or op";
  let description = [{
    A flexible binary elementwise or comparison operation with implicit broadcasting.

    Returns elementwise `x | y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.or(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_XorOp : RMO_PureBinaryLogicalOp<"xor", [Commutative]> {
  let summary = "Flexible xor op";
  let description = [{
    A flexible binary elementwise xor comparison operation with implicit broadcasting.

    Returns elementwise `x ^ y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.xor(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
  }];
}

def RMO_BroadcastLikeOp : RMO_Op<"broadcast_like", [Pure]> {
  let summary = "Broadcast the `input` tensor to be the same shape as `like`.";
  let description = [{
    Equivalent to the following:

      ```
      %shape_of = rmo.shape_of(%like)
      %result = rmo.broadcast_to(%broadcast, %shape_of)
      ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$shapeLike,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$shapeLike)>,
  ];

  let hasVerifier = 1;
}

def RMO_SigmoidOp : RMO_Op<"sigmoid", [Pure]> {
  let summary = "Logistic sigmoid operation";
  let description = [{
    Returns `\frac{1}{1+e^{-x}} = \fract{e^x}{e^x + 1}`, the logistic sigmoid.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %result = rmo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_ConcatOp : RMO_Op<"concat", [
  Pure,
  InferTypeOpAdaptor,
]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor. The input tensors and output tensors all
    has the same shape except along the constant concatenation dimension `axis`.
    The size of the concatenation dimension in output tensor would have be the
    sum of sizes of the concatenation dimension in input tensors.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Shape restrictions:
    1. All inputs must have known and equal ranks.
    2. Input dimensions must match along non-concat-axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %res = rmo.concat(%arg0, %arg1) {axis = 1}  : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
      Variadic<MO_KnownTensor>:$inputs,
      Builtin_IntegerAttr:$axis,
      OptionalAttr<AnyAttrOf<[Builtin_IntegerAttr,KGEN_ParamDeclRefAttr]>>:$out_dim,
      MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def RMO_ReshapeOp : RMO_Op<"reshape", [
  Pure,
  InferTypeOpAdaptor,
]> {
  let summary = "Reshape a tensor";
  let description = [{
    Returns a tensor with the same underlying data as `input`, but the shape of `newShape`.

    A static dimension set to `-1` will be automatically computed.
    Using `-1` will result in an error if `newShapes` contains a zero sized dimensions.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
  }];
  let arguments = (ins
    MO_KnownTensor:$input,
    MOSH_ShapeAttr:$newShape);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_BroadcastToOp : RMO_Op<"broadcast_to", [
  Pure,
  InferTypeOpAdaptor,
]> {
  let summary = "Broadcast a tensor";
  let description = [{
    Broadcasts the input tensor to the specified shape.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
  }];
  let arguments = (ins
    MO_KnownTensor:$input,
    MOSH_ShapeAttr:$newShape);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_ShapeToTensorOp : RMO_Op<"shape_to_tensor", [
  Pure,
  InferTypeOpAdaptorWithIsCompatible,
]> {
  let summary = "Converts a shape to a tensor";
  let description = [{
    Converts a statically known shape attr into a 1d tensor of the shape.
  }];
  let arguments = (ins MOSH_ShapeAttr:$shape);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_MatmulOp : RMO_Op<"matmul", [
  Pure,
  InferTypeOpAdaptor,
  MO_SingleDevice,
]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs general matrix multiplication with broadcasting.

    If the lhs is 1d, it will be reshaped to `1xD`.
    If the rhs is 1d, it will be reshaped to `Dx1`.
    In both cases, the addition `1` dimensions will be removed from the output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated as a maxtrix.
    The lhs matrix will have the shape `MxK`.
    The rhs matrix will have the shape `KxN`.
    The output will have the shape `MxN`
    The `K` dimensions must be equivalent in both matrices.

    The remaining outer dimensions will be broadcasted.

    Example shapes with outputs:
    [K] @ [K] -> []
    [5, K] @ [K] -> [5]
    [K] @ [K, 6] -> [6]
    [4, K] @ [K, 6] -> [4, 6]
    [8, 10, 4, K] @ [8, 1, K, 6] -> [8, 10, 4, 6]
    [10, 4, K] @ [K, 6] -> [10, 4, 6]
    [10, 4, K] @ [K] -> [10, 4]
    [K] @ [10, 4, K, 7] -> [10, 4, 7]
  }];
  let arguments = (ins
    MO_KnownTensor:$input_x,
    MO_KnownTensor:$input_y);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_ConvOp : RMO_Op<"conv", [
  Pure,
  InferTypeOpAdaptor,
  MO_SingleDevice,
]> {
  let summary = "Convolution operation";
  let description = [{
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    `strides`, `dilations`, and `padding` are all shape attributes.
    If the input has static rank, and must have have sizes of `input_rank - 2`,
    except padding, which must have size `2 * (input_rank - 2)`. Individual
    elements in the hyperparameters apply to corresponding dimensions of the
    input (after ignoring the batch and channel dimensions), with padding
    representing a before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match.

    The number of input and output channels must both be divisible by the number
    of groups `num_groups`.

    Example:

    ```mlir
    %input: !mo.tensor<[10, 5, 5, 32], f32>,
    %filter: !mo.tensor<[2, 2, 32, 64], f32>

    %result = rmo.conv(%input, %filter) {
      strides = #mosh<ape[2, 2]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      num_groups = 1 : si64
    } : (
      !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>
    ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_KnownTensor:$input,
    MO_KnownTensor:$filter,
    KnownShapeAttr:$strides,
    KnownShapeAttr:$dilations,
    KnownShapeAttr:$paddings,
    SI64Attr:$num_groups,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout
  );

  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_ConvTransposeOp : RMO_Op<"conv_transpose", [
  Pure,
  InferTypeOpAdaptor,
  MO_SingleDevice,
]> {
  let summary = "Transposed convolution operation";
  let description = [{
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
    %res = rmo.conv_transpose(%input, %filter) {
      strides = #mosh<ape<1, 1>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      output_paddings = #mosh<ape<0, 0>> : !mosh.ape
    } : (
      !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
    ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_KnownTensor:$input,
    MO_KnownTensor:$filter,
    KnownShapeAttr:$strides,
    KnownShapeAttr:$dilations,
    KnownShapeAttr:$paddings,
    KnownShapeAttr:$output_paddings,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout
  );

  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class RMO_PoolOp<string name> : RMO_Op<name, [
  Pure,
  InferTypeOpAdaptor,
]> {
    let arguments = (ins
      MO_KnownTensor:$input,
      KnownShapeAttr:$filter_shape,
      KnownShapeAttr:$strides,
      KnownShapeAttr:$dilations,
      KnownShapeAttr:$paddings,
      BoolAttr:$ceil_mode
    );
    let results = (outs MO_KnownTensor:$result);

    let assemblyFormat = [{
      `(` operands `)` attr-dict `:` functional-type(operands, results)
    }];
  }

def RMO_AvgPoolOp : RMO_PoolOp<"avg_pool">{
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d average pooling (so input must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    Example:

    ```mlir
    %res = rmo.avg_pool(%input) {
      filter_shape = #mosh<ape<2, 3>> : !mosh.ape
      strides = #mosh<ape<2, 3>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      ceil_mode = false
      count_boundary = true
    } : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_KnownTensor:$input,
    KnownShapeAttr:$filter_shape,
    KnownShapeAttr:$strides,
    KnownShapeAttr:$dilations,
    KnownShapeAttr:$paddings,
    BoolAttr:$ceil_mode,
    BoolAttr:$count_boundary
  );
}

def RMO_MaxPoolOp : RMO_PoolOp<"max_pool">{
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    Example:

    ```mlir
    %res = rmo.max_pool(%input) {
      filter_shape = #mosh<ape<2, 3>> : !mosh.ape
      strides = #mosh<ape<2, 3>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      ceil_mode = false
    } : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
  }];
}

def RMO_SelectOp : RMO_Op<"select", [
  Pure,
  InferTypeOpAdaptor,
  MO_SingleDevice,
]> {
  let summary = "Select operation with broadcasting and dtype casting";
  let description = [{
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Broadcasting is handled by first calculating the common shape between `x` and `y`.
    Then, the `cond` shape and common shape are merged to figure out the final shape.
    All inputs are broadcast to the same final shape.

    Dtype casting only applies to the `x` and `y` inputs. `cond` must be a bool.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[1, 3], f32>
      %y: !mo.tensor<[2, 3], i8>
      %res = rmo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins
    MO_BooleanMask:$cond,
    MO_KnownTensor:$input_x,
    MO_KnownTensor:$input_y
  );
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_TopKOp : RMO_Op<"top_k", [
  Pure,
  InferTypeOpAdaptor,
  MO_SingleDevice,
]> {
  let summary = "Top-K operation";
  let description = [{
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis.
    Returns values sorted along the axis.

    axis: The axis to compute the largest values over.
      The axis must be in [-rank, rank).
    k: The number of values to compute.

    Example:
    ```mlir
      %input: !mo.tensor<[2, 6], si64>
      %values, %indices = rmo.top_k(%input) {k = 3, axis = 1} :
        (!mo.tensor<[2, 6], si64>) ->
        (!mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>)
    ```
  }];
  let arguments = (ins
    MO_KnownTensor:$input,
    SI64Attr:$k,
    SI64Attr:$axis,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs
    MO_KnownTensor:$values,
    MO_I64Tensor:$indices
  );
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];
}

def RMO_SliceOp : RMO_Op<"slice", [
  Pure,
  InferTypeOpAdaptor,
]> {
  let summary = "Slice operation with type inference.""";
  let description = [{
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `starts`, `stops`, and
    `steps` shape attributes.
    Each shape attributes has N elements, one for each dimension of the `input`
    tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `starts[i]:stops[i]:steps[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `starts` and `stops`, e.g., -1
       represents the largest axis.
    3. Unlike `mo.slice`, Out of bound indices in `starts` and `stops` are not
       allowed and each must be in [-dim, dim], where `dim` is the dimension in
       the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[10, 10], f32>

      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.slice(%input) {starts = #mosh<ape[1, -6]> : !mosh.ape, steps = #mosh<ape[5, 1]> : !mosh.ape, stops = #mosh<ape[-3, 6]> : !mosh.ape} : (
        !mo.tensor<[10, 10], f32>,
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       MOSH_ShapeAttr:$starts,
                       MOSH_ShapeAttr:$stops,
                       MOSH_ShapeAttr:$steps,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
