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

include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/RMODialect/RMODialect.td"
include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"

// Base class for ops in this dialect.
class RMO_Op<string mnemonic, list<Trait> traits = []>
  : Op<RMODialect, mnemonic, traits # [RMO_TensorOpInterface, MO_ParamDeclaration]> {

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

// Base class for ops that map directly to a MO tensor op
class RMO_MOAnaloguePureTensorOp<string mnemonic, list<Trait> traits = []>
  : RMO_Op<mnemonic, traits # [DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>, Pure]> {
  let builders = [
    /// This builder is an implicit return type based on "inferMOTypes" from
    /// the RMO_MOAnalogue interface.
    OpBuilder<
      (ins "ValueRange":$inputValues, "MO::GraphOp":$graphOp),
      [{
        SmallVector<Type> _moOperandTypes, moResultTypes;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moOutputParamDecls;
        SmallVector<SmallVector<KGEN::ParamDeclAttr>> moInputParamDecls;

        // Rebind values to remove any possible `?`
        SmallVector<Value> reboundInputs = rebindUnknownShapeInValues(inputValues, graphOp, $_builder);

        inferMOTypesImpl(odsState.getContext(), graphOp, ValueRange{reboundInputs}, std::nullopt, _moOperandTypes,
                                moResultTypes, moInputParamDecls, moOutputParamDecls);

        // Flatten all decls to be used by the main operation
        SmallVector<KGEN::ParamDeclAttr> moOutputParamDeclsFlattened;
        for (const auto& resultDecls: moOutputParamDecls) {
          moOutputParamDeclsFlattened.append(resultDecls);
        }

        // Call normal output type, input value builder
        build($_builder, $_state, mlir::TypeRange{moResultTypes},
                            reboundInputs);

        // Manually set outputParamDecls field
        $_state.addAttribute(getOutputParamDeclsAttrName($_state.name), $_builder.getAttr<ParamDeclArrayAttr>(moOutputParamDeclsFlattened));
      }]
    >,
  ];

  let extraClassDeclaration = [{

    /// Implementation of `inferMOTypes` using operand adaptor
    static void inferMOTypesImpl(
      MLIRContext* context,
      MO::GraphOp graphOp,
      Adaptor operands,
      std::optional<ValueRange> rmoResults,
      SmallVectorImpl<Type>& moOperandTypes,
      SmallVectorImpl<Type>& moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>>& moOutputParamDecls
    );
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

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$output);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "ShapeAttr":$newShape, CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls)>
  ];

  // If we can prove the rebind is rebinding to the same shape, become no-op
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// 1-1 MO Op Analogues
//===----------------------------------------------------------------------===//

class RMO_MOAnaloguePureElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits> {
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureUnaryElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesElemwiseUnaryImpl(
          context, graphOp, operands.getInput().cast<TensorValue>(),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}

class RMO_MOAnaloguePureBinaryElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y, MO_ParamDecls:$outputParamDecls);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        MO::TensorType lhsType = cast<MO::TensorType>(operands.getInputX().getType());
        MO::TensorType rhsType = cast<MO::TensorType>(operands.getInputY().getType());

        MO::TensorType refinedShape = refineShape(lhsType, rhsType);
        moOperandTypes.push_back(refinedShape);
        moOperandTypes.push_back(refinedShape);
        moInputParamDecls.push_back({});
        moInputParamDecls.push_back({});

        moResultTypes.push_back(refinedShape);
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_AbsOp : RMO_MOAnaloguePureUnaryElementWiseOp<"abs"> {
  let summary = "Absolute value operation";
  let description = [{
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_CeilOp : RMO_MOAnaloguePureUnaryElementWiseOp<"ceil"> {
  let summary = "Ceiling operation";
  let description = [{
    Returns the smallest largest integer greater than `x`, where `x` is input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_CosOp : RMO_MOAnaloguePureUnaryElementWiseOp<"cos"> {
  let summary = "Cosine operation";
  let description = [{
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_ErfOp : RMO_MOAnaloguePureUnaryElementWiseOp<"erf"> {
  let summary = "(Gaussian) error function operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_ExpOp : RMO_MOAnaloguePureUnaryElementWiseOp<"exp"> {
  let summary = "Exponential function operation";
  let description = [{
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_FloorOp : RMO_MOAnaloguePureUnaryElementWiseOp<"floor"> {
  let summary = "Floor operation";
  let description = [{
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_GeluOp : RMO_MOAnaloguePureUnaryElementWiseOp<"gelu"> {
  let summary = "Gaussian error linear unit operation";
  let description = [{
    Returns `(x / 2) * (1 + erf(x/sqrt(2)))`, where `erf` denotes the
    (Gaussian) error function.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_LogOp : RMO_MOAnaloguePureUnaryElementWiseOp<"log"> {
  let summary = "Natural logarithm operation";
  let description = [{
    Returns the natural logorithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_Log1pOp : RMO_MOAnaloguePureUnaryElementWiseOp<"log1p"> {
  let summary = "Natural logarithm of `1 + x` operation ";
  let description = [{
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_LogSoftmaxOp : RMO_MOAnaloguePureUnaryElementWiseOp<"logsoftmax"> {
  let summary = "Logarighmic softmax operation";
  let description = [{
    Returns `log(softmax(x))`, where `x` is input tensor.

    The softmax is applied along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_NegativeOp : RMO_MOAnaloguePureUnaryElementWiseOp<"negative"> {
  let summary = "Negation operation";
  let description = [{
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_NotOp : RMO_MOAnaloguePureTensorOp<"not"> {
  let summary = "Logical negation operation";
  let description = [{
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = rmo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesElemwiseUnaryImpl(
          context, graphOp, operands.getInput().cast<TensorValue>(),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}

def RMO_ReluOp : RMO_MOAnaloguePureUnaryElementWiseOp<"relu"> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_RoundOp : RMO_MOAnaloguePureUnaryElementWiseOp<"round"> {
  let summary = "Rounding operation";
  let description = [{
    Returns the elementwise nearest integer, with ties going away from zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_RoundEvenOp : RMO_MOAnaloguePureUnaryElementWiseOp<"roundeven"> {
  let summary = "Round even operation";
  let description = [{
    Returns the elementwise neareset integer, with ties going toward the
    nearest even number.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.roundeven(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_RsqrtOp : RMO_MOAnaloguePureUnaryElementWiseOp<"rsqrt"> {
  let summary = "Inverse square root operation";
  let description = [{
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_SigmoidOp : RMO_MOAnaloguePureUnaryElementWiseOp<"sigmoid"> {
  let summary = "Logistic sigmoid operation";
  let description = [{
    Returns `\frac{1}{1+e^{-x}} = \fract{e^x}{e^x + 1}`, the logistic sigmoid.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_SinOp : RMO_MOAnaloguePureUnaryElementWiseOp<"sin"> {
  let summary = "Sine operation";
  let description = [{
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_SqrtOp : RMO_MOAnaloguePureUnaryElementWiseOp<"sqrt"> {
  let summary = "Square root operation";
  let description = [{
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_TanhOp : RMO_MOAnaloguePureUnaryElementWiseOp<"tanh"> {
  let summary = "Hyperbolic tangent operation";
  let description = [{
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_TruncOp : RMO_MOAnaloguePureUnaryElementWiseOp<"trunc"> {
  let summary = "Truncation operation";
  let description = [{
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_AddOp : RMO_MOAnaloguePureBinaryElementWiseOp<"add", [Commutative]> {
  let summary = "Binary addition";
  let description = [{
    Does a non-broadcasted elementwise addition.
  }];
}

def RMO_ShapeOfOp : RMO_MOAnaloguePureTensorOp<"shape_of"> {
  let summary = "Get shape of tensor";
  let description = [{
    Calculates the runtime shape of the given tensor as a tensor.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_IndicesTensor:$shape);
}

def RMO_BroadcastShapeOp : RMO_MOAnaloguePureTensorOp<"broadcast_shape"> {
  let summary = "Apply broadcasting rules";
  let description = [{
    Given two tensors representing shapes, calculate the result of broadcasting
    the shapes.
  }];
  let arguments = (ins MO_ShapeLikeIfRanked:$input_x,
                       MO_ShapeLikeIfRanked:$input_y,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLikeIfRanked:$shape);
}

def RMO_ReshapeOp : RMO_MOAnaloguePureTensorOp<"reshape"> {
  let summary = "Reshape a tensor";
  let description = [{
    Reshape the `input` tensor to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_BroadcastToOp : RMO_MOAnaloguePureTensorOp<"broadcast_to"> {
  let summary = "Broadcast a tensor";
  let description = [{
    Broadcast the given `input` to the shape represented in `newShape`.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_CastOp : RMO_MOAnaloguePureTensorOp<"cast"> {
  let summary = "Cast a tensor to another dtype";
  let description = [{
    Casts the given data, changing from one dtype to another.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

def RMO_MulOp : RMO_MOAnaloguePureBinaryElementWiseOp<"mul", [Commutative]> {
  let summary = "Binary multiplication";
  let description = [{
    Does a non-broadcasted elementwise multiplication.
  }];
}

def RMO_ModOp : RMO_MOAnaloguePureBinaryElementWiseOp<"mod"> {
  let summary = "Binary modulus op";
  let description = [{
    Does a non-broadcasted elementwise modulus.
  }];
}

def RMO_MinOp : RMO_MOAnaloguePureBinaryElementWiseOp<"min", [Commutative]> {
  let summary = "Binary minimum op";
  let description = [{
    Does a non-broadcasted elementwise minimum.
  }];
}

def RMO_MaxOp : RMO_MOAnaloguePureBinaryElementWiseOp<"max", [Commutative]> {
  let summary = "Binary maximum op";
  let description = [{
    Does a non-broadcasted elementwise maximum.
  }];
}

def RMO_DivOp : RMO_MOAnaloguePureBinaryElementWiseOp<"div"> {
  let summary = "Binary division op";
  let description = [{
    Does a non-broadcasted elementwise division.
  }];
}

def RMO_PowOp : RMO_MOAnaloguePureBinaryElementWiseOp<"pow"> {
  let summary = "Binary power op";
  let description = [{
    Does a non-broadcasted elementwise power.
  }];
}

def RMO_SubOp : RMO_MOAnaloguePureBinaryElementWiseOp<"sub"> {
  let summary = "Binary subtraction op";
  let description = [{
    Does a non-broadcasted elementwise subtraction.
  }];
}

/// TODO: of showing this in it's own namespace.
/// This is an example experimental thing to prove this is easy to write
def RMO_BroadcastAddOp : RMO_Op<"broadcast.add", [Pure, RMO_TensorSameDTypeOperandsAndResults]> {
  let summary = "Add with implicit broadcast Operation";
  let description = [{
    Like `rmo.add` but supports implicit broadcasting!
  }];
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}


class RMO_MOAnaloguePureReductionOp<string name, bit input_result_same_dtype=1,
  list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name,
    traits # !if(input_result_same_dtype, [AllDTypesMatch<["input", "result"]>], [])> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MeanOp : RMO_MOAnaloguePureReductionOp<"mean"> {
  let summary = "Mean reduction operation";
  let description = [{
    Reduces `input` elements across `axis` to their mean value, changng that
    axis's dimension to 1.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.mean(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def RMO_ReduceMulOp : RMO_MOAnaloguePureReductionOp<"reduce.mul"> {
  let summary = "Product reduction operation";
  let description = [{
    Reduces `input` elements across `axis` to their product, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.reduce.mul(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def RMO_ReduceAddOp : RMO_MOAnaloguePureReductionOp<"reduce.add"> {
  let summary = "Sum reduction operation";
  let description = [{
    Reduces `input` elements across `axis` to their sum, changing that axis's
    dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.reduce.add(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMinOp : RMO_MOAnaloguePureReductionOp<"reduce.min"> {
  let summary = "Minimum reduction operation";
  let description = [{
    Reduces `input` elements across `axis` to their minimum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.reduce.min(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMaxOp : RMO_MOAnaloguePureReductionOp<"reduce.max"> {
  let summary = "Maximum reduction operation";
  let description = [{
    Reduces `input` elements across `axis` to their maximum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.reduce.max(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

class RMO_ArgMinMaxOp<string mnemonic> : RMO_MOAnaloguePureReductionOp<mnemonic,
  /*input_result_same_dtype=*/0> {
  let results = (outs MO_IndicesTensor:$result);
}

def MO_ArgMaxOp : RMO_ArgMinMaxOp<"arg_max"> {
  let summary = "Maximum-value indices operation";
  let description = [{
    This op is equivalent to reduce_max, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical maximum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<si32>
      } : !mo.tensor<[], si32>
      %1 = rmo.arg_max(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_ArgMinOp : RMO_ArgMinMaxOp<"arg_min"> {
  let summary = "Minimum-value indices operation";
  let description = [{
    This op is equivalent to reduce_min, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical minimum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<si32>
      } : !mo.tensor<[], si32>
      %1 = rmo.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

class RMO_MOAnaloguePureScatterOp<string name> : RMO_MOAnaloguePureTensorOp<
  name, [AllDTypesMatch<["input", "updates", "result"]>]
> {
let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_IndicesTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesScatterImpl(
          context, graphOp, operands.getInput().cast<TensorValue>(),
          operands.getUpdates().cast<TensorValue>(),
          operands.getIndices().cast<TensorValue>(),
          operands.getAxis().cast<TensorValue>(), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_ScatterOp : RMO_MOAnaloguePureScatterOp<"scatter">{
  let summary = "`scatter` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.scatter(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_ScatterAddOp : RMO_MOAnaloguePureScatterOp<"scatter.add"> {
  let summary = "`scatter-add` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.scatter.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_ScatterMaxOp : RMO_MOAnaloguePureScatterOp<"scatter.max"> {
  let summary = "scatter-max operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.scatter.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_ScatterMinOp : RMO_MOAnaloguePureScatterOp<"scatter.min"> {
  let summary = "`scatter-min` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.scatter.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_ScatterMulOp : RMO_MOAnaloguePureScatterOp<"scatter.mul"> {
  let summary = "scatter-mul operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.scatter.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_ConvOp : RMO_MOAnaloguePureTensorOp<"conv">{
  let summary = "Convolution operation";
  let description = [{
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    `strides`, `dilations`, and `padding` must be of rank 1, or unranked.
    If the input has static rank, all hyperparameters with static shape must
    have sizes of `input_rank - 2`, except padding, which must have
    size `2 * (input_rank - 2)`. Individual elements in the hyperparameters
    apply to corresponding dimensions of the input (after ignoring the batch
    and channel dimensions), with padding representing a before/after pair for
    each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here repesents H and dim2 represents W. In python like
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

    The input, output and filter tensors' ranks must match if statically known.

    `num_groups` must be a ranked scalar. The number of input and output
    channels must both be divisible by the number of groups `num_groups`.

    This op currently only supports strides and padding on the input.

    Example:

    ```mlir
      %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>}
        : %!mo.tensor<[], si64>
      %res = rmo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_IndicesTensor:$strides,
    MO_IndicesTensor:$dilations,
    MO_IndicesTensor:$paddings,
    MO_IndicesTensor:$num_groups,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)`
    `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
                    `paddings` `=` $paddings `,`
                    `num_groups` `=` $num_groups `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_ConvTransposeOp : RMO_MOAnaloguePureTensorOp<"conv_transpose">{
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
    ConvTranspose, dim1 here repesents H_out and dim2 represents W_out. In
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
      %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %op = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %res = rmo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_IndicesTensor:$strides,
    MO_IndicesTensor:$dilations,
    MO_IndicesTensor:$paddings,
    MO_IndicesTensor:$output_paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)`
    `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
        `paddings` `=` $paddings `,` `output_paddings` `=` $output_paddings`]` attr-dict `:`
    functional-type(operands, results)
  }];
}

class RMO_MOAnaloguePurePoolOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name,
    traits # [AllDTypesMatch<["input", "result"]>]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$filter_shape,
    MO_IndicesTensor:$strides,
    MO_IndicesTensor:$dilations,
    MO_IndicesTensor:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_AvgPoolOp : RMO_MOAnaloguePurePoolOp<"avg_pool">{
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. `strides`, `dilations`, `padding`) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$filter_shape,
    MO_IndicesTensor:$strides,
    MO_IndicesTensor:$dilations,
    MO_IndicesTensor:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls);
}

def RMO_AvgPoolCeilOp : RMO_MOAnaloguePurePoolOp<"avg_pool_ceil_mode_true">{
  let summary = "Ceiling-mode average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$filter_shape,
    MO_IndicesTensor:$strides,
    MO_IndicesTensor:$dilations,
    MO_IndicesTensor:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls);
}

def RMO_MaxPoolOp : RMO_MOAnaloguePurePoolOp<"max_pool">{
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];
}

def RMO_MaxPoolCeilOp : RMO_MOAnaloguePurePoolOp<"max_pool_ceil_mode_true">{
  let summary = "Ceiling-mode maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    The op supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
