//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines MO analogue operations in the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD

include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/RMODialect/RMOOps.td"
include "GenericML/GraphCompiler/RMODialect/RMODialect.td"
include "GenericML/GraphCompiler/RMODialect/RMOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOEnums.td"
include "mlir/IR/BuiltinAttributes.td"

// Base class for ops that map directly to a MO tensor op
class RMO_MOAnaloguePureTensorOp<string mnemonic, list<Trait> traits = []>
  : RMO_Op<"mo." # mnemonic, traits # [DeclareOpInterfaceMethods<RMO_MOAnalogue, ["inferMOTypes"]>, Pure]> {
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
// 1-1 MO Op Analogues
//===----------------------------------------------------------------------===//

class RMO_MOAnaloguePureElementWiseOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name, traits> {
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
          context, graphOp, cast<TensorValue>(operands.getInput()),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}

class RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureUnaryElementWiseOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits>;

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

        DType rhsDType = rhsType.getDtype();

        MO::TensorType refinedShape = refineShape(lhsType, rhsType);
        moOperandTypes.push_back(refinedShape);
        moOperandTypes.push_back(refinedShape.cloneWith(rhsDType));
        moInputParamDecls.push_back({});
        moInputParamDecls.push_back({});

        moResultTypes.push_back(refinedShape);
        moOutputParamDecls.push_back({});
    }
  }];
}

class RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureBinaryElementWiseOp<name, [RMO_TensorSameDTypeOperandsAndResults] # traits>;

def RMO_MOAbsOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"abs"> {
  let summary = "Absolute value operation";
  let description = [{
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOCeilOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"ceil"> {
  let summary = "Ceiling operation";
  let description = [{
    Returns the smallest largest integer greater than `x`, where `x` is input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOCosOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"cos"> {
  let summary = "Cosine operation";
  let description = [{
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOErfOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"erf"> {
  let summary = "(Gaussian) error function operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOExpOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"exp"> {
  let summary = "Exponential function operation";
  let description = [{
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOFloorOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"floor"> {
  let summary = "Floor operation";
  let description = [{
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOGeluOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"gelu"> {
  let summary = "Gaussian error linear unit operation";
  let description = [{
    Returns `(x / 2) * (1 + erf(x/sqrt(2)))`, where `erf` denotes the
    (Gaussian) error function.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOLogOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"log"> {
  let summary = "Natural logarithm operation";
  let description = [{
    Returns the natural logorithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOLog1pOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"log1p"> {
  let summary = "Natural logarithm of `1 + x` operation ";
  let description = [{
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOLogSoftmaxOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"logsoftmax"> {
  let summary = "Logarighmic softmax operation";
  let description = [{
    Returns `log(softmax(x))`, where `x` is input tensor.

    The softmax is applied along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MONegativeOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"negative"> {
  let summary = "Negation operation";
  let description = [{
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MONotOp : RMO_MOAnaloguePureTensorOp<"not"> {
  let summary = "Logical negation operation";
  let description = [{
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
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
          context, graphOp, cast<TensorValue>(operands.getInput()),
          rmoResults, moOperandTypes, moResultTypes, moInputParamDecls,
          moOutputParamDecls);
    }
  }];
}

def RMO_MOReluOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"relu"> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MORoundOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"round"> {
  let summary = "Rounding operation";
  let description = [{
    Returns the elementwise nearest integer, with ties going away from zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MORoundEvenOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"roundeven"> {
  let summary = "Round even operation";
  let description = [{
    Returns the elementwise neareset integer, with ties going toward the
    nearest even number.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.roundeven(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MORsqrtOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"rsqrt"> {
  let summary = "Inverse square root operation";
  let description = [{
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSinOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"sin"> {
  let summary = "Sine operation";
  let description = [{
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOSqrtOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"sqrt"> {
  let summary = "Square root operation";
  let description = [{
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTanhOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"tanh"> {
  let summary = "Hyperbolic tangent operation";
  let description = [{
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTruncOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"trunc"> {
  let summary = "Truncation operation";
  let description = [{
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOAddOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"add", [Commutative]> {
  let summary = "Binary addition";
  let description = [{
    Does a non-broadcasted elementwise addition.
  }];
}

def RMO_MOShapeOfOp : RMO_MOAnaloguePureTensorOp<"shape_of"> {
  let summary = "Get shape of tensor";
  let description = [{
    Calculates the runtime shape of the given tensor as a tensor.
  }];
  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_IndicesTensor:$shape);
}

def RMO_MOBroadcastShapeOp : RMO_MOAnaloguePureTensorOp<"broadcast_shape"> {
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

def RMO_MOReshapeOp : RMO_MOAnaloguePureTensorOp<"reshape"> {
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

def RMO_MOBroadcastToOp : RMO_MOAnaloguePureTensorOp<"broadcast_to"> {
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

def RMO_MOCastOp : RMO_MOAnaloguePureTensorOp<"cast"> {
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

def RMO_MOMulOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"mul", [Commutative]> {
  let summary = "Binary multiplication";
  let description = [{
    Does a non-broadcasted elementwise multiplication.
  }];
}

def RMO_MOModOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"mod"> {
  let summary = "Binary modulus op";
  let description = [{
    Does a non-broadcasted elementwise modulus.
  }];
}

def RMO_MOMinOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"min", [Commutative]> {
  let summary = "Binary minimum op";
  let description = [{
    Does a non-broadcasted elementwise minimum.
  }];
}

def RMO_MOMaxOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"max", [Commutative]> {
  let summary = "Binary maximum op";
  let description = [{
    Does a non-broadcasted elementwise maximum.
  }];
}

def RMO_MODivOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"div"> {
  let summary = "Binary division op";
  let description = [{
    Does a non-broadcasted elementwise division.
  }];
}

def RMO_MOPowOp : RMO_MOAnaloguePureBinaryElementWiseOp<"pow"> {
  let summary = "Binary power op";
  let description = [{
    Does a non-broadcasted elementwise power. Allows mixed precision operands.
  }];
}

def RMO_MOSubOp : RMO_MOAnaloguePureBinaryElementWiseMonoTypeOp<"sub"> {
  let summary = "Binary subtraction op";
  let description = [{
    Does a non-broadcasted elementwise subtraction.
  }];
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

def RMO_MOMeanOp : RMO_MOAnaloguePureReductionOp<"mean"> {
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
      %res = rmo.mo.mean(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def RMO_MOReduceMulOp : RMO_MOAnaloguePureReductionOp<"reduce.mul"> {
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
      %res = rmo.mo.reduce.mul(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def RMO_MOReduceAddOp : RMO_MOAnaloguePureReductionOp<"reduce.add"> {
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
      %res = rmo.mo.reduce.add(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_MOReduceMinOp : RMO_MOAnaloguePureReductionOp<"reduce.min"> {
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
      %res = rmo.mo.reduce.min(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_MOReduceMaxOp : RMO_MOAnaloguePureReductionOp<"reduce.max"> {
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
      %res = rmo.mo.reduce.max(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

class RMO_MOAnalogeArgMinMaxOp<string mnemonic> : RMO_MOAnaloguePureReductionOp<mnemonic,
  /*input_result_same_dtype=*/0> {
  let results = (outs MO_IndicesTensor:$result);
}

def MO_MOArgMaxOp : RMO_MOAnalogeArgMinMaxOp<"arg_max"> {
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
      %1 = rmo.mo.arg_max(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_MOArgMinOp : RMO_MOAnalogeArgMinMaxOp<"arg_min"> {
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
      %1 = rmo.mo.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
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
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getUpdates()),
          cast<TensorValue>(operands.getIndices()),
          cast<TensorValue>(operands.getAxis()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOScatterOp : RMO_MOAnaloguePureScatterOp<"scatter">{
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
      %res = rmo.mo.scatter(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterAddOp : RMO_MOAnaloguePureScatterOp<"scatter.add"> {
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
      %res = rmo.mo.scatter.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMaxOp : RMO_MOAnaloguePureScatterOp<"scatter.max"> {
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
      %res = rmo.mo.scatter.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMinOp : RMO_MOAnaloguePureScatterOp<"scatter.min"> {
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
      %res = rmo.mo.scatter.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOScatterMulOp : RMO_MOAnaloguePureScatterOp<"scatter.mul"> {
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
      %res = rmo.mo.scatter.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def RMO_MOConvOp : RMO_MOAnaloguePureTensorOp<"conv">{
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
      %res = rmo.mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
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

def RMO_MOConvTransposeOp : RMO_MOAnaloguePureTensorOp<"conv_transpose">{
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
      %res = rmo.mo.conv_transpose(%input, %filter)
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

def RMO_MOAvgPoolOp : RMO_MOAnaloguePurePoolOp<"avg_pool">{
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
      %res = rmo.mo.avg_pool(%arg) [
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

def RMO_MOAvgPoolCeilOp : RMO_MOAnaloguePurePoolOp<"avg_pool_ceil_mode_true">{
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
      %res = rmo.mo.avg_pool_ceil_mode_true(%arg) [
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

def RMO_MOMaxPoolOp : RMO_MOAnaloguePurePoolOp<"max_pool">{
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
      %res = rmo.mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];
}

def RMO_MOMaxPoolCeilOp : RMO_MOAnaloguePurePoolOp<"max_pool_ceil_mode_true">{
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
      %res = rmo.mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];
}

class RMO_MOAnaloguePureComparisonOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<name, traits> {
  let results = (outs MO_BooleanMask:$result);
}

class RMO_MOAnaloguePureBinaryComparisonOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureComparisonOp<name, traits #  [AllDTypesMatch<["input_x", "input_y"]>]> {
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

        moResultTypes.push_back(refinedShape.cloneWith(DType::kBool));
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_MOEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"equal", [Commutative]> {
  let summary = "Equality test operation";
  let description = [{
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGreaterOp : RMO_MOAnaloguePureBinaryComparisonOp<"greater"> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                          !mo.tensor<[2, 3], f32>
                                          ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGreaterEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"greater_equal"> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                                !mo.tensor<[2, 3], f32>
                                                ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MONotEqualOp : RMO_MOAnaloguePureBinaryComparisonOp<"not_equal", [Commutative]> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MORangeOp : RMO_MOAnaloguePureTensorOp<"range",
  [AllDTypesMatch<["start", "limit", "step", "result"]>]>
{
  let summary = "Range operation";
  let description = [{
    Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant {
        value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
      %step = mo.constant {
        value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
      %res = rmo.mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant {
        value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
      %stepInt = mo.constant {
        value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
      %limitInt = mo.constant {
        value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
      %oddNumbersBelowTen = rmo.mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$start,
    MO_Tensor:$limit,
    MO_Tensor:$step,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def RMO_MOResizeLinearOp : RMO_MOAnaloguePureTensorOp<"resize.linear"> {
  let summary = "Linear resize operation";
  let description = [{
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    When set to true, the antialias attribute causes an antialiasing filter to be applied
    when downscaling.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    BoolAttr:$antialias,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}


def RMO_MOResizeNearestOp : RMO_MOAnaloguePureTensorOp<"resize.nearest"> {
  let summary = "Nearest-neighbor resize operation";
  let description = [{
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    DefaultValuedAttr<I64Attr, "0">:$round_mode,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureLogicalBinaryOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureBinaryElementWiseOp<name, traits # [Commutative]> {
  let arguments = (ins MO_BooleanMask:$input_x, MO_BooleanMask:$input_y, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);
}

def RMO_MOAndOp : RMO_MOAnaloguePureLogicalBinaryOp<"and"> {
  let summary = "Conjunction operation";
  let description = [{
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOOrOp : RMO_MOAnaloguePureLogicalBinaryOp<"or"> {
  let summary = "Disjunction operation";
  let description = [{
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                     !mo.tensor<[2, 3], bool>
                                     ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOXorOp : RMO_MOAnaloguePureLogicalBinaryOp<"xor"> {
  let summary = "Exclusive disjunction operation";
  let description = [{
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOSelectOp : RMO_MOAnaloguePureTensorOp<"select",
  [AllDTypesMatch<["x", "y", "result"]>]> {
  let summary = "Select operation";
  let description = [{
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$cond, MO_Tensor:$x, MO_Tensor:$y, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureUnaryComparisonOp<string name, list<Trait> traits = []>
    : RMO_MOAnaloguePureComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_ParamDecls:$outputParamDecls);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
        MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
        std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
        SmallVectorImpl<Type> &moResultTypes,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
        SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {

        auto inputTy = cast<MO::TensorType>(operands.getInputX().getType());

        // Expected input type
        moOperandTypes.push_back(inputTy);
        moInputParamDecls.push_back({});

        // Expected output type
        moResultTypes.push_back(inputTy.cloneWith(DType::kBool));
        moOutputParamDecls.push_back({});
    }
  }];
}

def RMO_MOIsNaNOp : RMO_MOAnaloguePureUnaryComparisonOp<"is_nan"> {
  let summary = "NaN test operation";
  let description = [{
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOIsInfOp : RMO_MOAnaloguePureUnaryComparisonOp<"is_inf"> {
  let summary = "Infinity test operation";
  let description = [{
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def RMO_MOGatherOp : RMO_MOAnaloguePureTensorOp<"gather",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "`gather` operation";
  let description = [{
    Gathers slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.mo.gather(%input, %indices, %axis) : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesTensor:$indices,
                       MO_IndicesTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOGatherNdOp : RMO_MOAnaloguePureTensorOp<"gather_nd",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Multi-dimensional `gather` operation";
  let description = [{
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension stores the index whereas
    the outer dimensions act like batch dimensions. The size of the last
    dimension is at most the rank of the input. When the dimension size is less
    than the rank of the input, slices of the input are gathered, starting from
    the leftmost dimension.

    ```
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )
    ```

    ```mlir
      %input = mo.constant {value =
        #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> :
        tensor<2x2x4xsi64>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant {value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} :
        !mo.tensor<[3], si64>

      %result = rmo.mo.gather_nd(%input, %indices) {batchDims = 0} :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesTensor:$indices,
                       // TODO(#21324): Support dynamic batchDims
                       Builtin_IntegerAttr:$batchDims,
                       MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_MOMatmulOp : RMO_MOAnaloguePureTensorOp<"matmul"> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two 2D tensors.
    Example:

    ```mlir
      %lhs: ... !mo.tensor<[10, 20], f32>
      %rhs: ... !mo.tensor<[20, 5], f32>
      %res = mo.matmul(%lhs, %rhs) : (
        !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
      ) -> !mo.tensor<[10, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input_a,
    MO_Tensor:$input_b,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOBatchMatmulOp : RMO_MOAnaloguePureTensorOp<"batch_matmul"> {
  let summary = "Batched matrix-matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two batches of matrices, represented by
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
    OptionalAttr<BoolAttr>:$transpose_b,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);


  let builders = [
    /// Create matmul with default attribute flags off.
    OpBuilder<(ins "Type":$type, "Value":$input_a, "Value":$input_b)>,
    OpBuilder<(ins "Type":$type, "Value":$input_a, "Value":$input_b, "ArrayRef<KGEN::ParamDeclAttr>":$decls)>,
  ];
}

def RMO_MOSoftmaxOp : RMO_MOAnaloguePureUnaryElementWiseMonoTypeOp<"softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Returns `exp(input) / sum(exp(input))`, where `x` is input tensor.

    The `sum` reduction is applieed along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.softmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def RMO_MOTileOp : RMO_MOAnaloguePureTensorOp<"tile",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Tiling operation";
  let description = [{
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = rmo.mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$repeats,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOSqueezeShapeOp : RMO_MOAnaloguePureTensorOp<"squeeze_shape",
  [AllDTypesMatch<["input_shape", "result"]>]> {
  let summary = "Squeeze-shape operation";
  let description = [{
    Calculates the shape from squeeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N - M`. The
    indices represent the 0-based index of dimensions in the original rank `N`
    tensor.

    The indicated indices must represent dimensions of size 1. If
    an index does not point to a dimension to size 1, an error is thrown
    instead.

    This operator supports negative indexing with python-like semantics.
    That is all indices must be in [-N, N), if an index is < 0, it is as if
    we added `N` to it.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = rmo.mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLikeIfRanked:$input_shape,
    MO_ShapeLikeIfRanked:$remove_indices,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLikeIfRanked:$result);
}

def RMO_MOUnsqueezeShapeOp : RMO_MOAnaloguePureTensorOp<"unsqueeze_shape",
  [AllDTypesMatch<["input_shape", "result"]>]> {
  let summary = "Unsqueeze-shape operation";
  let description = [{
    Calculates the shape from unsqueeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N + M`.

    The indices in the given list map to the new vector of length `N + M` where
    the indicated dimensions are replaced with `1`. The remaining dimension in
    the original input shape vector are copied over in the non-1 dimensions.

    This operator supports negative indexing.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = rmo.mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLikeIfRanked:$input_shape,
    MO_ShapeLikeIfRanked:$padding_indices,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLikeIfRanked:$result);
}

def RMO_MOTransposeOp : RMO_MOAnaloguePureTensorOp<"transpose",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Transposition operation";
  let description = [{
    Returns a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$perm,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureTopKOp<string name> : RMO_MOAnaloguePureTensorOp<
  name, [AllDTypesMatch<["input", "values"]>]
> {
  let arguments = (ins MO_Tensor:$input,
                       MO_IndexScalarIfRanked:$K,
                       MO_IndexScalarIfRanked:$axis,
                       MO_CondLikeIfRanked:$sorted,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$values,
                      MO_I64Tensor:$indices);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesTopKImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getK()),
          cast<TensorValue>(operands.getAxis()),
          cast<TensorValue>(operands.getSorted()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOTopKOp : RMO_MOAnaloguePureTopKOp<"top_k"> {
  let summary = "Top-K operation";
  let description = [{
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis. Returned values along the axis are always sorted
    (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant() { value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant { value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant { value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = rmo.mo.top_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

def RMO_MOBottomKOp : RMO_MOAnaloguePureTopKOp<"bottom_k"> {
  let summary = "Bottom-K operation";
  let description = [{
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant { value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant { value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant { value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = rmo.mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

class RMO_MOAnaloguePureLinalgOp<string mnemonic, list<Trait> traits = []>
    : RMO_MOAnaloguePureTensorOp<"linalg." # mnemonic, traits> {
  string cppNamespace = RMODialect.cppNamespace # "::LinAlg";
}

def RMO_MOBandPartOp : RMO_MOAnaloguePureLinalgOp<"band_part",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Matrix band part operation";
  let description = [{
    Copies a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

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
      %exclude = mo.constant {
        value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
      %res = rmo.mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndexScalarIfRanked:$num_lower,
    MO_IndexScalarIfRanked:$num_upper,
    MO_CondLikeIfRanked:$exclude,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_MOSolveOp : RMO_MOAnaloguePureLinalgOp<"solve"> {
  let summary = "Linear solve operation";
  let description = [{
    Solves the system of linear equations described by

      `AX = B`

    Where input A is a batch of square matrices and has shape [I, J, ..., M, M]
    which are assumed to be full rank/invertible and rhs B is a batch of
    results of shape [I, J, ..., M, K]. The result, X, will be a tensor of
    shape [I, J, ..., M, K]

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 3, 3], f32>
      %rhs: !mo.tensor<[3, 3, 2], f32>
      %res = rmo.mo.linalg.solve(%arg, %rhs) : (
        !mo.tensor<[3, 3, 3], f32>, !mo.tensor<[3, 3, 2], f32>, !mo.tensor<[]
        ) -> !mo.tensor<[3, 3, 2], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$rhs,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_MOConcatOp : RMO_MOAnaloguePureTensorOp<"concat"> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor. The input tensors and output tensors all
    has the same shape except along the concatenation dimension `axis`. The size
    of the concatenation dimension in output tensor would have be the sum of
    sizes of the concatenation dimension in input tensors.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = rmo.mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$axis,
    Variadic<MO_Tensor>:$inputs,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def RMO_MOSplitOp : RMO_MOAnaloguePureTensorOp<"split"> {
  let summary = "Splitting operation";
  let description = [{
    Splits the input tensor into multiple tensors along a given dimension.

    `mo.split` splits the tensor `input` into multiple output tensors.
    The number of output tensors is equal to the number of elements in
    `splitSizes`, which is a rank-1 tensor of integers.
    Each of the output tensors has the same shape as `input` except along the
    split dimension `axis`, where the size is given by the corresponding
    element in `splitSizes`.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 8], f32>
      %splitSizes = mo.constant {
        value = #M.dense_array<3, 5> : tensor<2xsi64>
      } : !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>
      } : !mo.tensor<[], si64>
      %res = rmo.mo.split[%axis: !mo.tensor<[], si64>](%input, %splitSizes) : (
        !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
      ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$splitSizes,
    MO_IndicesTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $input `,` $splitSizes `)` attr-dict `:` `(`
      qualified(type($input)) `,` qualified(type($splitSizes))
    `)` `->` `(` type($results) `)`
  }];
}

def RMO_MOPadConstantOp : RMO_MOAnaloguePureTensorOp<"pad.constant",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Constant-mode padding operation";
  let description = [{
    Pads the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assumes input has rank `N`, the `paddings` tensor should have
    shape `(2 * N)`, where each consecutive pair of elements has the form
    `[before, after]`, indicating how many of `constant` to add before and after
    the contents of `input` in that dimension. The size of each dimension D of
    the padded output is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant {
        value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ScalarTensorIfRanked:$constant,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOPadReflectOp : RMO_MOAnaloguePureTensorOp<"pad.reflect",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Reflection-mode padding operation";
  let description = [{
    Pads the `input` tensor by reflecting it according to the `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOPadRepeatOp : RMO_MOAnaloguePureTensorOp<"pad.repeat",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Repeat-mode padding operation";
  let description = [{
    Pads the `input` tensor by repeating border values according to `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MOSliceOp : RMO_MOAnaloguePureTensorOp<"slice",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Slice operation";
  let description = [{
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesVectorIfRanked:$start,
                       MO_IndicesVectorIfRanked:$stop,
                       MO_IndicesVectorIfRanked:$step,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

def RMO_MORoiAlignOp : RMO_MOAnaloguePureTensorOp<"roi_align"> {
  let summary = "ROI alignment operation";
  let description = [{
    ROI align consumes an input tensor and regions of interest in which to apply pooling.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant {value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant {value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant {value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

      %res = rmo.mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
        {aligned = false,  mode = "AVG"}
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```

  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$rois,
    MO_IndexScalarIfRanked:$output_height,
    MO_IndexScalarIfRanked:$output_width,
    MO_FloatScalarIfRanked:$spatial_scale,
    MO_FloatScalarIfRanked:$sampling_ratio,
    BoolAttr:$aligned,
    Builtin_StringAttr:$mode,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

def RMO_MORandomNormalOp : RMO_MOAnaloguePureTensorOp<"random.normal",
  [AllDTypesMatch<["mean", "variance"]>]> {
  let summary = "Normal distribution operation";
  let description = [{
    Returns a tensor with shape `shape` populated with random
    values from a normal distribution, with the mean of the distribution equal
    to `mean` and the standard deviation equal to `variance`.

  Example:
    ```mlir
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %mean = mo.constant {
        value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
      %variance = mo.constant {
        value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
      %seed = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64> } : !mo.tensor<[], si64>
      %res = rmo.mo.random.normal(%size, %mean, %variance, %seed) :
            (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
            !mo.tensor<[], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesVectorIfRanked:$shape,
    MO_FloatScalarIfRanked:$mean,
    MO_FloatScalarIfRanked:$variance,
    MO_IndexScalarIfRanked:$seed,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);
}

class RMO_MOAnaloguePureScatterNdOp<string name> : RMO_MOAnaloguePureTensorOp<
  name, [AllDTypesMatch<["input", "updates", "result"]>]
> {
let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let extraClassDeclaration = [{
    static void inferMOTypesImpl(
      MLIRContext *context, MO::GraphOp graphOp, Adaptor operands,
      std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
      SmallVectorImpl<Type> &moResultTypes,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
      SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls) {
        inferMOTypesScatterNdImpl(
          context, graphOp, cast<TensorValue>(operands.getInput()),
          cast<TensorValue>(operands.getUpdates()),
          cast<TensorValue>(operands.getIndices()), rmoResults, moOperandTypes,
          moResultTypes, moInputParamDecls, moOutputParamDecls);
      }
  }];
}

def RMO_MOScatterNdOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd"> {
  let summary = "Multi-dimensional `scatter` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdAddOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.add"> {
  let summary = "Multi-dimensional `scatter-add` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMaxOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.max"> {
  let summary = "Multi-dimensional `scatter-max` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMinOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.min"> {
  let summary = "Multi-dimensional `scatter-min` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MOScatterNdMulOp : RMO_MOAnaloguePureScatterNdOp<"scatter_nd.mul"> {
  let summary = "Multi-dimensional `scatter-mul` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def RMO_MONonMaximumSuppressionOp : RMO_MOAnaloguePureTensorOp<"non_maximum_suppression"> {
  let summary = "Non-max suppression operation";
  let description = [{
   Filters out boxes that have high intersection-over-union (IOU).

   `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
   coordinates of any diagonal pair of box corners and the coordinates can be
   provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

    Example:

    ```mlir
      %boxes : !mo.tensor<[1, 6, 4], f32>
      %scores : !mo.tensor<[1, 1, 6], f32>
      %maxOutputBoxesPerClass : !mo.tensor<[], si64>
      %iouThreshold : !mo.tensor<[], f32>
      %scoreThreshold : !mo.tensor<[], f32>
      %res = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold)
        : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
        -> !mo.tensor<[?, ?], si64>
    ```
  }];
  let arguments = (ins MO_Tensor:$boxes,
                       MO_Tensor:$scores,
                       MO_IndexScalarIfRanked:$maxOutputBoxesPerClass,
                       MO_FloatScalarIfRanked:$iouThreshold,
                       MO_FloatScalarIfRanked:$scoreThreshold,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_IndicesTensor:$output);
}

def RMO_MOArgNonzeroOp : RMO_MOAnaloguePureTensorOp<"arg_nonzero"> {
  let summary = "Nonzero-value indices operation";
  let description = [{
    Returns a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
      } : !mo.tensor<[3, 3], si32>
      %1 = rmo.mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si64>
    ```
    }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_I64Tensor:$result);
}

def RMO_MOCumSumOp: RMO_MOAnaloguePureTensorOp<"cumsum",
  [AllDTypesMatch<["input", "result"]>]> {
  let summary = "Cumulative summation operation";
  let description = [{
    Returns the cumulative summation of input tensors along an axis. By default,
    it copies the first element as is. If the `exclusive` attribute is set to 1,
    then the first element is excluded. The `reverse` attribute causes the
    summation to be done in the opposite direction of the axis.

    The value of `axis` follows numpy sematics, e.g., -1 represents the last
    axis.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis: !mo.tensor<[], i64>
    %res = rmo.mo.cumsum(%arg, %axis) {exclusive = 1 : i64, reverse = 0 : bool} : (
      !mo.tensor<[2, 3], f32>., !mo.tensor<[], i64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_IndexScalarIfRanked:$axis,
                       Builtin_IntegerAttr:$exclusive,
                       Builtin_IntegerAttr:$reverse,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOMOANALOGUEOPS_TD
