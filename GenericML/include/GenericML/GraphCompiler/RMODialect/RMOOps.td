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

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
