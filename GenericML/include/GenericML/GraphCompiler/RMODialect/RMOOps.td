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
include "GenericML/GraphCompiler/MODialect/MOEnums.td"
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

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$output);

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input, "ShapeAttr":$newShape, CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls)>
  ];

  // If we can prove the rebind is rebinding to the same shape, become no-op
  let hasFolder = 1;
}


class RMO_PureBinaryElementWiseOp<string name, list<Trait> traits = []>
    : RMO_Op<name, traits # [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs);
  }];
  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                            ValueRange operands, DictionaryAttr attributes,
                            mlir::OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryElemwiseImpl<$cppClass>(
          context, location, operands, attributes, properties, regions,
          inferredReturnTypes);
    }

    // InferTypeOpInterface:
    bool $cppClass::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
      return isCompatibleReturnTypesBinaryElemwiseImpl(lhs, rhs);
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

// TODO: investigate pow. It looks like the MO op allows for different input types without casting.
// For example, raising a float to the power of an integer.
// We may need to expand the analogue op and this op to support that as well.
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
    : RMO_Op<name, traits # [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins
    MO_Tensor:$input_x,
    MO_Tensor:$input_y,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_BooleanMask:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];

  let extraClassDeclaration = [{
    // InferTypeOpInterface:
    static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs);
  }];
  let extraClassDefinition = [{
    LogicalResult
    $cppClass::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                            ValueRange operands, DictionaryAttr attributes,
                            mlir::OpaqueProperties properties, RegionRange regions,
                            SmallVectorImpl<Type> &inferredReturnTypes) {
      return inferReturnTypesBinaryComparisonImpl<$cppClass>(
          context, location, operands, attributes, properties, regions,
          inferredReturnTypes);
    }

    // InferTypeOpInterface:
    bool $cppClass::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
      return isCompatibleReturnTypesBinaryComparisonImpl(lhs, rhs);
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
  let summary = "Flexible greate than comparison op";
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

  let builders = [
    /// Inferred return type builder
    OpBuilder<(ins "TensorValue":$input)>,
  ];
}

def RMO_ConcatOp : RMO_Op<"concat", [
  Pure,
  // TODO(GRA-551) remove `refineReturnTypes` once we eliminate `?` in Graph API.
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["refineReturnTypes"]>,
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
      Variadic<MO_Tensor>:$inputs,
      Builtin_IntegerAttr:$axis,
      MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_TD
