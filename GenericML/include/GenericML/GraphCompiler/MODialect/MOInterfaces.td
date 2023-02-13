//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

def MO_MOOpInterface : OpInterface<"MOOpInterface"> {
  let description = [{
    Interface that all MO ops should implement, mainly providing typed accessors
    to inputs, outputs, and their shapes.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get an input tensor as a TypedValue",
      "TypedValue<TensorType>", "getInputTensor",
      (ins "unsigned int":$idx), [{}],
      [{
        return ::llvm::cast<TypedValue<TensorType>>($_op->getOperand(idx));
      }]>,
    InterfaceMethod<"Get an output tensor as a TypedValue",
      "TypedValue<TensorType>", "getOutputTensor",
      (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TypedValue<TensorType>>($_op->getResult(idx)); }]>
  ];
}

def MO_Parameterization : OpInterface<"MOOpParameterizationInterface"> {
  let description = [{
    Interface to be implemented by ops that support resolving unknown shape
    and dimension parameters in their tensors.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];

  let methods = [
    // TODO: provide default implementation
    InterfaceMethod<"Rewrites the IR to resolve unknown shapes and dims.",
      "LogicalResult", "resolveUnknownParameters",
      (ins "mlir::RewriterBase &":$rewriter)>,
  ];

  let extraClassDeclaration = [{
  private:
    static size_t numShapeParams;
    static size_t numDimParam;
  public:
    /// Convenience method to generate a new shape parameter declaration.
    KGEN::ParamDeclAttr getNewShapeParam();

    /// Convenience method to generate a new dimension parameter declaration.
    KGEN::ParamDeclAttr getNewDimParam();
  }];
}

def MO_ForwardShapeInferenceMaterialization :
    OpInterface<"MOForwardShapeInferenceMaterialization"> {
  let description = [{
    Forward shape inference materialization interface.

    Implementers of this interface are responsible for materializing the shape
    inference function as IR in the module containing the op, as well as gluing
    in the appropriate invocation in the graph.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];

  let methods = [
    InterfaceMethod<
      "Materializes and invokes the shape inference function for the op.",
      "LogicalResult", "materializeForwardShapeInference",
      (ins "mlir::RewriterBase &":$rewriter)>,
  ];
}

def MO_MatmulLike : OpInterface<"MatmulLike"> {
  let description = [{
    Interface for modeling operations that implement matmul-like behavior,
    including vanilla matmul and batchmatmul.

    Dynamically ranked/shaped inputs and results are permitted, but if the shape
    is known, the last 2 dimensions of each input are assumed to be the matrix
    dimension. This interface refers to the left and right matrices as A and B,
    respectively.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];

  let methods = [
    InterfaceMethod<"Get the left input tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorA", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorB", (ins), [{}],
      [{ return $_op.getInputTensor(1); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TypedValue<TensorType>", "getTensorResult", (ins), [{}],
      [{
        return ::llvm::cast<TypedValue<TensorType>>($_op.getOutputTensor(0));
      }]>
  ];

  let verify = [{ return impl::verifyMatmulLike($_op); }];
}

def MO_SlidingWindow : OpInterface<"SlidingWindow"> {
  let description = [{
    Interface for modeling operations that have sliding window semantics.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];

  let methods = [
    InterfaceMethod<"Get the strides as a TypedValue",
      "TypedValue<TensorType>", "getStrides", (ins)>,
    InterfaceMethod<"Get the dilations as a TypedValue",
      "TypedValue<TensorType>", "getDilations", (ins)>
  ];

  let extraClassDeclaration = [{
    /// Helper for verifying window op shape/rank invariants.
    LogicalResult verifyWindowParamShape(
      TypedValue<::M::MO::TensorType> $windowParam, StringRef name);
    LogicalResult verifyWindowParamSizes(
      ArrayRef<TypedValue<M::MO::TensorType>> values);
  }];

  let verify = [{ return impl::verifySlidingWindow($_op); }];
}

def MO_ElementWiseUnary : OpInterface<"ElementWiseUnary"> {
  let description = [{
    Interface for modeling unary element-wise operations.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];
}

def MO_ElementWiseBinary : OpInterface<"ElementWiseBinary"> {
  let description = [{
    Interface for modeling binary element-wise operations.
  }];
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_MOOpInterface];
}

def MO_SameInputOutputTypesParameterization :
    NativeOpTrait<"SameInputOutputTypesParameterization"> {
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_Parameterization];
}

def MO_SameOperandsAndResultType : TraitList<[
  SameOperandsAndResultType,
  MO_Parameterization,
  MO_SameInputOutputTypesParameterization,
]>;

/// Logically similar to the builtin SameOperandsAndResultElementType, but this
/// allows us to use our own logic of how the element types are accessed from
/// the op's operands and results. In the builtin implementation, element types
/// are only support for mlir::ShapedType. Moreover, if `MO::TensorType` needs
/// to support element type parameters in the future, this trait would be able
/// to support them.
def MO_SameOperandsAndResultElementType :
    NativeOpTrait<"MOSameOperandsAndResultElementType"> {
  let cppNamespace = "::M::MO";
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
