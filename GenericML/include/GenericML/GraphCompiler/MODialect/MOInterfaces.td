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

def MO_OpInterface : OpInterface<"MOOpInterface"> {
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

/// MO interfaces should inherit from this to ensure they are in the right
/// namespace and have the right dependencies.
class MO_OpInterfaceBase<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces> {
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_OpInterface];
}

def MO_Parameterization : MO_OpInterfaceBase<"ParameterizationInterface"> {
  let description = [{
    Interface to be implemented by ops that support resolving unknown shape
    and dimension parameters in their tensors.
  }];

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

def MO_ShapeInferenceMaterialization :
    MO_OpInterfaceBase<"ShapeInferenceMaterialization"> {
  let description = [{
    Shape inference materialization interface.

    Implementers of this interface are responsible for materializing the shape
    inference functions as IR in the module containing the op, as well as gluing
    in the appropriate invocations in the graph.
  }];

  let methods = [
    InterfaceMethod<
      "Materializes and invokes the shape inference functions for the op.",
      "LogicalResult", "materializeShapeInference",
      (ins "mlir::RewriterBase &":$rewriter)>,
  ];
}

def MO_MatmulLike : MO_OpInterfaceBase<"MatmulLike"> {
  let description = [{
    Interface for modeling operations that implement matmul-like behavior,
    including vanilla matmul and batchmatmul.

    Dynamically ranked/shaped inputs and results are permitted, but if the shape
    is known, the last 2 dimensions of each input are assumed to be the matrix
    dimension. This interface refers to the left and right matrices as A and B,
    respectively.
  }];

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

def MO_SlidingWindow : MO_OpInterfaceBase<"SlidingWindow"> {
  let description = [{
    Interface for modeling operations that have sliding window semantics.
  }];

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

def MO_ElementWiseUnary : MO_OpInterfaceBase<"ElementWiseUnary"> {
  let description = [{
    Interface for modeling unary element-wise operations.
  }];
}

def MO_ElementWiseBinary : MO_OpInterfaceBase<"ElementWiseBinary"> {
  let description = [{
    Interface for modeling binary element-wise operations.
  }];
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

/// MO traits should inherit from this to ensure they are in the right namespace
/// and have the right dependencies.
class MO_OpTrait<string name, list<Trait> traits = []>
    : NativeOpTrait<name, traits> {
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_OpInterface];
}

/// Should not be used outside this file.
def _MO_SameInputOutputTypesParameterization :
    MO_OpTrait<"SameInputOutputTypesParameterization">;

def MO_SameOperandsAndResultType : TraitList<[
  SameOperandsAndResultType,
  MO_Parameterization,
  _MO_SameInputOutputTypesParameterization,
]>;

/// Should not be used outside this file.
def _MO_ReductionParameterization : MO_OpTrait<"ReductionParameterization">;

def MO_ReductionParameterization : TraitList<[
  MO_Parameterization, _MO_ReductionParameterization]>;

/// Logically similar to the builtin SameOperandsAndResultElementType, but this
/// allows us to use our own logic of how the element types are accessed from
/// the op's operands and results. In the builtin implementation, element types
/// are only support for mlir::ShapedType. Moreover, if `MO::TensorType` needs
/// to support element type parameters in the future, this trait would be able
/// to support them.
def MO_SameOperandsAndResultElementType :
    MO_OpTrait<"MOSameOperandsAndResultElementType">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
