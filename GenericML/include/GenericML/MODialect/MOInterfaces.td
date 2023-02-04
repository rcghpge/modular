//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOINTERFACES_TD
#define GENERICML_MODIALECT_MOINTERFACES_TD

include "mlir/IR/OpBase.td"

def MO_MOOpInterface : OpInterface<"MOOpInterface"> {
  let description = [{
    Interface that all MO ops should implement, mainly providing typed accessors
    to inputs, outputs, and their shapes.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get an input tensor as a TypedValue",
      "TypedValue<TensorType>", "getInputTensor", (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TypedValue<TensorType>>($_op->getOperand(idx)); }]>,
    InterfaceMethod<"Get the shape of an input tensor",
      "ArrayRef<int64_t>", "getInputTensorShape", (ins "unsigned int":$idx), [{}],
      [{ return $_op.getInputTensor(idx).getType().getShape(); }]>,
    InterfaceMethod<"Get an output tensor as a TypedValue",
      "TypedValue<TensorType>", "getOutputTensor", (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TypedValue<TensorType>>($_op->getResult(idx)); }]>,
    InterfaceMethod<"Get the shape of an output tensor",
      "ArrayRef<int64_t>", "getOutputTensorShape", (ins "unsigned int":$idx), [{}],
      [{ return $_op.getOutputTensor(idx).getType().getShape(); }]>,
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
      [{ return ::llvm::cast<TypedValue<TensorType>>($_op.getOutputTensor(0)); }]>,
    InterfaceMethod<"Get the shape of the left input tensor",
      "ArrayRef<int64_t>", "getTensorAShape", (ins), [{}],
      [{ return $_op.getInputTensorShape(0); }]>,
    InterfaceMethod<"Get the shape of the right input tensor",
      "ArrayRef<int64_t>", "getTensorBShape", (ins), [{}],
      [{ return $_op.getInputTensorShape(1); }]>,
    InterfaceMethod<"Get the shape of the result tensor",
      "ArrayRef<int64_t>", "getTensorResultShape", (ins), [{}],
      [{ return $_op.getTensorResult().getType().getShape(); }]>,
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


#endif // GENERICML_MODIALECT_MOINTERFACES_TD
