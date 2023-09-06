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

include "KGEN/KGENDialect/KGENInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def MO_TensorOpInterface : OpInterface<"MOTensorOpInterface"> {
  let description = [{
    Interface that all MO ops should implement, mainly providing typed accessors
    to inputs, outputs, and their shapes.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get an input tensor as a TypedValue",
      "TensorValue", "getInputTensor",
      (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TensorValue>($_op->getOperand(idx)); }]>,
    InterfaceMethod<"Get an output tensor as a TypedValue",
      "TensorValue", "getOutputTensor",
      (ins "unsigned int":$idx), [{}],
      [{ return ::llvm::cast<TensorValue>($_op->getResult(idx)); }]>
  ];
}

/// MO interfaces should inherit from this to ensure they are in the right
/// namespace and have the right dependencies.
class MO_TensorOpInterfaceBase<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces> {
  let cppNamespace = "::M::MO";
  let dependentTraits = [MO_TensorOpInterface];
}

def MO_ControlOpInterface : OpInterface<"MOControlOpInterface"> {
  let description = [{
    Interface marking ops that are control flow
  }];
  let cppNamespace = "::M::MO";
}

def MO_Parameterization
    : MO_TensorOpInterfaceBase<"ParameterizationInterface"> {
  let description = [{
    Interface to be implemented by ops that support resolving unknown shape and
    dimension parameters in their tensors.
  }];

  let methods = [
    InterfaceMethod<"Rewrites the IR to resolve unknown shapes and dims.",
      "LogicalResult", "resolveUnknownParameters",
      (ins "mlir::RewriterBase &":$rewriter)>,
  ];
}

def MO_ParamDeclaration : MO_TensorOpInterfaceBase<"ParamDeclarationInterface",
  [MemoryEffectsOpInterface, ParamOpInterface]> {
  let description = [{
    Interface to be implemented by ops that declare shape or dimension
    parameters.
  }];

  let methods = [
    InterfaceMethod<
      "Return the output shape or dimension parameters declared by this op",
      "llvm::ArrayRef<KGEN::ParamDeclAttr>", "getOutputParamDecls",
      (ins), [{}], [{ return {}; }]>,
    InterfaceMethod<
      "Set the output parameter declarations associated with this op",
      "void", "setOutputParamDecls",
      (ins "ArrayRef<KGEN::ParamDeclAttr>":$paramDecls), [{}],
      [{ assert(paramDecls.empty()); }]>,
  ];

  let extraTraitClassDeclaration = [{
    void getEffects(
      llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      if (!paramDeclItf.getOutputParamDecls().empty())
        effects.emplace_back(mlir::MemoryEffects::Allocate::get());
    }

    void walkDeclarations(function_ref<void(KGEN::ParamDeclAttr)> walkDecl) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      llvm::for_each(paramDeclItf.getOutputParamDecls(), walkDecl);
    }

    void walkDefinitions(function_ref<void(KGEN::ParamDeclAttr,
        const KGEN::ParamDefValue &)> walkDef) {
      mlir::AttrTypeWalker walker;
      SmallVector<Attribute> refs;
      walker.addWalk([&](KGEN::ParamDeclRefAttr declRef) {
        refs.emplace_back(declRef);
      });
      Operation *op = this->getOperation();
      for (auto t : op->getOperandTypes())
        walker.walk(t);

      auto paramDeclItf = cast<ParamDeclarationInterface>(op);
      KGEN::ParamDefValue deps(refs, {});
      for (KGEN::ParamDeclAttr decl : paramDeclItf.getOutputParamDecls())
        walkDef(decl, deps);
    }

    void renameDeclarations(ArrayRef<KGEN::ParamDeclAttr> decls) {
      auto paramDeclItf = cast<ParamDeclarationInterface>(this->getOperation());
      assert(decls.size() == paramDeclItf.getOutputParamDecls().size());
      paramDeclItf.setOutputParamDecls(decls);
    }
  }];
}

def MO_ShapeInferenceMaterialization :
    MO_TensorOpInterfaceBase<"ShapeInferenceMaterialization"> {
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
    (ins "M::MO::ShapeFuncMaterializationState &":$state)>,
  ];
}

def MO_MatmulLike : MO_TensorOpInterfaceBase<"MatmulLike"> {
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
      "TensorValue", "getTensorA", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TensorValue", "getTensorB", (ins), [{}],
      [{ return $_op.getInputTensor(1); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getTensorResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>
  ];

  let verify = [{ return impl::verifyMatmulLike(cast<MatmulLike>($_op)); }];
}

def MO_SlidingWindow : MO_TensorOpInterfaceBase<"SlidingWindow"> {
  let description = [{
    Interface for modeling operations that have sliding window semantics.
  }];

  let methods = [
    InterfaceMethod<"Get the strides as a TypedValue",
      "TensorValue", "getStrides", (ins)>,
    InterfaceMethod<"Get the dilations as a TypedValue",
      "TensorValue", "getDilations", (ins)>, 
    InterfaceMethod<"Get the paddings as a TypedValue",
      "TensorValue", "getPaddings", (ins)>
  ];

  let extraClassDeclaration = [{
    /// Helper for verifying window op shape/rank invariants.
    LogicalResult verifyWindowParamShape(
      TensorValue $windowParam, StringRef name);
    LogicalResult verifyWindowParamSizes(ArrayRef<TensorValue> values);
  }];

  let verify = [{ return impl::verifySlidingWindow(cast<SlidingWindow>($_op)); }];
}

def MO_SinkStaticBroadcastTo :
    MO_TensorOpInterfaceBase<"SinkStaticBroadcastToInterface"> {
  let description = [{
    Interface for ops that can commute with `mo.static.broadcast_to`.
  }];

  let methods = [
    InterfaceMethod<
      "Check if the op is preceded by `mo.static.broadcast_to`, and sink the "
      "broadcast below the op if possible. Implementers can assume that "
      "insertion point is set to the op when the interface method is called "
      "and they should not erase the preceding `mo.static.broadcast_to` "
      "operations.",
      "ErrorOrSuccess", "sinkStaticBroadcastTo",
      (ins "mlir::RewriterBase &":$rewriter)>,
  ];
}

/// Should not be used outside this file.
class _MO_ElementWise<string name, list<Interface> baseInterfaces = []> :
    MO_TensorOpInterfaceBase<name,
                             baseInterfaces # [MO_SinkStaticBroadcastTo]> {
  let extraTraitClassDeclaration = [{
    ErrorOrSuccess sinkStaticBroadcastTo(mlir::RewriterBase &rewriter) {
      return impl::elementwiseSinkStaticBroadcastTo($_op, rewriter);
    }
  }];
}

def MO_ElementWiseUnary : _MO_ElementWise<"ElementWiseUnary"> {
  let description = [{
    Interface for modeling unary element-wise operations.
  }];

  let methods = [
    InterfaceMethod<"Get the input tensor as a TypedValue",
      "TensorValue", "getInput", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>,
    InterfaceMethod<"Set the input tensor to given Value",
      "void", "setInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(0, val); }]>,
  ];
}

def MO_ElementWiseBinary : _MO_ElementWise<"ElementWiseBinary"> {
  let description = [{
    Interface for modeling binary element-wise operations.
  }];

  let methods = [
    InterfaceMethod<"Get the left input tensor as a TypedValue",
      "TensorValue", "getLhsInput", (ins), [{}],
      [{ return $_op.getInputTensor(0); }]>,
    InterfaceMethod<"Get the right input tensor as a TypedValue",
      "TensorValue", "getRhsInput", (ins), [{}],
      [{ return $_op.getInputTensor(1); }]>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getOutputTensor(0)); }]>,
    InterfaceMethod<"Set the left input tensor to given Value",
      "void", "setLhsInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(0, val); }]>,
    InterfaceMethod<"Set the right input tensor to given Value",
      "void", "setRhsInput", (ins "mlir::Value":$val), [{}],
      [{ $_op->setOperand(1, val); }]>,
  ];
}

def MO_Reduction : MO_TensorOpInterfaceBase<"Reduction",
    [MO_ParamDeclaration]> {
  let description = [{
    Interface for modeling reduction operations.
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "MutableOperandRange", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the axis tensor as a TypedValue",
      "TensorValue", "getAxis", (ins)>,
    InterfaceMethod<"Get the axis tensor as a MutableOperandRange",
      "MutableOperandRange", "getAxisMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return impl::verifyReduction(cast<Reduction>($_op)); }];
}

def MO_DefaultParameterization : MO_TensorOpInterfaceBase<
  "DefaultParameterization", [MO_Parameterization, MO_ParamDeclaration]
> {
  let description = [{
    Interface providing a default (naive) parameterization that is appropriate
    for ops without strict invariants on arguments and results (e.g. shape
    and/or dimension equalities). The interface can only handle ops with a
    single result, and ignores all arguments/results that are not MO tensors.
  }];

  let extraTraitClassDeclaration = [{
    LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
      auto paramInterface =
        mlir::cast<ParameterizationInterface>(this->getOperation());
      return impl::defaultResolveUnknownParameters(paramInterface, rewriter);
    }
  }];
}

def MO_SameInputOutputShapesParameterization : MO_TensorOpInterfaceBase<
  "SameInputOutputShapesParameterization",
  [MO_Parameterization, MO_ParamDeclaration]
> {
  let description = [{
    Interface providing a default parameterization for ops that should have the
    same input and output shapes.
  }];

  let extraTraitClassDeclaration = [{
    LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
      auto paramInterface =
        mlir::cast<ParameterizationInterface>(this->getOperation());
      return impl::resolveSameOperandsAndResultShapeUnknownParameters(
        paramInterface, rewriter);
    }
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
}

def MO_SameOperandsAndResultType : TraitList<[
  SameOperandsAndResultType,
  MO_SameInputOutputShapesParameterization
]>;

/// Logically similar to the builtin SameOperandsAndResultShape, but uses
/// MOSH::ShapeAttr which is used to store the MO::Tensor shape.
def MO_SameOperandsAndResultShapeAttr :
    MO_OpTrait<"MOSameOperandsAndResultShapeAttr">;

def MO_SameOperandsAndResultShape: TraitList<[
  MO_SameOperandsAndResultShapeAttr,
  MO_SameInputOutputShapesParameterization
]>;

/// Should not be used outside this file.
def _MO_ReductionShapeInferenceMaterialization
  : MO_OpTrait<"ReductionShapeInferenceMaterialization">;

/// All traits that reduction ops should have.
def MO_ReductionTraits: TraitList<[
  MO_Reduction,
  MO_ShapeInferenceMaterialization,
  MO_DefaultParameterization,
  _MO_ReductionShapeInferenceMaterialization
]>;

/// Logically similar to the builtin SameOperandsAndResultElementType, but this
/// allows us to use our own logic of how the element types are accessed from
/// the op's operands and results. In the builtin implementation, element types
/// are only support for mlir::ShapedType. Moreover, if `MO::TensorType` needs
/// to support element type parameters in the future, this trait would be able
/// to support them.
def MO_SameOperandsAndResultElementType :
    MO_OpTrait<"MOSameOperandsAndResultElementType">;

def MO_SameOperandsElementType :
    MO_OpTrait<"MOSameOperandsElementType">;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
