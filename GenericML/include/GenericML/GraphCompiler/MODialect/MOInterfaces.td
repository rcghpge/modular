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

def MO_ConditionallyInPlaceInterface : OpInterface<"MOConditionallyInPlaceInterface"> {
  let description = [{
    Interface that ops that can conditionally represent an in-place computation
    (e.g. a custom op that directly operates on a mo.buffer value or a
     mogg.kernel after it has been load and store fused).

    Should be used in conjunction with MOMutableOpInterface.
  }];

  let methods = [
    InterfaceMethod<"Query if this op currently represents an in-place computation.",
                    "bool", "isInPlace", (ins)>,
  ];

  let cppNamespace = "::M::MO";
}

def MO_MutableOpInterface : OpInterface<"MOMutableOpInterface", []> {
  let description = [{
    Interface that all mutable ops under rmo/mo implement and any ops that
    could represent in-place compute should implement as well.

    In the case where an op can be conditionally in-place (e.g. mo.custom) it
    should also implement the MOConditionallyInPlaceInterface as well.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Get the input chain as a ChainValue",
      "llvm::SmallVector<::M::MO::ChainValue>",
      "getInChains", (ins),
      [{}],
      [{ return {$_op.getInChain()}; }]>,

    InterfaceMethod<"Get the output chain as a ChainValue",
      "llvm::SmallVector<::M::MO::ChainValue>",
      "getOutChains", (ins),
      [{}],
      [{ return {$_op.getOutChain()}; }]>,

    InterfaceMethod<"Get the input chain as a OpOperand ref",
      "llvm::SmallVector<::mlir::OpOperand *>",
      "getInChainsMutable", (ins),
      [{}],
      [{ return {&($_op.getInChainMutable())}; }]>
  ];

  let extraClassDeclaration = [{
    std::optional<::M::MO::ChainValue> getUniqueInChain() {
      auto chains = getInChains();
      if (chains.size() != 1)
        return {};
      return chains[0];
    }

    std::optional<::M::MO::ChainValue> getUniqueOutChain() {
      auto chains = getOutChains();
      if (chains.size() != 1)
        return {};
      return chains[0];
    }

    ::mlir::OpOperand *getUniqueInChainMutable() {
      auto chains = getInChainsMutable();
      if (chains.size() != 1)
        return nullptr;
      return chains[0];
    }
  }];
}

def MO_HasDeviceInterface : OpInterface<"MOHasDeviceInterface"> {
  let description = [{
    Interface for ops with an optional `#M.device_ref` attribute describing the
    intended execution device. The reference must resolve to an `#M.device_spec`
    in the op's containing `mo.graph` 'device_specs' attribute.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<
      [{
        Returns an array of `#M.device_ref` corresponding to the
        devices on which this operation executes.
      }],
      "llvm::SmallVector<DeviceRefAttr>", "getExecutionDevices", (ins)
    >,

    InterfaceMethod<
      [{
        Returns a `#M.device_ref` to the device where this operation executes.
        Assumes that this is not a multi-device operation.
      }],
      "DeviceRefAttr", "getExecutionDevice", (ins), "",
      [{
        auto devices = $_op.getExecutionDevices();
        assert(devices.size() == 1 && "Operation must execute on exactly one device");
        return devices[0];
      }]
    >,
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

def MO_Staticization : OpInterface<"Staticization"> {
  let description = [{
    Interface that models op which, if given staticized versions of its inputs,
    might be able to get staticized as well, where "being staticized" means
    having the op's output value represented as a parameter expression, i.e.,
    one of:
    - `IntegerAttr` for a integer or boolean constant, e.g., `42`.
    - `BoolAttr` for a boolean constant, e.g., `true`.
    - `KGEN::ParamDeclRefAttr` for a parameter reference, e.g., `D0`.
    - `KGEN::ParamOperatorAttr` for a parameter expression, e.g., `add(D1, 2)`.
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<"Try to staticize this op given staticized input values. "
    "The return value is either an error (for fatal error to be reported), "
    "a null `TypedAttr` (when staticization cannot happen due to input being "
    "too dynamic), or the result of staticization. The `ParamExprBuilder` is "
    " used to build new expression or assertions.",
      "ErrorOr<TypedAttr>", "tryStaticize",
      (ins "ArrayRef<TypedAttr>":$inputs,
           "ParamExprBuilder&":$b),
      [{}]>,
  ];
}

def MO_ShapeMaterialization : MO_TensorOpInterfaceBase<"ShapeMaterialization", [MO_ParamDeclaration]> {
  let description = [{
    Interface that models ops which
    1. declare parameters whose values depend on the op's input shape or data.
    2. might know how to define the declared parameters in terms of new ops
       (a best effort process that depends on the op type and its inputs).
  }];
  let cppNamespace = "::M::MO";

  let methods = [
    InterfaceMethod<
    "Return indices of the inputs whose actual data (not shape) are needed for "
    "computing the output parameters.",
    "SmallVector<int32_t>", "getDataDependentInputIndices",
      (ins), [{}]>,

    InterfaceMethod<
    "Try returning a list of ops (ordered by use-def chains) that will define "
    "the shape parameters declared by this op, assuming `inputAttrs` are the "
    "symbolic values of the op's individual input shape or data.",
    "ShapeMaterializeResult", "materializeShapeDefs",
      (ins "ArrayRef<TypedAttr>":$inputAttrs),
    [{}]>,
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

  let verify = [{ return Impl::verifyMatmulLike(cast<MatmulLike>($_op)); }];
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

  let verify = [{ return Impl::verifySlidingWindow(cast<SlidingWindow>($_op)); }];
}

def MO_ElementWiseUnary : MO_TensorOpInterfaceBase<"ElementWiseUnary"> {
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

def MO_ElementWiseBinary : MO_TensorOpInterfaceBase<"ElementWiseBinary"> {
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
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the axis tensor as a TypedValue",
      "TensorValue", "getAxis", (ins)>,
    InterfaceMethod<"Get the axis tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getAxisMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyReduction(cast<Reduction>($_op)); }];
}

def MO_ScatterLike : MO_TensorOpInterfaceBase<"ScatterLike",
    [MO_ParamDeclaration,]> {
  let description = [{
    Interface for modeling Scatter-like operations (i.e., regular Scatter
    and Scatter with reductions).
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the updates tensor as a TypedValue",
      "TensorValue", "getUpdates", (ins)>,
    InterfaceMethod<"Get the updates tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getUpdatesMutable", (ins)>,
    InterfaceMethod<"Get indices tensor as a TypedValue",
      "TensorValue", "getIndices", (ins)>,
    InterfaceMethod<"Get indices tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getIndicesMutable", (ins)>,
    InterfaceMethod<"Get axis tensor as a TypedValue",
      "TensorValue", "getAxis", (ins)>,
    InterfaceMethod<"Get axis tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getAxisMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyScatterLike(cast<ScatterLike>($_op)); }];
}

def MO_ScatterNdLike : MO_TensorOpInterfaceBase<"ScatterNdLike",
    [MO_ParamDeclaration]> {
  let description = [{
    Interface for modeling ScatterND-like operations (i.e., regular ScatterND
    and ScatterND with reductions).
  }];

  let methods = [
    InterfaceMethod<"Get input tensor as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get input tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getInputMutable", (ins)>,
    InterfaceMethod<"Get the updates tensor as a TypedValue",
      "TensorValue", "getUpdates", (ins)>,
    InterfaceMethod<"Get the updates tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getUpdatesMutable", (ins)>,
    InterfaceMethod<"Get indices tensor as a TypedValue",
      "TensorValue", "getIndices", (ins)>,
    InterfaceMethod<"Get indices tensor as a MutableOperandRange",
      "::mlir::OpOperand&", "getIndicesMutable", (ins)>,
    InterfaceMethod<"Get result tensor as a TypedValue",
      "TensorValue", "getResult", (ins)>
  ];

  let verify = [{ return Impl::verifyScatterNdLike(cast<ScatterNdLike>($_op)); }];
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
        llvm::cast<ParameterizationInterface>(this->getOperation());
      return Impl::defaultResolveUnknownParameters(paramInterface, rewriter);
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
        llvm::cast<ParameterizationInterface>(this->getOperation());
      return Impl::resolveSameOperandsAndResultShapeUnknownParameters(
        paramInterface, rewriter);
    }
  }];
}

def MO_SameFirstArgumentOutputShapesParameterization : MO_TensorOpInterfaceBase<
  "SameFirstArgumentOutputShapesParameterization",
  [MO_Parameterization, MO_ParamDeclaration]
> {
  let description = [{
    Interface providing a default parameterization for ops that should have the
    same output shape as the first operand.
  }];

  let extraTraitClassDeclaration = [{
    LogicalResult resolveUnknownParameters(mlir::RewriterBase &rewriter) {
      auto paramInterface =
        llvm::cast<ParameterizationInterface>(this->getOperation());
      return Impl::resolveSameFirstOperandAndResultShapeUnknownParameters(
        paramInterface, rewriter);
    }
  }];
}

def MO_PadLike : MO_TensorOpInterfaceBase<"PadLike", [MO_ParamDeclaration]> {
  let description = [{
    Interface for modeling pad operations.
  }];

  let methods = [
    InterfaceMethod<"Get input as a TypedValue",
      "TensorValue", "getInput", (ins)>,
    InterfaceMethod<"Get padding tensor as a TypedValue",
      "TensorValue", "getPaddings", (ins)>,
    InterfaceMethod<"Get result tensor type",
      "TensorType", "getType", (ins)>
  ];

  let verify = [{ return Impl::verifyPadLike(cast<PadLike>($_op)); }];
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

/// MO traits should inherit from this to ensure they are in the right namespace
/// and have the right dependencies.
class MO_OpTrait<string name, list<Trait> traits = [],
    code extraOpDeclaration = [{}],
    code extraOpDefinition = [{}]>
    : NativeOpTrait<name, traits, extraOpDeclaration, extraOpDefinition> {
  let cppNamespace = "::M::MO";
}

// Marks an op that must execute on the host, with all operands and results also
// on the host
def MO_HostOnly : MO_OpTrait<"MOHostOnly">;

// Marks an op that executes on the host using tensors from any device.
def MO_ExecutesOnHost : MO_OpTrait<"MOExecutesOnHost">;

// Helper to get the operand numbers for the given operand names.
class getOperandNumbers<list<string> operandNames> {
  string ret = !interleave(!foreach(name, operandNames,
    "get" # snakeCaseToCamelCase<name>.ret
    # "Mutable().getOperandNumber()"), ", ");
}

// Marks an op that executes on a single device.
// - Operands listed in hostOperandNames are expected to be on the host.
// - Operands listed in deviceOperandNames are expected to be on the device.
// - All results are expected to be on the device.
// - The execution device of this op is inferred from the first device operand
//   in the list.
//
// If deviceOperandNames is empty, the verifier will instead expect all
// operands not mentioned in hostOperandNames to be on the device. In this case
// the execution device is inferred from the first result.
class MO_SingleDeviceWithHostOperands<list<string> hostOperandNames = [],
                                      list<string> deviceOperandNames = []>
  : MO_OpTrait<"MOSingleDeviceWithHostOperands">
{
  // Make operand names from ODS available as operand indices in C++
  let extraConcreteClassDeclaration = [{
    llvm::SmallVector<size_t> getHostOperandIndices() {
      return {}] # getOperandNumbers<hostOperandNames>.ret # [{};
    }
    llvm::SmallVector<size_t> getDeviceOperandIndices() {
      return {}] # getOperandNumbers<deviceOperandNames>.ret # [{};
    }
  }];
}

// Marks an op that executes on a single device, with all operands and results
// also on the same device.
// This trait assumes that the op's first result tensor is on the target device.
def MO_SingleDevice : MO_SingleDeviceWithHostOperands<>;

// Marks an op that executes on multiple devices.
def MO_MultiDevice : MO_OpTrait<"MOMultiDevice">;

// Marks an op that stores a deviceRef attribute to indicate where it will
// execute, in contrast to the other device traits which do not require a
// deviceRef attribute on the op.
def MO_ExplicitDevice : MO_OpTrait<"MOExplicitDevice">;

// Note: we could add a MO_ExplicitDevices trait if we ever add a multi-device
// op where we can't just infer the devices from the operands and results.

def MO_SameOperandsAndResultType : TraitList<[
  SameOperandsAndResultType,
  MO_SameInputOutputShapesParameterization
]>;

/// Verify invariants we expect about operations working with !mo.buffer.
/// All MO and RMO (really RMO analogue) ops inherit from this trait.
///
/// It enforces that operations consuming buffers correctly consume and produce
/// !mo.chain values.
def MO_BufferOpConformance : MO_OpTrait<"MOBufferOpConformance">;

/// Logically similar to the builtin SameOperandsAndResultShape, but uses
/// MOSH::ShapeAttr which is used to store the MO::Tensor shape.
def MO_SameOperandsAndResultShapeAttr :
    MO_OpTrait<"MOSameOperandsAndResultShapeAttr">;

def MO_SameOperandsAndResultShape: TraitList<[
  MO_SameOperandsAndResultShapeAttr,
  MO_SameInputOutputShapesParameterization
]>;

/// All traits that reduction ops should have.
def MO_ReductionTraits: TraitList<[
  MO_Reduction,
  MO_ShapeMaterialization,
  MO_DefaultParameterization,
  MO_OpTrait<"ReductionShapeMaterialization">,
]>;

/// All traits that ScatterND-like ops should have.
def MO_ScatterLikeTraits: TraitList<[
  MO_ScatterLike,
  MO_ShapeMaterialization,
  MO_DefaultParameterization,
  MO_OpTrait<"ScatterLikeShapeMaterialization">,
]>;

/// All traits that ScatterND-like ops should have.
def MO_ScatterNdLikeTraits: TraitList<[
  MO_ScatterNdLike,
  MO_DefaultParameterization,
  MO_OpTrait<"ScatterNdLikeShapeMaterialization">,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
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

/// Checks the results of an op have the same dtype as the first operand.
/// This is useful for shape-like operators like reshape, squeeze, etc.
/// which typically take in an input tensor, some additional operands
/// describing how to change the input shape, and produce a newly shaped
/// output tensor. The dtype of the original tensor should not change in
/// the transformation.
def MO_SameFirstOperandAndResultsElementType :
    MO_OpTrait<"MOSameFirstOperandAndResultsElementType">;

def MO_PadLikeTraits: TraitList<[
  MO_PadLike,
  MO_DefaultParameterization,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
]>;

def MO_ConstantLike : MO_TensorOpInterfaceBase<"ConstantLike"> {
  let description = [{
    Interface for modeling constant operations.
  }];

  let methods = [
    InterfaceMethod<"Get result tensor type",
      "TensorType", "getType", (ins)>,
    InterfaceMethod<"Get the result tensor as a TypedValue",
      "TensorValue", "getResult", (ins), [{}],
      [{ return ::llvm::cast<TensorValue>($_op.getResult(0)); }]>,
  ];
}

def MO_PreservedDuringKernelLowering : OpInterface<"PreservedDuringKernelLowering"> {
  let description = [{
    Represents a MO operation that must have must lowered using a kernel
    implementation.
  }];
  let cppNamespace = "::M::MO";
}

def MO_ViewLike : OpInterface<"ViewLike"> {
  let description = [{
    Represents a view op.
  }];
  let cppNamespace = "::M::MO";
}

def MO_ElementWiseLike : OpInterface<"ElementWiseLike"> {
  let description = [{
    Represents an generic element-wise op.
  }];
  let cppNamespace = "::M::MO";
}

def MO_SameVariadicOperandSizeInterface : OpInterface<"SameVariadicOperandSizeInterface"> {
  let description = [{
    Interface that represent MO Ops that take multiple variadics, all with the same size.
    Wrapper around the builtin `SameVariadicOperandSize` that can't be checked in C++.
  }];
  let cppNamespace = "::M::MO";
}

def MO_SameVariadicOperandSize: TraitList<[
  MO_SameVariadicOperandSizeInterface,
  SameVariadicOperandSize,
]>;

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOINTERFACES_TD
