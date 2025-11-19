//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOOPS_TD

include "GenericML/GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGEnums.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGTypes.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOEnums.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "GenericML/GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "GenericML/GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GenericML/MGPDialect/MGPOpInterfaces.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "GraphRT/MEFDialect/MEFTypes.td"


// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOGGDialect, mnemonic, traits>;

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []>
    : MOGG_Op<mnemonic, traits # [DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>]>;

class MOGG_KernelTerminatorOp<string mnemonic, list<Trait> traits = []>
    : MOGG_Op<mnemonic, traits # [
              Terminator,
              DeclareOpInterfaceMethods<MOGG_KernelTerminator>
            ]>;

def DimensionType : ScalarOfDType<BoundAnd<DTypeConstraint<"*$_self == KGEN::KGENDType::si64">>, "!pop.simd<1, si64>">,
    BuildableType<"KGEN::POP::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::si64))">;

//===----------------------------------------------------------------------===//
// Internal helper ops
//===----------------------------------------------------------------------===//

def MOGG_ConcatOp : MOGG_LowerableOp<"concat", [
  Pure, MO_HasDeviceInterface, MO_SingleDevice
]> {
  let summary = "Concat Operation along axis zero";
  let description = [{
    Concat input tensors along axis zero.

    Example:

    ```mlir
    %arg0: !mo.tensor<[3, 3], f32>
    %arg1: !mo.tensor<[2, 3], f32>
    %res = mogg.concat(%arg0, %arg1) : (
      !mo.tensor<[3, 3], f32>, !mo.tensor<[2, 3], f32>
    ) -> !mo.tensor<[5, 3], f32>
    ```

    Works for CUDA and host devices.
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def MOGG_SplitOp : MOGG_LowerableOp<"split", [Pure]> {
  let summary = "Split Operation along axis zero";
  let description = [{
    Split input tensors into multiple tensors along axis zero.

    Each of the output tensors has the same shape as `input` except along axis
    zero where the sum of the zero dimensions from all output tensors should be
    equal to the one of the input tensor. Note that this op is a special case
    of `mo.split` where `axis` is zero. The `split_sizes` is implicitly defined
    by `mogg.split`'s output tensors.

    Example:

    ```mlir
    %input: !mo.tensor<[9, 8], f32>
    %res = mogg.split(%input) : (!mo.tensor<[9, 8], f32> ) ->
    (!mo.tensor<[2, 8], f32>, !mo.tensor<[3, 8], f32>, !mo.tensor<[4, 8], f32>)
    ```

    Currently only for the 'host' device.
    TODO: Generalize to arbitrary devices.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(`
      qualified(type($input)) `)` `->` `(` type($results) `)`
  }];
}

//===----------------------------------------------------------------------===//
// AsTupleOp
//===----------------------------------------------------------------------===//

def MOGG_AsTupleOp
    : MOGG_Op<"as_tuple", [Pure, DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "MOGG tuple creation operation";
  let description = [{
    This operation creates tuple of values that will be later been translated to
    mojo's StaticTuple.

    ```mlir
    %r = mogg.as_tuple(%t1, %2: mo.tensor<...>, mo.tensor<...>) : tuple<mo.tensor<...>, mo.tensor<...>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyTuple:$result);
  let assemblyFormat = "attr-dict `(`$inputs `)` `:` type($inputs) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MOGG_OutputOp
    : MOGG_Op<"output", [Terminator]> {
  let summary = "MOGG output operation";
  let description = [{
    The "mogg.output" operation specifies the output values for a region in a
    MOGG operation. It is intended to allow the regions of mogg operations to
    grow in complexity with specific outputs.

    ```mlir
      mogg.op foo() {
      ... mogg.output e0, % 1 : i32, f8
    } : i32, f8
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}


//===----------------------------------------------------------------------===//
// MOGG abstract compiled components representation
//===----------------------------------------------------------------------===//

def MOGG_CallStubOp : MOGG_Op<"call.stub", [DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>]>  {
  let summary = "Calls into an outlined stub";
  let description = [{}];
  let arguments = (ins SymbolRefAttr:$name, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>: $outputs);
  let assemblyFormat = "$name ($inputs^ `:` type($inputs))? (`->` type($outputs)^)? attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    KernelStubOp getCallee();
  }];
}

def MOGG_CallForeachOp : MOGG_KernelTerminatorOp<"call.foreach", [
    DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>
]>  {
  let summary = "call the registered foreach kernel";
  let description = [{
    This represents a call to the mojo implementation of `foreach` function. If
    a mogg.kernel terminates with the operation, it also implies that the kernel
    is an element-wise kernel.
  }];

  let arguments = (ins MO_Tensor:$input,
                       AnyTypeOf<[MO_Tensor, MO_Buffer]>:$output,
                       Variadic<AnyTypeOf<[MOGG_Context, AnyTuple]>>:$context);

  let assemblyFormat = [{
    `input` `(` $input `)` `output` `(` $output `)` (`context` `(` $context^ `)`)? `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallExecuteOp : MOGG_KernelTerminatorOp<"call.execute", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>]>  {
  let summary = "Call the registered `execute` function for a particular kernel";
  let description = [{
    This represents a call to the mojo implementation of `Kernel::execute`
    function.

    Example:
    ```mlir
    mogg.call.execute["mo.add"] inputs(...) outputs(...)

    // Lowered to
    kgen.call "Add::execute"(...)
    ```
  }];

  let arguments = (ins StrAttr:$kernelName,
                       OptionalAttr<TypeAttr>:$regRetType,
                       UnitAttr:$isThrows,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `[` $kernelName `]` (`inputs` `(` $inputs^ `:` qualified(type($inputs)) `)`)?
                        (`outputs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
                        (`->` $regRetType^)? attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallMaterializeOp : MOGG_KernelTerminatorOp<"call.materialize", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>
]>  {
  let summary = "Call the registered `view_copy_impl` function for materializing a view kernel";
  let description = [{
    This represents a call to the mojo implementation of `view_copy_impl`
    function.

    Example:
    ```mlir
    mogg.call.materialize inputs(...) outputs(...)

    // Lowered to
    kgen.call "view_copy_impl"(...)
    ```
  }];

  let arguments = (ins AnyType:$input,
                       MO_Tensor:$output,
                       Variadic<AnyTypeOf<[MOGG_Context, AnyTuple]>>:$context);

  let assemblyFormat = [{
    `input` `(` $input `)` `output` `(` $output `)` (`context` `(` $context^ `)`)? `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallElementwiseOp : MOGG_Op<"call.elementwise", [
    DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface>,
    Pure,
    HasParent<"MOGG::BindLambdaOp">]> {
  let summary = "Call the registered `::elementwise` function for a particular kernel";
  let description = [{
    This represents a call to the `elementwise` method.

    Example:
    ```mlir
    mogg.call.elementwise["mo.add"](...)

    // Lowered to
    kgen.call "Add::elementwise"(...)
    ```
  }];
  let arguments = (ins StrAttr: $kernelName, Variadic<AnyType>: $inputs);
  let results = (outs Variadic<AnyType>: $outputs);

  let assemblyFormat = [{
    `[` $kernelName `]` `(` $inputs `)` attr-dict `:` type($inputs) `->` type($outputs)
  }];
}


def MOGG_CallViewOp : MOGG_Op<"call.view", [
    DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface>,
    Pure]> {
  let summary = "Call the corresponding `update_input_view` function for a particular kernel";
  let description = [{
    This represents a call to `update_input_view` functions for a corresponding
    view kernel.

    Example:
    ```mlir
    mogg.call.view["mo.add"]<[?, 2, 3]>(...)

    // Lowered to
    kgen.call "Add::update_input_view"(...)
    ```
  }];
  let arguments = (ins StrAttr: $kernelName,
                       Variadic<AnyType>: $inputs);
  let results = (outs MO_Tensor: $output);

  let assemblyFormat = [{
    `[` $kernelName `]` `(` $inputs `)` attr-dict `:` type($inputs) `->` qualified(type($output))
  }];
}

def MOGG_AsScalarOp: MOGG_Op<"as_scalar", [
    Pure,
    DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "Converts a tensor that contains one element to a scalar";
  let description = [{
    Given a tensor with one element, ie with a shape of [] or [1], etc, convert
    it to a scalar because the kernel using it expects it as a scalar in Mojo.
  }];

  let arguments = (ins MO_ScalarOfAnyRank:$input);
  let results = (outs MO_Scalar:$result);
  let hasVerifier = 1;
}

def MOGG_ExpKernelOp : MOGG_LowerableOp<"experimental.kernel", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    AttrSizedOperandSegments,
    IsolatedFromAbove,
    SingleBlock,
    MO_HasDeviceInterface,
    MO_ExplicitDevice
]> {
  let summary = "A experimental representation of a kernel which can be fused into and compiled without using kgen.";
  let description = [{
    This is mostly a replica of `mogg.kernel` operation, the purposes
    of the operations are:
     1. to make the transition to mogg-based code-gen simpler (by making sure
        unsupported pass wouldn't accidentally pick up a `mogg.kernel` but with
        a mogg-based representation).
     2. Provides a set of util functions that are only available under the
        new representation.

    The operation should be short-lived and will be a drop-in replacement for
    `mogg.kernel` after the transition is done.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Buffer, Index]>>:$inputs,
    Optional<MO_Chain>:$chain,
    MO_ParamDecls:$outputParamDecls,
    DefaultValuedOptionalAttr<StrAttr, "">:$summary,
    DefaultValuedOptionalAttr<BoolAttr, "true">:$isAsync,
    DefaultValuedOptionalAttr<ArrayAttr, "{}">:$allocIndices,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect
  );

  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Chain, MOGG_ShapeFuncResult]>>:$results);
  let regions = (region SizedRegion<1>:$execBody);

  let extraClassDeclaration = [{
    KernelTerminator getTerminator();

    // Whether the kernel is an view kernel.
    bool isViewOp() { return isa<MOGG::CallMaterializeOp>(getTerminator());}

    // Get the input lambda associated with the kernel operand at position operandIdx.
    BindLambdaOp getBoundInputLambda(uint64_t operandIdx);
    // Get the output lambda associated with the kernel result at position operandIdx.
    BindLambdaOp getBoundOutputLambda(uint64_t resultIdx);
    // Get the elementwise lambda of this kernel.
    BindLambdaOp getElemwiseBodyLambda();

    bool isOperandFusible(uint64_t operandIdx, bool lookThroughView = false);
    bool hasFusibleOperand(bool lookThroughView = false) {
      for (unsigned i = 0, e = getExecBody().getNumArguments(); i < e; i++)
        if (isOperandFusible(i, lookThroughView))
          return true;

      return false;
    }
    /// Returns the OutputPlaceholder for this operation or null if it doesn't
    /// exist.
    OutputPlaceholdersOp getOutputPlaceholder();

    // Returns the StackAllocateTensorsOp for this operation or null if it
    // does not exist.
    StackAllocateTensorsOp getStackAllocateTensor();

    // Returns the CreateEmptyTensorsOp for this operation or null if it does
    // not exist.
    CreateEmptyTensorsOp getCreateEmptyTensor();

    // Returns the ScratchPlaceholderOp for this operation or null if it does
    // not exist.
    ScratchPlaceholdersOp getScratchPlaceholder();

    // Returns the OutputOp for this operation or null if it does not exist.
    OutputOp getOutputOp();

    /// Returns the sole input chain (if present) and std::nullopt otherwise.
    std::optional<MO::ChainValue> maybeGetInChain();
    /// Returns the sole output chain (if present) and std::nullopt otherwise.
    std::optional<MO::ChainValue> maybeGetOutChain();

    // Use the following methods to update inputs/chains
    // (instead of op->insertOperand/eraseOperand) to keep the
    // `operandSegmentSize` up-to-date.
    //
    // Insert a range of new inputs at `idx`, this updates both operands and
    // block arguments.
    void insertInputsAt(unsigned idx, ValueRange vals);
    // Remove a range of inputs at `[idx, idx + len)`, this updates both operands and
    // block arguments.
    void eraseInputsAt(unsigned idx, size_t len = 1);
    //
    // Chain update.
    //
    // Set and erase chain.
    void setChain(Value chain);
    void eraseChain();
  }];

  // This should set all attributes including MO_HasDeviceInterface
  let builders = [
    OpBuilder<(ins
      "TypeRange":$results,
      "ValueRange":$operands,
      "StringAttr":$summary,
      "BoolAttr":$isAsync,
      "DeviceRefAttr":$device,
      CArg<"ArrayRef<ParamDeclAttr>", "{}">:$outputParamDecls,
      CArg<"ArrayAttr", "{}">:$allocIndices,
      CArg<"bool", "false">:$hasReadEffect,
      CArg<"bool", "false">:$hasWriteEffect
    )>,
  ];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders=1;
  let hasVerifier = 1;
}

// lower to ManageTensorSlice.load
def MOGG_LoadTensorOp : MOGG_Op<"tensor.load", [HasParent<"MOGG::BindLambdaOp">,
    DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "Load a value from mo.tensor at the given indices";
  let description = [{
     This operation represent a tensor load at the given indices. It is needed
     to be able to fuse N-ary elementwise kernels into a output lambda, in which
     case, only one of the operands get fused.

     For example:
     ```mlir
     %a = bind (%out) output_lambda (%out_v, %indices) {
       // The unfused lhs need a concrete tensor load.
       %lhs = tensor.load(%add_lhs, %indices);
       %fused_add = mogg.call.elementwise["mo.add"](%lhs, %out_v)
       mogg.output %fused_add, %indices
     }
     ```

     It will lowered to a call to
     ```mojo
     fn simd_load_from_managed_tensor_slice(...)
     ```

  }];

  let arguments = (ins MO_Tensor:$tensor, POP_ArrayType:$indices);
  let results = (outs AnyType: $outputs);
}


def MOGG_BindLambdaOp : MOGG_Op<"bind", [
    SingleBlock,
    DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "Bind a fusion interface to a tensor";
  let description = [{
     This operation exposes a fusion interface to a tensor. It build the connection
     between in/out_lambda in `StaticTensorSpec` with the tensor.

     For example:
     ```mlir
     // To make %in a FusedInputTensor
     %a = bind (%in) input_lambda (%in_v, %indices) {
       %fused_v = ... (%in_v) // whatever got fused.
       mogg.output %in_v
     }

     // To make %out a FusedOutputTensor
     %a = bind (%out) output_lambda (%out_v, %indices) {
       %fused_v = ... (%out_v) // whatever got fused.
       mogg.output %out_v, %indices
     }

     // To make %in Elemwise fusible
     %a = bind (%in) elemwise_lambda (%in_v, %indices) {
       %fused_v = ... (%in_v) // other elemwise that got fused.
       mogg.output %in_v
     }
     ```
  }];

  let arguments = (ins MOGG_BindLambdaKind: $bindKind,
                       Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>: $fusibleTensors);


  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>: $output);
  let regions = (region SizedRegion<1>:$lambda);

  let extraClassDeclaration = [{
    BlockArgument getIndicesArg();
    ArrayRef<BlockArgument> getValueArgs();

    // Whether the bound lambda is identity or not.
    bool isIdentityLambda();

    // Returns true if the lambda contains some affine-view operations.
    bool hasAffineViewOperation();
  }];

  let builders = [
    OpBuilder<(ins
      "Type":$outputTp,
      "ValueRange":$fusibleTensors,
      "BindLambdaKind":$bindKind
    )>,
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}


def MOGG_ConcretizeTensorOp: MOGG_Op<"concretize", [Pure, SameOperandsAndResultType]> {
  let summary = "Concretizes the virtual tensor by assigning to it the pointer of the physical tensor";
  let description = [{
    This op is necessary when we want to concretize/materialize a tensor that is
    the result of a lambda op. It happens with mixed precision output fusion on
    kernels that still need the virtual tensor to exist.
  }];

  // Can't use `virtual` because it's a reserved keyword in C++.
  let arguments = (ins MO_Tensor:$phy, MO_Tensor:$vir);
  let results = (outs MO_Tensor:$result);
}


def MOGG_KernelOp : MOGG_LowerableOp<"kernel", [
    IsolatedFromAbove,
    SingleBlock,
    SymbolTable,
    DeclInterface,
    DeclareOpInterfaceMethods<MO_MutableOpInterface, [
      "getInChains", "getOutChains", "getInChainsMutable"
    ]>,
    DeclareOpInterfaceMethods<MO_ConditionallyInPlaceInterface>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    MO_HasDeviceInterface,
    MO_ExplicitDevice
]> {
  let summary = "A representation of a kernel which can be fused into and compiled";
  let description = [{
    This kernel represents the handover from the MO graph to the KGEN land.

    The operation itself is a fully legal graph semantic operation which
    composes with all other operations at the MO level.

    Inside the kernel however the semantics are mirrors of the KGEN semantics.
    We operate on lambdas and kgen function calls. This lets us have both inline
    in the same graph while keeping the hard dependencies contained to lowering.

    The kernel abstraction also exposes a number of hooks. These hooks are
    details of the fusion algorithms. They allow this kernel to express the
    semantics by which its outputs or inputs may be folded by being fused into.

    Example:

    ```mlir
    %1 = mogg.kernel (%0, %arg1) : (%arg2: !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>{
      ...
    } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn", "input_1_fn"], outputFusionInterface = ["output_0_fn"]}
    ```

    This above kernel has a graph semantic %1 and inputs %0, %arg1 which work
    with the rest of the MO/MOGG operations.  Internally it works on `(%arg2:
    !mo.tensor<[2, 3], f32>, %arg3: !mo.tensor<[2, 3], f32>)` which essentially
    are the KGEN view of those values.

    `outputParamDecls` allows this compiled operation to define parameter
     values. These values are explicit parameter returns of the kernel. They
     cannot be referred to within the kernel other than to set them. When the
     kernel is executed they will be updated in the MGP parameter map. This is
     completely separate from the normal kernel tensor returns. Their values
     are set within the kernel using `SetParameterOp`.

    The attributes at the end.
    ```mlir
    { fusionBodyLambda = "elemwise_compute_lambda",
      inputFusionInterface = ["input_0_fn", "input_1_fn"],
      outputFusionInterface = ["output_0_fn"]
    }
    ```
    `fusionBodyLambda` Represents the fusable elementwise / view operation body
    contained within. This signifies that this kernel is fully implemented by
    applying this lambda to an otherwise identity operation, therefore we can
    safely fuse it into a user.  Most ops will not have this, only elementwise
    and view ops. After being fused into an op that doesn't have `fuseHook` it
    will be lost and can no longer be fused. E.G elementwise into matmul.

    `inputFusionInterface` Represents the lambdas within this kernel that are
    responsible for loading each input. The name doesn't matter, we know which
    input this matches by the index in the attr array vs the index of the
    operand in the block.

    `outputFusionInterface` Is the same as the above but for storing.

    `isViewOp` Is an easy way for us to check if something can be represented as
    a view, e.g., a broadcast in the form of a view change followed by a copy. We
    can drop the copy and keep the view change op.

    `allocIndices` is a list attribute containing the indices of the output tensors
    that have in-place allocation. For example, allocIndices = [0] means the first
    output tensor has been allocated in the kernel.

    Kernels preserve the device of their original MO op.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Buffer, MO_Chain, Index]>>:$operands,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<ArrayAttr>:$inputFusionInterface,
    OptionalAttr<ArrayAttr>:$outputFusionInterface,
    OptionalAttr<StrAttr>:$fusionBodyLambda,
    OptionalAttr<BoolAttr>:$isViewOp,
    OptionalAttr<StrAttr>:$summary,
    OptionalAttr<BoolAttr>:$isAsync,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$computationallyTrivial,
    OptionalAttr<ArrayAttr>:$allocIndices,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Chain]>>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body, $outputParamDecls) attr-dict
  }];


  let extraClassDeclaration = [{

    /// Sometimes it is necessary to temporarily pretend that a kernel has
    /// fusion enabled however we still need to distinguish between those
    /// which have fusion disabled by the user and those which we have
    /// temporarily marked. These must be removed before the verifier runs.
    static constexpr StringLiteral getTempLambdaName() {
      return "__TMP__FUSE_HOOK";
    }

    /// Strip any of the temporary fusion lambda markers.
    void removeTemporaryFusionMarks();

    /// Returns the OutputPlaceholder for this operation or null if it doesn't
    /// exist.
    OutputPlaceholdersOp getOutputPlaceholder();

    // Returns the StackAllocateTensorsOp for this operation or null if it
    // does not exist.
    StackAllocateTensorsOp getStackAllocateTensor();

    // Returns the CreateEmptyTensorsOp for this operation or null if it does
    // not exist.
    CreateEmptyTensorsOp getCreateEmptyTensor();

    // Returns the OutputOp for this operation or null if it does not exist.
    OutputOp getOutputOp();

    // Returns the ScratchPlaceholderOp for this operation or null if it does
    // not exist.
    ScratchPlaceholdersOp getScratchPlaceholder();

    /// Disables future fusion on this input.
    void disableInputFusionOnInput(size_t index);

    /// Removes an operand at the given index and disables input fusion on that
    /// position if this kernel otherwise supports it.
    void removeOperandAtIndex(size_t index);

    /// Add an operand to the kernel at the idx specified by insertIdx and adds an
    /// argument to the kernel region. Fixes up the input fusion interfaces to
    /// disallow input fusion on this position. Returns the added kernel argument.
    /// NOTE: this is an O(n) operation.
    Value addOperandAtIndex(Value operand, size_t insertIdx);

    /// Add an operand to the kernel and adds an argument to the kernel region.
    /// Fixes up the input fusion interfaces to disallow input fusion on this
    /// position. Returns the added kernel argument.
    Value addOperand(Value operand);

    /// Mark a given input as having a lambda fusion interface
    void setInputFusionInterface(size_t index, StringRef lambdaName);

    /// Fuse this body into the given builder and symbol table, using the
    /// remapper to update the SSA values and tracking the changed lambda
    /// names in `oldLambdaNamesToNew`.
    void fuseBodyIntoNew(OpBuilder& builder, mlir::SymbolTable& symTable,
        IRMapping& remapper, llvm::StringMap<StringRef>& oldLambdaNamesToNew);

    /// The same as the above but adds the option to skip an operation in the
    /// body.
    void fuseBodyIntoNewWithSkip(OpBuilder& builder, mlir::SymbolTable& symTable,
        IRMapping& remapper, llvm::StringMap<StringRef>& oldLambdaNamesToNew,
        std::function<bool(Operation*)> shouldSkip);

    /// Sets the insertion point to the start of the kernel. This may not be
    /// the precise start as it will place the operation after the output
    /// placeholders if they exist.
    void setInsertionPointToTop(OpBuilder& builder);

    /// Returns the KGEN input parameters this depends on. Necessary to
    /// implement the DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }

    /// Finds a lambda in the body of the op.
    KGEN::ParamDeclareRegionOp findLambda(StringAttr name);

    /// Returns an input lambda or an empty string if it does not exist.
    StringAttr getInputFusionInterfaceAt(size_t index);

    /// Returns an output lambda or an empty string if it does not exist.
    StringAttr getOutputFusionInterfaceAt(size_t index);

    /// Get the indices of the output tensors that are in-place allocated
    DenseSet<int> getInplaceAllocIndicesSet();

    /// Returns whether the kernel is a in-place allocated kernel
    bool isInPlaceAlloc();

    /// Returns the sole input chain (if present) and std::nullopt otherwise.
    std::optional<MO::ChainValue> maybeGetInChain();

    /// Returns the sole output chain (if present) and std::nullopt otherwise.
    std::optional<MO::ChainValue> maybeGetOutChain();

    //===------------------------------------------------------------------===//
    // Helpers for operands/results excluding chains
    //===------------------------------------------------------------------===//

    /// Returns the number of operands excluding any MO::ChainType operands.
    auto getNumNonChainOperands() {
      return getNumOperands() - maybeGetInChain().has_value();
    }

    /// Returns the number of results excluding any MO::ChainType results.
    auto getNumNonChainResults() {
      return getNumResults() - maybeGetOutChain().has_value();
    }

    /// Returns a range of operands excluding any MO::ChainType operands.
    auto getNonChainOperands() {
      return getOperands().drop_back(maybeGetInChain().has_value());
    }

    /// Returns a range of results excluding any MO::ChainType results.
    auto getNonChainResults() {
      return getResults().drop_back(maybeGetOutChain().has_value());
    }

    /// Returns a range of operand types excluding any MO::ChainType operands.
    auto getNonChainOperandTypes() {
      auto types = getOperandTypes();
      return llvm::make_range(types.begin(),
                              types.end() - maybeGetInChain().has_value());
    }

    /// Returns a range of result types excluding any MO::ChainType results.
    auto getNonChainResultTypes() {
      auto types = getResultTypes();
      return llvm::make_range(types.begin(),
                              types.end() - maybeGetOutChain().has_value());
    }
  }];

  // This should set all attributes including MO_HasDeviceInterface
  let builders = [
    OpBuilder<(ins
      "TypeRange":$results,
      "ValueRange":$operands,
      "ArrayRef<ParamDeclAttr>":$outputParamDecls,
      "ArrayAttr":$inputFusionInterface,
      "ArrayAttr":$outputFusionInterface,
      "StringAttr":$fusionBodyLambda,
      "BoolAttr":$isViewOp,
      "StringAttr":$summary,
      "BoolAttr":$isAsync,
      "BoolAttr":$computationallyTrivial,
      "ArrayAttr":$allocIndices,
      "bool":$hasReadEffect,
      "bool":$hasWriteEffect,
      "DeviceRefAttr":$device
    )>,
  ];
  let skipDefaultBuilders=1;
}

def MOGG_GenericCompiledRegionOp : MOGG_Op<"generic.compiled.region", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"MOGG::OutputOp">,
    MO_HasDeviceInterface,
    MO_ExplicitDevice,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclInterface
]> {
  let summary = "An abstract function representing a compiled body of code";
  let description = [{
    This function takes and returns any number of AsyncValues and operates on
    them by compiling the code contained within its body.

    Currently this is always blocking and the code will run to completion
    within the body.

    This is a more generic version of the MOGG.kernel which specifically
    operates on Tensors and other MO level structures. It is intended that
    eventually we lower mogg.kernel into this form directly. The first
    return output is always a mef.chain for now.

    `canFuseIntoOther`, `canBeFusedInto`, and `canDuplicate` dictate how
    sequence fusion (e.g. fusion which just concatenates code without
    additional optimization) can interact with the compiled region.

    Sequence fusion looks at potential `parent --> child` relationships
    between compiled regions and tries to fuse the parent into the child.
    `canFuseIntoOther` determines if this region can fuse as `parent`.
    `canBeFusedInto` determines if this region can fuse as `child`.
    `canDuplicate` determines whether we can duplicate kernels in the
    fusion process (usually small ones). E.g. consider a region A with
    children B and C, if `canDuplicate` for A, then we may allow A
    to inline into B AND C, effectively copying the code for A twice.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    MO_ParamDecls:$outputParamDecls,
    M_InOutSignatureAttr:$inOutSignature,
    OptionalAttr<StrAttr>:$summary,
    OptionalAttr<BoolAttr>:$isAsync,
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canFuseIntoOther,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canBeFusedInto,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canDuplicate,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$emitsMojo
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body, $outputParamDecls) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns the KGEN input parameters this depends on. Necessary to
    /// implement the DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }

    /// Return the single body block for the operation.
    Block &getBodyBlock() { return getBody().front(); }
  }];
}

def MOGG_GetDimOp : MOGG_Op<"get_dim", [Pure]> {
  let summary = "Returns the runtime representation of a dimension";
  let description = [{
    Returns the runtime representation for a single dimension
  }];

  let arguments = (ins Builtin_StringAttr:$dimension);
  let results = (outs DimensionType:$result);

  let assemblyFormat = [{
    custom<Dim>($dimension, type($result)) attr-dict
  }];
}

def MOGG_SetParameterOp : MOGG_Op<"set_param"> {
  let summary = "Assign a param a value";
  let description = [{
    Sets the parameter to the given dimension.

    ```
    %d0 = kgen.param.constant: scalar<si64> = <4>
    mogg.set_param N = %d0
    ```
  }];

  let arguments = (ins
     DimensionType:$dim,
     Builtin_StringAttr:$paramName
  );

  let assemblyFormat = [{
    custom<SetDim>($paramName, $dim) attr-dict
  }];
}

def MOGG_CallOp : MOGG_Op<"call", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<EmitsMojoInterface>
  ]> {
  let summary = "Represents  KGEN function to call";
  let description = [{
    This represents a call to a KGEN function. The kernel target will be a
    function we retrieve from the registered ops. The call can also take an
    optional list of symbols in the form of `$params`. These are lambdas within
    this kernel which should be passed as parameters to the KGEN call.

    Example:
    ```mlir
    mogg.call "fake_op" [@test_lambda, @test_lambda2](%input, %output) : !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>
    ```

    The above will look up `fake_op` in the mojo op register to find the actual
    mojo symbol for that function and will compile down to a kgen call to that
    symbol with the lambda functions passed as parameters.

   ```mlir
    %0 = mogg.call "simd_load"(%input, %indices) : !mo.tensor<[2, 3], f32>, !mogg.indices -> !mogg.simd<f32>
   ```

    These calls can operate on tensors, simd values, and indices, and can return
    any of those types.
  }];

  let arguments = (ins
     StrAttr:$mojoKernelTarget,
     Variadic<AnyType>:$operands,
     OptionalAttr<ArrayAttr>:$params,
     OptionalAttr<DictionaryAttr>:$kgenParams
  );

   let results = (outs Variadic<AnyType>:$results);

   let assemblyFormat = [{
     $mojoKernelTarget ($params^)? `(`($operands^)?`)` ($kgenParams^)? attr-dict
       (`:` type($operands)^)? (`->` type($results)^)?
   }];

  let extraClassDeclaration = [{
    /// Changes the value of a given KGEN parameter if it exists. Returns true
    /// if the parameter exists and the value is now the same as value, false
    /// if the parameter doesn't exist.
    bool setKGENParamToValueIfPresent(StringRef name, Attribute value);
  }];
}

def MOGG_LambdaInvokeOp : MOGG_Op<"invoke",
   [MemoryEffectsOpInterface]> {
  let summary = "Represents an inlined call to a lambda";
  let description = [{
    Lambda invoke is a direct call of that lambda. This will lower to a
    kgen.call_param operation. It cannot take in any parameters. It can take
    in/return only on simd values or indices, not tensors directly. Tensors are
    just captured from above in the invoked lambda.

    Example:
    ```mlir
    %0 = mogg.invoke @test_lambda(%input, %output) : !mogg.indices, !mogg.simd<f32> -> !mogg.simd<f32>
    ```
  }];

  let arguments = (ins
     SymbolRefAttr:$lambda,
     Variadic<AnyTypeOf<[MOGG_Simd, MOGG_Indices]>>:$operands,
     OptionalAttr<BoolAttr>:$readOnly
  );
  let results = (outs Variadic<MOGG_Simd>:$outs);

  let assemblyFormat = [{
    $lambda `(`($operands^)?`)` attr-dict
      (`:` type($operands)^)? (`->` type($outs)^)?
  }];

  // TODO(10098): We will want to come up with a proper side effect analysis here.
  let extraClassDeclaration = [{
    void getEffects(
      llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
      // If we don't have attr or it is false assume this has a side effect.
      bool isReadOnly = getReadOnly().value_or(false);
      if (!isReadOnly)
        effects.emplace_back(mlir::MemoryEffects::Write::get());
    }
  }];
 }


def MOGG_StackAllocateTensorsOp : MOGG_Op<"stack_allocate_tensors", [Pure]> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "Allocates statically sized NDBuffers/Tensor on the host's stack.";
  let description = [{
    Each result of this op is an NDBuffer/Tensor:
     - Only statically shaped `!mo.tensor` types are allowed.
     - The shapes should be 'small' so as to not blow the stack.
     - The NDBuffer/Tensor will be backed by a stack allocated buffer/tensor
       on the host stack.
     - Obviously these NDBuffers/Tensor may only be used for host-side temporaries
       within the kernel and cannot be returned to the caller.

    Example:
    ```mlir
      %tensors:2 = mogg.stack_allocate_tensors : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], f32>)
    ```
  }];

  let hasVerifier = 1;

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_StackAllocateOpaquesOp : MOGG_Op<"stack_allocate_opaques"> {
  let results = (outs Variadic<AnyTypeOf<[MO_Opaque, MGP_OpaqueType]>>:$results);

  let summary = "Allocates opaque types on the host's stack.";
  let description = [{
    Each result of this op is an opaque object.
    It represents the outputs of a kernel call.
    This op does a stack allocation for the outputs of the kernel.
    The data is uninitialized, and will be filled by the kernel.

    Example:
    ```mlir
      %vals:2 = mogg.stack_allocate_opaques : (!mgp.opaque, !mo.opaque<"my_fp">)
    ```
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_CreateEmptyTensorsOp : MOGG_Op<"empty_tensors", [Pure]> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "This creates some tensors which have only shape information but no value";
  let description = [{
    These tensors will be created with the correct shape and type but will have
    no memory allocated. These are a temporary construct to allow shape functions
    to create null tensors without needing to deal with outputs etc.

    Example:
    ```mlir
    %alloc = mogg.create_empty_tensors : !mo.tensor<[2, 3], f32>
    ```
    This returns a tensor with an internal memory allocation.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_BufferBorrowOp : MOGG_Op<"buffer.borrow", [Pure]> {
  let arguments = (ins AnyType:$buf, AnyType:$borrowee);
  let results = (outs AnyType:$result);
  let summary = "Intrinsic to create a buffer but extend lifetime of another";
  let description = [{
    Represents a placeholder operation which on lowering to will produce a
    new buffer which extends a shared lifetime with the other.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToTensorFromMGPOp : MOGG_Op<"to_tensor_from_mgp", [Pure]> {
  let arguments = (ins AnyType:$dataPointer, Variadic<Index>:$dims);
  let results = (outs MO_Tensor:$result);
  let summary = "Represents the construction of an NDBuffer or Tensor";
  let description = [{
    Takes a data pointer and a list of dimensions. Will lower this into a call
    to NDBuffer or Tensor constructor under the hood.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MOGG_DataPointerFromBufferOp : MOGG_Op<"get_buffer_data_ptr", [Pure]> {
  let arguments = (ins AnyType:$buffer);
  let results = (outs AnyType:$result);
  let summary = "Returns the data pointer of a given tensor or buffer.";
  let description = [{
    Get the data pointer of the given MGP buffer.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToMGPFromTensorOp : MOGG_Op<"to_mgp_from_tensor", [Pure]> {
  let arguments = (ins AnyType:$tensor);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the data pointer from the kgen tensor";
  let description = [{
    Get the data pointer from the kgen tensor. The input tensor is the output of
    a `to_kgen` taking a `mo.tensor`.
  }];
}

def MOGG_ToListFromMGPOp : MOGG_Op<"to_list_from_mgp", [Pure]> {
  let arguments = (
    ins AnyType:$dataPointer
  );
  let results = (outs AnyType:$result);
  let summary = "Represents the construction of an list type from a pointer";
  let description = [{
    Takes a pointer to !mgp.list to !mo.list.

    `dataPointer` is an !mgp.list object and `result` is a !mo.list.

    At the implementation level, it's just a rebind since we define the ABI for
    both objects the same way (ABIList).
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_ToMGPFromOpaqueOp : MOGG_Op<"to_mgp_from_opaque", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the opaque pointer from the kgen value";
  let description = [{
    `input` is a KGEN object, and `dataPointer` is an !mgp.opaque
    Get the opaque pointer from the kgen opaque tensor.
    At the implementation level, it's just a bitcast / rebind / no-op.
    But it helps preserving the correct types in the IR during lowering.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
}

def MOGG_GetResourcePtrOp : MOGG_Op<"get_resource_ptr", [Pure,
  HasParent<"MOGG::GenericCompiledRegionOp">
]> {
  let arguments = (ins IndexAttr:$resourceSlot);
  let results = (outs Index: $byteCount, PointerOf<KGEN_NoneType>:$resPtr);

  let summary = "Retrieves a pointer to the corresponding constant resources";
  let description = [{
    Retrieves a pointer to the corresponding constant resources attached to the
    parent `mgp.generic.execute` operation.
  }];

  let hasVerifier = 1;
}

def MOGG_MGPToAsyncRefOp : MOGG_Op<"mgp_to_async_ref", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs PointerOf<KGEN_NoneType>:$result);
  let summary = "A nop just to reconcile type conflict after lower-abi";

  let description = [{
    A no-op just to reconcile type conflict after lower-abi, the operation will
    be removed after generic region models async value correctly.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_AsyncRefToMGPOp : MOGG_Op<"async_ref_to_mgp", [Pure]> {
  let arguments = (ins PointerOf<KGEN_NoneType>:$async);
  let results = (outs AnyType:$mgpResult);
  let summary = "A nop just to reconcile type conflict after lower-abi";

  let description = [{
    A no-op just to reconcile type conflict after lower-abi, the operation will
    be removed after generic region models async value correctly.

    The inverse operation of mogg.mgp_to_async_ref
  }];

  let assemblyFormat = [{
    `(` $async `)` attr-dict `:` functional-type($async, $mgpResult)
  }];
}

def MOGG_AllocateAsyncRefOp : MOGG_Op<"allocate_async_ref_for", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs PointerOf<KGEN_NoneType>:$result);
  let summary = "Returns pointer to the async value of the corresponding output";

  let description = [{
    Each input and output are represented by an async value. This operation
    allocate an async value explicitly (for output).

    Example:

    ```mlir
    %async_storage = rt.allocate_async_ref %tensor: () -> !kgen.pointer<NoneType>
    %async_val = call "builtin.pack_xxx" (%tensor, %async_storage, ...)
    ```

    When the async value is returned, the allocation is folded into a
    pre-allocated async values from runtime. E.g.,

    ```mlir
    # Before
    %async_storage = rt.allocate_async_ref %tensor: () -> !kgen.pointer<NoneType>
    %async_val = call "builtin.pack_xxx" (%tensor, %async_storage, ...)
    mogg.output %tensor

    # After
    %async_storage = rt.get_output_async_value_for %tensor: () -> !kgen.pointer<NoneType>
    %async_val = call "builtin.pack_xxx" (%tensor, %async_storage, ...)
    mogg.output %tensor
    ```

    When the async value is not returned (due to post-abi fusion), a destructor
    will have to be inserted.

    ```mlir
    # Before
    %async_storage = rt.allocate_async_ref %tensor: () -> !kgen.pointer<NoneType>
    %async_val = call "builtin.pack_xxx" (%tensor, %async_storage, ...)
    %borrowed = mgp.borrows(async_val)
    mogg.output %borrowed

    # After
    %async_storage = rt.allocate_async_ref %tensor: () -> !kgen.pointer<NoneType>
    %async_val = call "builtin.pack_xxx" (%tensor, %async_storage, ...)
    %borrowed = mgp.borrows(%tensor)
    call "builtin.destruct_async_val" (%async_val)
    mogg.output %borrowed
    ```
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_DeallocateAsyncRefOp : MOGG_Op<"deallocate_async_ref"> {
  let arguments = (ins Variadic<PointerOf<AnyType>>:$asyncRefs);
  let summary = "Deallocate the async value ref";

  let description = [{
     Explicitly deallocate the provided AsyncValueRef * (i.e., the argument
     should be the address of a AnyAsyncRef).
  }];

  let assemblyFormat = [{
    `(` $asyncRefs `)` attr-dict `:` qualified(type($asyncRefs))
  }];
}

def MOGG_CreateAsyncMojoValueOp : MOGG_Op<"create_async_mojo_value"> {
  let arguments = (ins
      AnyType:$mojoPointer,
      AnyType:$asyncVal
  );
  let summary = "Create an AsyncValue from a mojo pointer";
  let description = [{
    Create an AnySyncValue<MojoValue> from a mojo pointer.
    This can't just be a mogg.call like for other asyncs, as the logic is a bit more complex.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_PackToAsyncRef : MOGG_Op<"pack_to_async_ref", [Pure]> {
  let arguments = (ins AnyType:$inputVal,
                       Optional<MGP_DeviceContextType>:$deviceCtx);
  let results = (outs PointerOf<KGEN_NoneType>:$resultAsync);
  let summary = "Pack a KGEN value to a async ref";

  let description = [{
    Pack a KGEN value to a async ref. I.e,

    ```mlir
      %async = mogg.pack_to_async_ref(%kgen)
      // is equivalent to
      %async = allocate_async_value()
      mogg.call "builtin.create_xxx_async(%kgen, %async)"
    ```
  }];

  let assemblyFormat = [{
    `(` $inputVal (`,` $deviceCtx^)? `)` attr-dict
    `:` qualified(type($inputVal)) (`,` qualified(type($deviceCtx))^)?
  }];
}

def MOGG_UnpackFromAsyncRef : MOGG_Op<"unpack_from_async_ref", [Pure]> {
  let arguments = (ins PointerOf<KGEN_NoneType>:$inputAsync);
  let results = (outs AnyType:$resultVal);
  let summary = "Unpack a KGEN value from a async ref";

  let description = [{
    Unpack a KGEN value from a async ref. I.e,

    ```mlir
      %kgen = mogg.unpack_from_async_ref(%async)
      // is equivalent to
      %kgen = mogg.call "builtin.unpack_xxx_async(%async)"
    ```
  }];

  let assemblyFormat = [{
    `(` $inputAsync `)` attr-dict `:` qualified(type($resultVal))
  }];
}

def MOGG_TransferAsyncRef : MOGG_Op<"transfer_async_ref", [Pure]> {
  let arguments = (ins PointerOf<KGEN_NoneType>:$asyncSrc);
  let results = (outs PointerOf<KGEN_NoneType>:$asyncDst);
  let summary = "Transfer the ownership of an async ref to a new one.";

  let description = [{
    Transfer the ownership of an async ref to a new one. I.e,

    ```mlir
      %dst = mogg.transfer_async_ref(%src)
      // is equivalent to
      %async = allocate_async_value()
      mogg.call "builtin.transfer_async_ref"(%kgen, %async)
    ```
  }];

  let assemblyFormat = [{
    `(` $asyncSrc `)` attr-dict
  }];
}

def MOGG_ReleaseMojoValueOp : MOGG_Op<"release_mojo_value"> {
  let arguments = (ins
      AnyType:$mojoValue
  );
  let summary = "Call the mojo destructor on a mojo object";
  let description = [{
    Call the destructor method on a KGEN mojo value.
    Mojo objects are usually stored in an AnySyncValue<MojoValue>.
    But when fusing regions, some leftover objects are never released.
    This op takes care of realising them.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];
}

def MOGG_AlignmentPlaceholderOp : MOGG_Op<"alignment_placeholder", [
    DeclareOpInterfaceMethods<ParamOpInterface>]>, OneParamDecl {
  let summary = "Inserts a placeholder for a value and the param name";
  let description = [{
    This is a placeholder for each input/output buffer's alignment parameter.
    All allocations and alignment becomes concrete in MGP. This placeholder
    associates a value operand and the param for alignment, so that information
    about the input value and the parameter can be associated when the
    alignment is known.

    This must resolve to a `kgen.param_declare` with the specified name.
    Soft-fusion (seq fusion) will uniquify the name when fusing.

    Example:
    ```mlir
      mogg.alignment_placeholder %operand, ALIGN_0
    ```
  }];

  let arguments = (
    ins AnyType:$operand,
    KGEN_ParamDeclAttr:$paramDecl
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];

}

def MOGG_AnyAsyncRefPlaceholderOp : MOGG_Op<"any_async_ref_placeholder"> {
  // Allocate a `void *`, returning a `void **`
  let results = (outs PointerOf<PointerOf<KGEN_NoneType>>:$result);
  let summary = "Returns a placeholder for holding a AsyncRT::AnyAsyncValueRef";
  let description = [{
    This is resolved to an pop.stack_allocation during lowering. The content of
    the placeholder is initialized to be nullptr.
  }];

  let assemblyFormat = [{attr-dict}];
}

def MOGG_OutputPlaceholdersOp : MOGG_Op<"output_placeholders", [Pure]> {
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Opaque, MO_Chain]>>:$results);

  let summary = "Represents buffer semantic SSA values for the outputs";
  let description = [{
    As the KGEN kernels require the output to be a buffer semantic input to the
    kernel we need to represent this within the kernel. This op produces a
    placeholder we can use for each output. During lowering these become inputs
    to the kernel as KGEN expects.

    Example:
    ```mlir
      %out = mogg.output_placeholders : !mo.tensor<[2, 3], f32>
    ```
    This provides a `%out` SSA value which corresponds to the first output which
    we can now pass into the KGEN calls which expect a buffer.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_ScratchPlaceholdersOp : MOGG_Op<"scratch_placeholders"> {
  let results = (outs Variadic<MO_Tensor>:$results);

  let summary = "Represents scratch buffer tensors that are not returned from the kernel";
  let description = [{
    Similar to `mogg.output_placeholders`, this operation produces tensors for use within a kernel op.
    However, unlike output placeholders, these scratch tensors will not be returned from the kernel.

    Example:
    ```mlir
      %scratch = mogg.scratch_placeholders : !mo.tensor<[2, 3], f32>
    ```

    This provides a `%scratch` SSA value which corresponds to a temporary buffer
    that can be used within the kernel but will not be part of the kernel's results.
  }];

  let assemblyFormat = [{
   attr-dict `:` type($results)
  }];
}

def MOGG_CreateReadyChainOp : MOGG_Op<"create_ready_chain", [Pure]> {
  let results = (outs AnyType:$result);

  let summary = "Represents a chain output placeholder";
  let description = [{
    This is to help resolve any chain dependencies for data-dependent operations
    and help elide chains during fusion when lowering down to a compiled region.

    This is a special-case to the output placeholders because we remove this much lower
    in the kgen pipeline compared to the output_placeholders

    Example:
    ```mlir
      %out = mogg.create_ready_chain : !mef.chain
    ```
    This provides a `%out` SSA value which corresponds to the chain output that kernels
    want to output.
  }];
}

//===----------------------------------------------------------------------===//
// MOGG Runtime compiled code representation ops.
//===----------------------------------------------------------------------===//

def MOGG_DeviceContextPlaceholderOp: MOGG_Op<"device_context_placeholder", [Pure]> {
  let summary = "Launders the runtime device context into KGEN land";
  let description = [{
    Returns the device context passed to the Mojo kernel by the C++ runtime.
  }];
  let arguments = (ins M_DeviceRefAttr:$device);
  let results = (outs AnyType:$results);

  let assemblyFormat = [{
    attr-dict `:` type($results)
  }];
}

def MOGG_ToKGENOp: MOGG_Op<"to_kgen", [Pure,
  DeclareOpInterfaceMethods<MOGG_MojoObjectInfosInterface>]> {
  let summary = "Converts a MOGG value to a KGEN value";
  let description = [{
    Converts a mogg/mo value to a kgen value. Includes the shape and optionally,
    the stride parameter that appears in the kernel.
  }];

  let arguments = (ins AnyType:$input,
    OptionalAttr<Builtin_StringAttr>:$staticTensorSpecName,
    OptionalAttr<DictionaryAttr>:$typeMethodsDict,
    UnitAttr:$isRegisterType,
    // Whether something was a pure input, mutable input, or output tensor
    // Denotes the underlying tensor type for our supported collection types as
    // well (that is VariadicTensors and lists of tensors)
    // Value must be one of {"InputTensor", "OutputTensor", "MutableInputTensor"}
    // TODO: Might want to look into making this a proper enum
    OptionalAttr<Builtin_StringAttr>:$tensorIOType
  );
  let results = (outs AnyType:$results);

  let extraClassDeclaration = [{
    /// Necessary to implement MojoObjectInfosInterface.
    Value getKGENMojoValueImpl() { return getResults(); }

    /// Returns the ParamParamDeclRefAttr associated to the staticTensorSpec.
    /// Or asserts if staticTensorSpecName is null or the definition is not found
    KGEN::ParamDeclRefAttr getStaticTensorRefAttr();
  }];

  let hasVerifier = 1;
}

def MOGG_ToMOGGOp: MOGG_Op<"to_mogg", [Pure,
  DeclareOpInterfaceMethods<MOGG_MojoObjectInfosInterface>]> {
  let summary = "Converts a KGEN value to a MOGG value";
  let description = [{
    Converts a KGEN value to a MOGG value. This is the converse of to_kgen. It
    helps do partial lowering of the IR.
  }];

  let arguments = (ins AnyType:$input,
    OptionalAttr<DictionaryAttr>:$typeMethodsDict,
    UnitAttr:$isRegisterType
  );
  let results = (outs AnyTypeOf<[MOGG_Simd, MO_Tensor, MOGG_Indices, MOGG_Shape, MO_Opaque]>:$results);

  let extraClassDeclaration = [{
    /// Necessary to implement MojoObjectInfosInterface.
    Value getKGENMojoValueImpl() { return getInput(); }
  }];

  let hasCanonicalizeMethod = 1;
}

def MOGG_TensorGetShapeOp: MOGG_Op<"tensor.shape", [
  Pure,
]> {
  let summary = "Gets the shape of a tensor";
  let arguments = (ins TypeAttr:$tensor);
  let results = (outs MOGG_Shape:$shape);

  let assemblyFormat = [{
    `<` $tensor `>` attr-dict
  }];
}

def MOGG_AssertOp: MOGG_Op<"assert"> {
  let summary = "Placeholder for inserting a try catch block";
  let description = [{
    Placeholder operations that will eventually be lowered to a try catch block.
    This is necessary because mogg.kernel can't contain the try catch block
    until after JITing.
  }];

  let arguments = (ins AnyType:$condition, AnyType:$errorHandle);
}

def MOGG_LoadCompiledKernelOp : MOGG_LowerableOp<"load_compiled_kernel"> {
  let summary = "Loads pairs of (symbol, compiled object) into the runtime "
                "for a given device.";
  let description = [{
    For each of the input symbol/resource pair, associates the symbol with
    the resource (a blob of compiled kernel object) in the runtime (so the
    symbol names can be used by other ops like `ExecuteCompiledOp`).

    The compiled objects are with respect to the op's device attribute. The
    final representation for the compiled objects, and how they are registered
    with the runtime, depends on the device.
  }];

  let arguments = (ins ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value,
    DefaultValuedAttr<BoolAttr, "false">:$debuginfo
  );
}


def MOGG_ExecuteCompiledOp : MOGG_Op<"execute_kernel", [
   DeclareOpInterfaceMethods<MO_ParamDeclaration>,
   MO_HasDeviceInterface,
   MO_ExplicitDevice
]> {
  let summary = "Executes a compiled kernel on a given device.";
  let description = [{
    Executes a compiled kernel with the given name and operands on the given
    device.

    This will use the kernel name to look up the compiled object (loaded by
    `LoadCompiledKernelOp`), and call into the kernel with the lowered operands and
    other values required by the ABI.

    The `usedParams` attribute represent unique shape parameters used inside the
    compiled kernel; they will be lowered to runtime values and passed into the
    kernel, which will use them to reconstruct the tensor shapes.

    `outputParamDecls` allows this compiled operation to define a parameter
    values. These are exported into MGP land where they will be translated
    into calls into the parameter system.

    If `isAsync` is true, the kernel is implemented using the Mojo async/await
    scheme. Otherwise the kernel is synchronous.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$kernelName,
                       ArrayAttr:$usedParams,
                       MO_ParamDecls:$outputParamDecls,
                       StrAttr:$summary,
                       BoolAttr:$isAsync,
                       ArrayAttr:$constantResources
                       );
  let results = (outs Variadic<AnyType>:$results);
}

def MOGG_KernelStubOp : MOGG_Op<"kernel.stub", [
  IsolatedFromAbove,
  Symbol,
  SingleBlock,
  FunctionOpInterface,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>,
]> {
  let summary = "Op that outlines a kernel";
  let description = [{
    This op defines a kernel that will eventually be compiled to Mojo text.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$summary,
                       OptionalAttr<BoolAttr>:$isAsync,
                       DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
                       MOGG_ABIKind:$abi
                       );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    Region *getCallableRegion() { return &getBodyRegion(); }
    ArrayRef<Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr attrs){}
    void setResAttrsAttr(mlir::ArrayAttr attrs){}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def MOGG_TensorInitOp : MOGG_Op<"tensor.__init__", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Represents the construction of a tensor";
  let description = [{
    Takes a data pointer and a list of dimensions. Will lower this into a call
    to the constructor of a ManagedTensorSlice.
  }];

  let arguments = (ins MGP_DeviceBufferType:$buffer,
                       Variadic<AnyInteger>:$dims,
                       BoolAttr:$mut,
                       MO_IOKind:$io,
                       BoolAttr:$exclusive
                  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncReadyOp : MOGG_Op<"async.ready", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>
]> {
  let results = (outs ChainType:$output);

  let summary = "Represents a chain output placeholder";
  let description = [{
    This is to help resolve any chain dependencies for data-dependent operations
    and help elide chains during fusion when lowering down to a compiled region.
  }];

  let assemblyFormat = [{
    attr-dict `:` type(results)
  }];
}

/// TODO(GEX-2434): Wrap kernel body arguments in a mogg.async<T> to fully model
/// the fact that they are not values but pointers that must be unpacked.
def MOGG_AsyncUnpackOp : MOGG_Op<"async.unpack", [
  Pure,
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelArgumentConsumer>,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let summary = "Unpacks an async value and returns it as sync";
  let description = [{
    The values outside of a `mogg.generic.compiled.region` are asynchronous.
    This operation unpacks them and returns values of the same type. This could
    have been an implicit behaviour but this op is meant to formalize the
    process of unpacking which will actually happen.

    Note: The real types of the kernel operands should be pointers but this is
    not reflected in the IR. This is why this operation has the same input and
    output types.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncDelOp : MOGG_Op<"async.__del__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let arguments = (ins AnyType:$input);
  let summary = "Decrements the async reference of the underlying object";
  let description = [{
    This is the 1:1 mirror of mogg.async.unpack. It must be deref an async value
    after its last use.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncPackOp : MOGG_Op<"async.pack", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>
]> {
  let arguments = (ins AnyType:$input,
                   Optional<MGP_DeviceContextType>:$deviceCtx);
  let results = (outs AnyType:$output);
  let summary = "Packs an sync value and returns it as async";
  let description = [{
    This is the opposite of the unpacking operation. It packs synchronous values
    into asynchronous values.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}


def MOGG_AsyncPackBorrowOp : MOGG_Op<"async.pack.borrow", [
  Pure,
  SameOperandsAndResultType,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let summary = "Borrows the input tensor ref value";
  let description = [{
    When another operation returns an async value, we add this borrow to keep
    track of the fact that the returned async value needs to be borrowed by the
    output of the kernel.

    For example, for an op like `mgp.buffer.get_cached`, the returned value will
    be an AnyAsyncValueRef. If that value is immediately used by an output op,
    at some point during the Mojo emission, we will have to grab the output from
    the implicit void** (array of inputs and outputs) and assign to it this
    value. This is effectively a borrow. The reason we don't do this in the op
    that returns the async value is because this is re-usable. If a user of this
    operation is not an output op, it should just use the input. This is done as
    a canonicalization step. Also, if this operation has no user, it will be
    folded. This is why it isn't pure at this time.

    Note that this operation is effectively a symptom of an incomplete type
    system in generic compiled regions. In reality, we should explicitly model
    the void** input/output array and mark types as mogg.async<T>. This way, the
    signature of this borrow op would instead be two inputs (borrow and
    borrowee) and there would be no output op in a kernel.

    Another similar op has this issue: mogg.async.ready. It does not create a
    chain but rather gets the kernel chain from the void** and sets it to ready.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncPackBorrowV2Op : MOGG_Op<"async.pack.borrow.v2", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
]> {
  let arguments = (ins AnyType:$input, AnyType:$mem);
  let results = (outs AnyType:$output);
  let summary = "v2 version of AsyncPackBorrowOp";

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_InlineConstantBufferOp : MOGG_Op<"inline_constant_buffer", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Represents a constant buffer value inlined inside the kernel";
  let description = [{
    Takes a the raw elements attrs and returns a buffer (pointer) to the constant.
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[M_ArrayElementsAttr,
                  Builtin_DenseResourceElementsAttr,
                  M_AlignedBytesAttr]>]
    >:$value);
  let results = (outs MGP_DeviceBufferType:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_EmptyBufferOp : MOGG_Op<"empty_buffer", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Represents an empty buffer (actually of size 1) allocated on the stack";
  let description = [{
    This op is used to build empty tensors objects passed to some kernels after fusion.
  }];

  let results = (outs MGP_DeviceBufferType:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_TensorFusedOp : MOGG_Op<"tensor.fused", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Build a Mojo FusedInput / FusedOutput ManagedTensorSlice object";
  let description = [{
    This operation takes a fusion `lambda`, a `tensor` object, and build a new tensor bind with the corresponding lambda.
    `lambda` must be the output of a `BindLambdaOp`.

    In most situations, `tensor` is the input of the BindLambdaOp.
    Examples:
    ```
    %lambda = mogg.bind (%tensor) output_lambda (%arg12: f32, %arg13: !pop.array<4, index>) {
        mogg.output %arg12, %arg13 : f32, !pop.array<4, index>
      }: (!mo.tensor<[4, 5, 1, 3], f32, {io = 0 : i32}>) -> !mo.tensor<[4, 5, 1, 3], f32>
    %fused_tensor = mogg.tensor.fused(%lambda, %tensor) : (!mo.tensor<[4, 5, 1, 3], f32>, !mo.tensor<[4, 5, 1, 3], f32>) -> !mo.tensor<[4, 5, 1, 3], f32>
    ```

    For some fusion types (eg input fusion), the kernel must take an empty / reshaped tensor.
    This is represented using this op
    - the lambda will load using the `fusedTensor`
    - the kernel call will take the `emptyTensor`.

    Example:
    ```
      %lambda = mogg.bind (%tensor) input_lambda (%arg8: f32, %arg9: !pop.array<4, index>) {
        mogg.output %arg8, %arg9 : f32, !pop.array<4, index>
      }: (!mo.tensor<[4, 5, 2, 3], f32>) -> !mo.tensor<[4, 5, 2, 3], f32>
      %empty_buff = mogg.empty_buffer : !mgp.buffer
      %empty_tensor = mogg.tensor.__init__(%empty_buff) {io = 1 : i32, mut = false} : (!mgp.buffer) -> !mo.tensor<[4, 5, 2, 3], f32>
      %fused_tensor = mogg.tensor.fused(%lambda, %empty_tensor) : (!mo.tensor<[4, 5, 2, 3], f32>, !mo.tensor<[4, 5, 2, 3], f32>) -> !mo.tensor<[4, 5, 2, 3], f32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[MO_Tensor, MO_Buffer]>:$lambda,
                       AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{

    BindLambdaOp getBindLambdaOp() {
      return cast<BindLambdaOp>(getLambda().getDefiningOp());
    }
  }];

  let hasVerifier = 1;
}

def MOGG_TensorReshapeOp : MOGG_Op<"tensor.reshape", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Reshape a ManagedTensorSlice object into a new aliased tensor with a different shape";
  let description = [{
    Reshape the ManagedTensorSlcie `tensor` into a new tensor.
    If the output shape has dynamic dimensions, they must be specified with `dynamic_dims`.
    Both `tensor` and `result` must have a compatible shape.
    This op is used to represent the reshape operation before the kernel call with a fused output lambda with an affine operation.

    Example:
    ```
    %lambda = mogg.bind (%output) output_lambda (%arg8: f32, %arg9: !pop.array<2, index>) {
      %17 = mogg.call "split_dim_indices"(%arg9, %8) {axis = 0 : index} : !pop.array<2, index>, !mo.scalar<si64> -> !pop.array<3, index>
      mogg.output %arg8, %17 : f32, !pop.array<3, index>
    }: (!mo.tensor<[2, 2, 1], f32, {io = 0 : i32}>) -> !mo.tensor<[4, 1], f32>
    %reshaped = mogg.tensor.reshape(%output) : (!mo.tensor<[2, 2, 1], f32, {io = 0 : i32}>) -> !mo.tensor<[4, 1], f32>
    %fused = mogg.materialize_fused_tensor(%lambda, %reshaped) : (!mo.tensor<[4, 1], f32>, !mo.tensor<[4, 1], f32>) -> !mo.tensor<[4, 1], f32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor,
                       Variadic<AnyInteger>:$dynamic_dims);
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_ConstantScalarOp : MOGG_Op<"constant.scalar", [Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>]> {

  let summary = "create a scalar constant value";
  let description = [{


    Example:

    ```mlir
    %cst2 = mogg.constant.scalar {value = 2 : si64} : !mo.scalar<si64>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs MO_Scalar:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_YieldOp
    : MOGG_Op<"yield",
      [Terminator, HasParent<"MOGG::TryOp">]> {
  let summary = "MOGG yield operation";
  let description = [{
    The "mogg.yield" operation specifies the output values for a mogg.try / except statement.
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def MOGG_TryOp : MOGG_Op<"try", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>
]> {
  let summary = "Op that represent a try / except statement";
  let description = [{
    This op defines a try / except statement that will be lowered to mojo.
  }];
  let regions = (region SizedRegion<1>:$tryRegion, SizedRegion<1>:$exceptRegion);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $tryRegion
    `except` $exceptRegion
    attr-dict `:` type($outputs)
  }];
}

def MOGG_AsyncErrorOp : MOGG_Op<"async.error", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let arguments = (ins MOGG_Error:$error);
  let results = (outs AnyType:$output);

  let summary = "Represents setting an error on a kernel output";
  let description = [{
    This represent setting an error on a chain or non-tensor output of a kernel
    and is emitted in the except block of a raising kernel.

    ```
    mogg.try {
     ...
    } except (%err: !mogg.Error) {
      %chain = mogg.async.error %err : !mgp.chain
      %dim0 = mogg.async.error %err : !si64
      mogg.yield %chain, %dim0
    }
    ```

    This op gets lowered to a mojo call to `mogg_async_error`, which assigns
    the error to the corresponding AsyncValue.
  }];

  let assemblyFormat = [{
    $error attr-dict `:` type(results)
  }];
}

def MOGG_CallShapeOp : MOGG_KernelTerminatorOp<"call.shape", [
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isRaising"]>,
  Pure
]> {
  let summary = "Call a shape function";
  let description = [{
    This operation calls a shape function with the given name and operands,
    returning a shape value via the kernel terminator interface rather than as an SSA value.

    This follows the same pattern as other call operations like mogg.call.execute and
    mogg.call.foreach, where the return value is accessed through getKernelReturnOutput().

    Example:
    ```mlir
    mogg.call.shape["mo.avg_pool"](%input, %kernel_size, %strides, %dilations, %paddings) :
      (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>,
       !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> ()
    ```
  }];

  let arguments = (ins
    StrAttr:$shapeFuncName,
    TypeAttr:$retType,
    UnitAttr:$isThrows,
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Scalar, AnyTuple]>>:$operands
  );
  let assemblyFormat = [{
    `[` $shapeFuncName `]` `(` $operands `:` type($operands) `)` `->` $retType attr-dict
  }];
}

def MOGG_ShapeResultGetItemOp : MOGG_Op<"shape_result.__getitem__", [Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "Gets a single dimension from a shape function result";
  let description = [{
    This operation extracts a single dimension at the specified index from a
    !mogg.shape_func_result<rank> value.

    Example:
    ```
    %dim0 = mogg.shape_result.__getitem__(%shape, 0) : (!mogg.shape_func_result<4>) -> si64
    %dim1 = mogg.shape_result.__getitem__(%shape, 1) : (!mogg.shape_func_result<4>) -> si64
    ```
  }];

  let arguments = (ins
    MOGG_ShapeFuncResult:$shape,
    Builtin_IntegerAttr:$index
  );
  let results = (outs SI64:$result);

  let assemblyFormat = [{
    `(` $shape `,` $index `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MOGG_UnpackShapeOp : MOGG_Op<"unpack_shape", [Pure]> {
  let summary = "Unpacks a shape value into individual dimensions";
  let description = [{
    This operation takes a !mogg.shape_func_result<rank> value and unpacks it into individual
    si64 dimension values. The number of results is variadic and corresponds
    to the rank of the shape.

    Example:
    ```
    %dims:3 = mogg.unpack_shape(%shape) : (!mogg.shape_func_result<3>) -> (si64, si64, si64)
    ```
  }];

  let arguments = (ins MOGG_ShapeFuncResult:$shape);
  let results = (outs Variadic<SI64>:$dims);

  let assemblyFormat = [{
    `(` $shape `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MOGG_ShapeInitOp : MOGG_Op<"shape.__init__", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>
]> {
  let summary = "Construct a mojo shape object";
  let description = [{
    Given a list of dimensions, build a mojo shape object (IndexList)
    The op supports dynamic shapes by including both the static shape attribute and a list of dynamic dimensions.

    Example:
    ```
      %shape = mogg.shape.__init__() {shape = #mosh<ape[1, 2, 3, 4]> : !mosh.ape} : (si64) -> !mogg.shape
      %dyn_shape = mogg.shape.__init__(%d0_val) {shape = #mosh<ape[2, D0, 3]> : !mosh.ape} : (si64) -> !mogg.shape
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>:$dynamic_dims,
                       MOSH_ShapeAttr:$shape);
  let results = (outs MOGG_Shape:$result);

  let assemblyFormat = [{
    `(` $dynamic_dims `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncPlaceholderAnyAsyncValueRefOp : MOGG_Op<"async.placeholder.AnyAsyncValueRef", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let results = (outs PointerOf<PointerOf<KGEN_NoneType>>:$result);
  let summary = "Returns an empty AsyncRT::AnyAsyncValueRef";
  let description = [{
    Required to serve as the borrowee for mogg.async.pack.borrow when the value
    is coming from a mgp.buffer.constant. The content of the placeholder is
    initialized to be nullptr.
  }];

  let assemblyFormat = [{attr-dict}];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
