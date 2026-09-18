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

include "GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/MOGGDialect/MOGGEnums.td"
include "GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GraphCompiler/MOGGDialect/MOGGAttrs.td"
include "GraphCompiler/MOGGDialect/MOGGTypes.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/MODialect/MOEnums.td"
include "GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/LITDialect/LITTypes.td"
include "Mojo/POPDialect/POPTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"


// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []> :
  Op<MOGGDialect, mnemonic, traits>;

class MOGG_LowerableOp<string mnemonic, list<Trait> traits = []> :
  MOGG_Op<mnemonic, traits # [
    DeclareOpInterfaceMethods<LowerableToMGPInterface, ["lower"]>,
  ]>;

class MOGG_KernelTerminatorOp<string mnemonic, list<Trait> traits = []> :
  MOGG_Op<mnemonic, traits # [
    DeclareOpInterfaceMethods<MOGG_KernelTerminator>,
    Terminator,
  ]>;

def DimensionType : ScalarOfDType<BoundAnd<DTypeConstraint<"*$_self == KGEN::KGENDType::si64">>, "!kgen.simd<1, si64>">,
    BuildableType<"KGEN::SIMDType::get(1, KGEN::DTypeConstantAttr::get("
                                   "$_builder.getContext(), KGEN::KGENDType::si64))">;

//===----------------------------------------------------------------------===//
// Internal helper ops
//===----------------------------------------------------------------------===//

def MOGG_ConcatOp : MOGG_LowerableOp<"concat", [
  MO_HasDeviceInterface,
  MO_SingleDevice,
  Pure,
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

def MOGG_SliceOp : MOGG_Op<"slice", [
  MO_HasDeviceInterface,
  MO_SingleDeviceWithHostOperands<["start", "stop"]>,
  Pure,
]> {
  let summary = "Contiguous zero-copy slice op lowerable to mgp.tensor.slice";
  let description = [{
    Represents a contiguous slice of a tensor that is known to be lowerable
    without data copies. Unlike mo.slice, this op is not lowered to a kernel
    by MOToMOGG. During FuseViews it is temporarily lowered to a mogg.kernel
    for fusion and raised back if unfused. It is finally lowered to
    mgp.tensor.slice in MOGGToMGP.
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndexScalarIfRanked:$start,
    MO_IndexScalarIfRanked:$stop
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// AsTupleOp
//===----------------------------------------------------------------------===//

def MOGG_AsTupleOp : MOGG_Op<"as_tuple", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
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

def MOGG_OutputOp : MOGG_Op<"output", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ReturnLike,
  Terminator,
]> {
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

def MOGG_CallStubOp : MOGG_Op<"call.stub", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  MOGG_StubCallOpInterface,
]> {
  let summary = "Calls into an outlined stub";
  let description = [{}];
  let arguments = (ins
    FlatSymbolRefAttr:$name,
    Variadic<AnyType>:$inputs,
    OptionalAttr<I64Attr>:$affinity_device_id
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = "$name ($inputs^ `:` type($inputs))? (`->` type($outputs)^)? attr-dict";

  let extraClassDeclaration = [{
    KernelStubOp getCallee();
  }];
}

def MOGG_CallStubAsyncOp : MOGG_Op<"call.stub.async", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  MOGG_StubCallOpInterface,
]> {
  let summary = "Calls into an outlined stub asynchronously";
  let description = [{
    Dispatches an async call to an outlined kernel stub using Mojo-native
    `create_task`. When `affinity_device_id` is set, the emitter passes a
    `desired_worker_id = task_id_for_device(affinity_device_id)` to the
    AsyncRT scheduler to pin the task to a CPU core with NUMA affinity for
    that GPU. When absent the standard `create_task` form (no affinity hint)
    is emitted.

    The device ordinal is stored as a plain integer rather than as a
    DeviceRefAttr because `mogg-emit-mojo` internally runs
    `stripDeviceIdsAndDynamicShapes` (which zeroes DeviceRefAttr IDs for
    deduplication) before emitting Mojo text. A plain I64Attr is unaffected
    by that transformation.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$name,
    Variadic<AnyType>:$inputs,
    OptionalAttr<I64Attr>:$affinity_device_id
  );
  let results = (outs MOGG_TaskHandle:$task_handle);
  let assemblyFormat = "$name (`(` $inputs^ `:` type($inputs) `)`)? `->` type(results) attr-dict";

  let extraClassDeclaration = [{
    KernelStubOp getCallee();
  }];
}

def MOGG_CallForeachOp : MOGG_KernelTerminatorOp<"call.foreach", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "call the registered foreach kernel";
  let description = [{
    This represents a call to the mojo implementation of `foreach` function. If
    a mogg.kernel terminates with the operation, it also implies that the kernel
    is an element-wise kernel.
  }];

  let arguments = (ins
    StrAttr:$kernelSymName,
    MO_Tensor:$input,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$output,
    Variadic<AnyTypeOf<[MOGG_Context, AnyTuple]>>:$context,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );

  let assemblyFormat = [{
    `[` $kernelSymName `]` `input` `(` $input `)` `output` `(` $output `)` (`context` `(` $context^ `)`)? `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallExecuteOp : MOGG_KernelTerminatorOp<"call.execute", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
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

  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    OptionalAttr<TypeAttr>:$regRetType,
    UnitAttr:$isThrows,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );

  let assemblyFormat = [{
    `[` $kernelSymName `]` (`inputs` `(` $inputs^ `:` qualified(type($inputs)) `)`)?
                        (`outputs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
                        (`->` $regRetType^)? attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallBuildGraphOp : MOGG_Op<"call.build_graph", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Record a kernel's `build_graph` function into a device-graph builder";
  let description = [{
    The device-graph counterpart of `mogg.call.execute`. Instead of dispatching
    the kernel's `execute` method against a `!mogg.context`, this op invokes the
    kernel's `build_graph` method against a `!mgp.device_graph.builder`, which
    records a node into the graph under construction rather than running the
    kernel immediately.

    The tensor signature (`inputs` / `outputs`) matches the `execute` call the
    op was synthesized from; only the launch argument differs -- a builder in
    place of a context.

    The op is node-free: it records into the builder but produces no SSA result.
    Node ordering is expressed entirely by the enclosing
    `mgp.device_graph.dependencies` op (which lowers to
    `DeviceGraphBuilder.region(work, dependencies=...)`), so no
    `!mgp.device_graph.node` is threaded through the kernel-facing ABI.

    Unlike `mogg.call.execute`, this is not a kernel terminator and is not
    wrapped in `mogg.with_trace`.

    Example:
    ```mlir
    mogg.call.build_graph["linalg_kernels.__init__.Matmul"]
        inputs(%in, %wt : !mo.tensor<...>, !mo.tensor<...>)
        outputs(%out : !mo.tensor<...>)
        builder(%builder)

    // Lowered to
    Matmul.build_graph[...](out, in, wt, builder)
    ```
  }];

  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    UnitAttr:$isThrows,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$outputs,
    MGP_DeviceGraphBuilderType:$graphBuilder,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );

  let assemblyFormat = [{
    `[` $kernelSymName `]` (`inputs` `(` $inputs^ `:` qualified(type($inputs)) `)`)?
                        (`outputs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
                        `builder` `(` $graphBuilder `)` attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallMaterializeOp : MOGG_KernelTerminatorOp<"call.materialize", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
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

  let arguments = (ins
    StrAttr:$kernelSymName,
    AnyType:$input,
    MO_Tensor:$output,
    Variadic<AnyTypeOf<[MOGG_Context, AnyTuple]>>:$context,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );

  let assemblyFormat = [{
    `[` $kernelSymName `]` `input` `(` $input `)` `output` `(` $output `)` (`context` `(` $context^ `)`)? `:` functional-type(operands, results) attr-dict
  }];
  let hasVerifier = 1;
}


def MOGG_CallElementwiseOp : MOGG_Op<"call.elementwise", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ParentOneOf<["MOGG::BindLambdaOp", "MOGG::CallForeachRegionOp"]>,
  Pure,
]> {
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
  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `[` $kernelSymName `]` `(` $inputs `)` attr-dict `:` type($inputs) `->` type($outputs)
  }];
}

def MOGG_CallViewOp : MOGG_Op<"call.view", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure,
]> {
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
  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    Variadic<AnyType>:$inputs,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs MO_Tensor:$output);

  let assemblyFormat = [{
    `[` $kernelSymName `]` `(` $inputs `)` attr-dict `:` type($inputs) `->` qualified(type($output))
  }];
}

def MOGG_AsScalarOp : MOGG_Op<"as_scalar", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Converts a tensor that contains one element to a scalar";
  let description = [{
    Given a tensor with one element, ie with a shape of [] or [1], etc, convert
    it to a scalar because the kernel using it expects it as a scalar in Mojo.
  }];

  let arguments = (ins MOGG_ScalarOfAnyRank:$input);
  let results = (outs MO_Scalar:$result);
  let hasVerifier = 1;
}

def MOGG_KernelOp : MOGG_LowerableOp<"kernel", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  IsolatedFromAbove,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  SingleBlock,
]> {
  let summary = "A representation of a kernel which can be fused into and compiled.";
  let description = [{
    This kernel represents a compiled region that can be fused with other
    kernels. It uses BindLambdaOp operations within its body to define
    fusion interfaces for inputs and outputs.

    The kernel takes tensor inputs and an optional chain, and produces
    tensor outputs and optionally a chain. Fusion interfaces are defined
    using `mogg.bind` operations within the kernel body.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Buffer, Index]>>:$inputs,
    Optional<MO_Chain>:$chain,
    MO_ParamDecls:$outputParamDecls,
    MOGG_FusionInfoAttr:$fusionInfo,
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
      "FusionInfoAttr":$fusionInfo,
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
def MOGG_LoadTensorOp : MOGG_Op<"tensor.load", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  HasParent<"MOGG::BindLambdaOp">,
]> {
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
     def simd_load_from_managed_tensor_slice(...)
     ```

  }];

  let arguments = (ins
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor,
    POP_ArrayType:$indices
  );
  let results = (outs AnyType:$outputs);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}


def MOGG_BindLambdaOp : MOGG_Op<"bind", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  SingleBlock,
]> {
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

  let arguments = (ins
    MOGG_BindLambdaKind:$bindKind,
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer]>>:$fusibleTensors
  );

  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$output);
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


def MOGG_GenericCompiledRegionOp : MOGG_Op<"generic.compiled.region", [
  DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
  DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
  DeclInterface,
  IsolatedFromAbove,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  SingleBlockImplicitTerminator<"MOGG::OutputOp">,
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
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    OptionalAttr<BoolAttr>:$isAsync,
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canFuseIntoOther,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canBeFusedInto,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$canDuplicate,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isStructuredKernel
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<Kernel>($operands, type($operands), type($results), $body, $outputParamDecls) custom<FusionInfo>($fusionInfo) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Returns the KGEN input parameters this depends on. Necessary to
    /// implement the DeclInterface.
    ArrayRef<ParamDeclAttr> getInputParams() { return {}; }

    /// Return the single body block for the operation.
    Block &getBodyBlock() { return getBody().front(); }

    /// Return the `mogg.output` op terminating the body block.
    MOGG::OutputOp getBodyTerminator();
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

def MOGG_CallAffineOp : MOGG_Op<"call.affine", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasParent<"MOGG::BindLambdaOp">,
  Pure,
]> {
  let summary = "Represents a mojo call to an affine remapping function";
  let description = [{
    This represents a call to a mojo function to remap the indices argument inside a fusion lambda.
    This op is inserted by the Graph Compiler during FuseReshapesIntoEpiloguesPass.

    Example:
    ```mlir
    mogg.call.affine "insert_index"(%arg) {axis = 0 : index, value = 0 : index} : !pop.array<2, index> -> !pop.array<3, index>
    ```
  }];

  let arguments = (ins
    StrAttr:$kernelSymName,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `[` $kernelSymName `]` `(`($operands^)?`)` ($kernel_param^)? attr-dict
      (`:` type($operands)^)? `->` type($result)
  }];
}

def MOGG_ToMGPFromOpaqueOp : MOGG_Op<"to_mgp_from_opaque", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$dataPointer);
  let summary = "Get the opaque pointer from the kgen value";
  let description = [{
    `input` is a KGEN object, and `dataPointer` is an !mgp.opaque
    Get the opaque pointer from the kgen opaque tensor.
    At the implementation level, it's just a bitcast / rebind / no-op.
    But it helps preserving the correct types in the IR during lowering.

    In emit-mojo path, this op is a no-op that converts mo.opaque to mgp.opaque
    for graph outputs. The emitMojo implementation just passes through the value.
    The op is folded away if input and output types become the same.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let hasFolder = 1;
}

def MOGG_AlignmentPlaceholderOp : MOGG_Op<"alignment_placeholder", [
  DeclareOpInterfaceMethods<ParamOpInterface>,
]>, OneParamDecl {
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

  let arguments = (ins
    AnyType:$operand,
    KGEN_ParamDeclAttr:$paramDecl
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
      functional-type(operands, results)
  }];

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

  let arguments = (ins
    ArrayAttr:$symbols,
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[Builtin_DenseResourceElementsAttr]>]
    >:$value,
    DefaultValuedAttr<BoolAttr, "false">:$debuginfo
  );
}


def MOGG_ExecuteCompiledOp : MOGG_Op<"execute_kernel", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
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

  let arguments = (ins
    Variadic<AnyType>:$operands,
    FlatSymbolRefAttr:$calleeSymbol,
    ArrayAttr:$usedParams,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    BoolAttr:$isAsync,
    ArrayAttr:$constantResources
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $calleeSymbol `(` $operands `)` custom<FusionInfo>($fusionInfo) attr-dict `:` functional-type($operands, $results)
  }];
}

def MOGG_KernelStubOp : MOGG_Op<"kernel.stub", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  FunctionOpInterface,
  IsolatedFromAbove,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  SingleBlockImplicitTerminator<"MOGG::OutputOp">,
  SymbolName,
  SymbolVisibility,
  Symbol,
]> {
  let summary = "Op that outlines a kernel";
  let description = [{
    This op defines a kernel that will eventually be compiled to Mojo text.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    MOGG_FusionInfoAttr:$fusionInfo,
    OptionalAttr<BoolAttr>:$isAsync,
    DefaultValuedAttr<ArrayAttr, "{}">:$constantResources,
    MOGG_ABIKind:$abi,
    OptionalAttr<ArrayAttr>:$trait_stub_buffer_capture,
    OptionalAttr<ArrayAttr>:$trait_stub_arg_bounds,
    // The following is necessary for -symbol-dce
    DefaultValuedAttr<StrAttr, "\"private\"">:$sym_visibility
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    OutputOp getBodyTerminator();
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

  private:
    mlir::LogicalResult emitMojoMOGGABI(EmitMojoState &state);
    mlir::LogicalResult emitMojoMGPABI(EmitMojoState &state);
    mlir::LogicalResult emitMojoMAPABI(EmitMojoState &state);
  }];
}

def MOGG_TensorInitOp : MOGG_Op<"tensor.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Represents the construction of a tensor";
  let description = [{
    Takes a data pointer and a list of dimensions. Will lower this into a call
    to the constructor of a ManagedTensorSlice.
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$buffer,
    Variadic<SI64Scalar>:$dims,
    BoolAttr:$mut,
    MO_IOKind:$io
  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TensorArgInitOp : MOGG_Op<"tensor_arg.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  AttrSizedOperandSegments,
  Pure,
]> {
  let summary = "Construct a trait-lane tensor argument from its contract";
  let description = [{
    The trait-lane counterpart of `mogg.tensor.__init__`. Where the legacy op
    builds a `ManagedTensorSlice` parameterized by `{io, mut}`, this op carries
    the distilled `#mogg.tensor_arg` contract and (in a later slice) builds the
    composable `TensorArg` struct that satisfies the kernel's trait bound. The
    two ops run side by side; the legacy op is unchanged (D5).

    Dense-slice operands: a device buffer `$buffer` and the dynamic dimensions
    `$dims`, exactly as `mogg.tensor.__init__`. The optional `$device_context`
    operand is defined for the Slice-3 GPU emit path (ctor 2 in D11) but is left
    unpopulated in this slice -- the unconditional context wiring in
    `findExtraKernelArguments` is Slice 3. The fused storage arm (a `mogg.bind`
    lambda in place of `$buffer`) and the `access`-driven fusion work are later
    slices; here `contract.fused` is always false and `contract.access` is
    always `none`.

    The contract is the single source of truth (D1): `role`/`access` are read
    off the attribute, never re-stamped onto the result `!mo.tensor` type (D4).
  }];

  let arguments = (ins
    MGP_DeviceBufferType:$buffer,
    Variadic<SI64Scalar>:$dims,
    Optional<MOGG_Context>:$device_context,
    MOGG_TensorArgContractAttr:$contract
  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncReadyOp : MOGG_Op<"async.ready", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
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

def MOGG_AsyncJoinOp : MOGG_Op<"async.join", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let arguments = (ins Variadic<MOGG_TaskHandle>:$task_handles);

  let summary = "Waits for async task handles and raises on error";
  let description = [{
    Waits for all task handles to complete (via Task.wait()), then checks
    for errors. If any task failed, raises the first error.

    This is a synchronization-only op with no results. Use a separate
    `mogg.async.ready` to produce the output chain for `mogg.output`.

    Example:
    ```mlir
    mogg.async.join(%t0, %t1) : (!mogg.task_handle, !mogg.task_handle)
    ```
  }];

  let assemblyFormat = [{
    `(` $task_handles `)` attr-dict `:` `(` type($task_handles) `)`
  }];
}

def MOGG_AsyncUnpackOp : MOGG_Op<"async.unpack", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  InferTypeOpAdaptor,
  Pure,
  TypesMatchWith<"input is the async wrapper of the result",
                 /*lhs=*/"output", /*rhs=*/"input",
                 "::M::MOGG::AsyncType::get($_self)">,
]> {
  let arguments = (ins MOGG_Async:$input);
  let results = (outs AnyType:$output);
  let summary = "Unpacks an async value and returns it as sync";
  let description = [{
    The values outside of a `mogg.generic.compiled.region` are asynchronous.
    This operation projects an `!mogg.async<T>` to its underlying `T`, modeling
    the fact that the async value is really a handle/pointer that must be
    unpacked before use. It is the inverse of `mogg.async.pack`.

    Only the unpacked element type `T` is printed; the `!mogg.async<T>` operand
    type is inferred from it.
  }];

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($output)
  }];

  let hasFolder = 1;
}

def MOGG_AsyncDelOp : MOGG_Op<"async.__del__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let summary = "Decrements the async reference of the underlying objects";
  let description = [{
    This is the 1:1 mirror of mogg.async.unpack. It must be deref an async value
    after its last use.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_AsyncPackOp : MOGG_Op<"async.pack", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
  TypesMatchWith<"result is the async wrapper of the input",
                 /*lhs=*/"input", /*rhs=*/"output",
                 "::M::MOGG::AsyncType::get($_self)">,
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs MOGG_Async:$output);
  let summary = "Packs an sync value and returns it as async";
  let description = [{
    This is the opposite of the unpacking operation. It injects a synchronous
    `T` value into an `!mogg.async<T>`.

    Only the input element type `T` is printed; the `!mogg.async<T>` result type
    is inferred from it.
  }];

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input)
  }];

  let hasFolder = 1;
}

def MOGG_EmptyBufferOp : MOGG_Op<"empty_buffer", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Represents an empty buffer (actually of size 1) allocated on the stack";
  let description = [{
    This op is used to build empty tensors objects passed to some kernels after fusion.
  }];

  let results = (outs MGP_DeviceBufferType:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MOGG_TensorFusedOp : MOGG_Op<"tensor.fused", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
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

  let arguments = (ins
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$lambda,
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor
  );
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
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
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

  let arguments = (ins
    AnyTypeOf<[MO_Tensor, MO_Buffer]>:$tensor,
    Variadic<SI64Scalar>:$dynamic_dims
  );
  let results = (outs AnyTypeOf<[MO_Tensor, MO_Buffer]>:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_ConstantScalarOp : MOGG_Op<"constant.scalar", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
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

def MOGG_YieldOp : MOGG_Op<"yield", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  ParentOneOf<["MOGG::TryOp", "MOGG::TensorCreatePrologueOp",
               "MOGG::TensorCreateEpilogueOp", "MOGG::WithTraceOp"]>,
  ReturnLike,
  Terminator,
]> {
  let summary = "MOGG yield operation";
  let description = [{
    The "mogg.yield" operation specifies the output values for a mogg.try /
    except statement, the value a `mogg._tensor.create.prologue` body hands back
    to the kernel reading through it, or the value a COMPUTE-lambda
    `mogg._tensor.create.epilogue` body hands back for the kernel to store.
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def MOGG_WithTraceOp : MOGG_Op<"with_trace", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Wraps a kernel terminator in a Trace context manager";
  let description = [{
    Models the `with Trace[...]` clause emitted around the kernel terminator
    in MOGG ABI kernel stubs.  The body region contains exactly one block
    whose terminator is the KernelTerminator for the stub.  This op is
    always considered throwing because Trace.__enter__ raises.
  }];
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict (`:` type($outputs)^)?";
}

def MOGG_ProfileScopeBeginOp : MOGG_Op<"profile_scope.begin", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Opens a trace range for a Graph.profile_scope() span";
  let description = [{
    Lowers to `var t = Trace[...](name, color=...); t.start()`.

    This is `mogg.with_trace`'s region-less sibling. `with_trace` owns its ops
    in a nested region, so a value defined inside it is invisible to later ops
    both in MLIR (dominance) and in the emitted Mojo (`var` is block-scoped).
    Sequence-fused region bodies routinely have SSA values live across the
    scope boundaries this op is meant to bracket, so the range is expressed as
    a begin/end pair of siblings instead, and the bracketed ops stay exactly
    where they were.

    `color`, when present, is the single `Graph.profile_scope(..., color=...)`
    color every op this range covers agreed on (see
    `resolveProfileScopeColor`); absent when the range covers no color at all,
    or when its ops disagree. Region-only by design: `mogg.with_trace`'s
    per-kernel trace name never carries a color, regardless of what
    `Graph.profile_scope()` was given.

    Always considered throwing because `Trace.start()` raises.
  }];
  let arguments = (ins StrAttr:$name, OptionalAttr<StrAttr>:$color);
  let results = (outs MOGG_ProfileScopeRange:$range);
  let assemblyFormat = "$name attr-dict `:` qualified(type($range))";
}

def MOGG_ProfileScopeEndOp : MOGG_Op<"profile_scope.end", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Closes the trace range opened by a `mogg.profile_scope.begin`";
  let description = [{
    Lowers to `t.end()` on the `Trace` variable its `mogg.profile_scope.begin`
    declared. Always considered throwing because `Trace.end()` raises.
  }];
  let arguments = (ins MOGG_ProfileScopeRange:$range);
  let assemblyFormat = "$range attr-dict `:` qualified(type($range))";
}

def MOGG_RaiseOp : MOGG_Op<"raise", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  Terminator,
]> {
  let summary = "Re-raises a kernel error with full context";
  let description = [{
    Emitted as the terminator of the except block in MOGG ABI kernel stubs.
    Formats the caught error with the enclosing stub name, fusion info, and
    Python source traceback, then re-raises it so the caller sees the full
    context.
  }];
  let arguments = (ins MOGG_Error:$error);
  let assemblyFormat = "$error attr-dict";
}

def MOGG_TryOp : MOGG_Op<"try", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
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

  let hasCanonicalizer = 1;
}

def MOGG_AsyncErrorOp : MOGG_Op<"async.error", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
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

    This op gets lowered to a mojo call to `mogg_async_error_value`, which
    returns a fresh error `AnyAsyncValueRef`.
  }];

  let assemblyFormat = [{
    $error attr-dict `:` type(results)
  }];
}

def MOGG_CallShapeOp : MOGG_KernelTerminatorOp<"call.shape", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  Pure,
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
    StrAttr:$kernelSymName,
    TypeAttr:$retType,
    UnitAttr:$isThrows,
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque, MO_Scalar, AnyTuple]>>:$operands,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let assemblyFormat = [{
    `[` $kernelSymName `]` `(` $operands `:` type($operands) `)` `->` $retType attr-dict
  }];
}

def MOGG_ShapeResultGetItemOp : MOGG_Op<"shape_result.__getitem__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Gets a single dimension from a shape function result";
  let description = [{
    This operation extracts a single dimension at the specified index from a
    !mogg.shape_func_result<rank> value.

    Example:
    ```
    %dim0 = mogg.shape_result.__getitem__(%shape, 0) : (!mogg.shape_func_result<4>) -> !kgen.scalar<si64>
    %dim1 = mogg.shape_result.__getitem__(%shape, 1) : (!mogg.shape_func_result<4>) -> !kgen.scalar<si64>
    ```
  }];

  let arguments = (ins
    MOGG_ShapeFuncResult:$shape,
    Builtin_IntegerAttr:$index
  );
  let results = (outs SI64Scalar:$result);

  let assemblyFormat = [{
    `(` $shape `,` $index `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MOGG_ShapeInitOp : MOGG_Op<"shape.__init__", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
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

  let arguments = (ins
    Variadic<SI64Scalar>:$dynamic_dims,
    MOSH_ShapeAttr:$shape
  );
  let results = (outs MOGG_Shape:$result);

  let assemblyFormat = [{
    `(` $dynamic_dims `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_ChainDummyOp : MOGG_Op<"chain.dummy", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Returns a chain to feed MRP primitives";
  let description = [{
    MGP primitives that consume a chain and get lowered into a kernel still need
    a chain internally. During sequence fusion, those input chains may become
    intermediate results. Instead of handling those specially after sequence
    fusion, we always feed them with this dummy op that does absolutely nothing.
    The better approach would be to change the signature of the op when we lower
    it into a kernel. Tracked here in GEX-2764.
  }];

  let arguments = (ins);
  let results = (outs ChainType:$chain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_ChainSinkOp : MOGG_Op<"chain.sink", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Consumes a chain and does nothing with it";
  let description = [{
    This is the same as ChainDummyOp but it consumes a chain for the sake of not
    DCE-ing MGP primitives that lower to kernels but don't end up having a
    consumer for their result.
  }];

  let arguments = (ins ChainType:$chain);
  let results = (outs);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}


//===----------------------------------------------------------------------===//
// KernelDeclOp
//===----------------------------------------------------------------------===//

def MOGG_KernelDeclOp
    : MOGG_Op<"kernel.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares a Mojo kernel's signatures inline in the module";
  let description = [{
    Holds the full function-type signatures (without bodies) for a Mojo kernel
    struct registered under a given MO op name. Multiple `mogg.kernel.decl` ops
    may share the same `registration` value when several structs are registered
    for the same op (e.g. different device implementations).

    All function attributes are optional. A kernel carrying an `update_view`
    signature is implicitly a view kernel.

    `is_user_defined` marks a kernel loaded from a user-supplied package
    rather than a built-in one. MojoRegistry uses this flag to implement
    override semantics: a user-defined decl takes precedence over any
    built-in decl with the same registration name.

    Each function variant (`execute` / `elementwise` / `tile_elementwise` /
    `update_view` / `build_graph`) is a `MOGG_KernelFunctionAttr` that
    bundles its signature, its fn-level `ParamDeclAttr` list, and the
    cached per-result parameter mapping.

    Shape inference is decoupled: the shape function for a registration lives
    in a separate `mogg.shape_func.decl` op looked up by registration string.

    A kernel may carry a `build_graph` signature alongside `execute`. The
    `build_graph` function has the same tensor signature as `execute` but
    takes a `DeviceGraphBuilder` as its launch argument in place of the
    `DeviceContext`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$registration,
    M_DeviceInfoAttr:$registration_device_info,
    OptionalAttr<MOGG_KernelFunctionAttr>:$execute,
    OptionalAttr<MOGG_KernelFunctionAttr>:$elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$tile_elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$update_view,
    OptionalAttr<MOGG_KernelFunctionAttr>:$build_graph,
    ArrayAttr:$struct_params,
    BoolAttr:$is_user_defined,
    OptionalAttr<StrAttr>:$sym_visibility
  );

  let assemblyFormat = [{
    $sym_name `registration` `=` $registration
    `device_info` `=` $registration_device_info
    (`execute` `=` $execute^)?
    (`elementwise` `=` $elementwise^)?
    (`tile_elementwise` `=` $tile_elementwise^)?
    (`update_view` `=` $update_view^)?
    (`build_graph` `=` $build_graph^)?
    `struct_params` `=` $struct_params
    `is_user_defined` `=` $is_user_defined
    attr-dict
  }];
}

def MOGG_PrimitiveDeclOp
    : MOGG_Op<"primitive.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares a primitive Mojo function inline in the module";
  let description = [{
    Records the `FnTypeGeneratorType` signature of a standalone (non-struct)
    Mojo function decorated with `@register_internal`, such as the singleton
    `foreach` driver or the view-materialisation kernel.

    `sym_name` is the fully-qualified Mojo source name of the function
    (e.g. `MOGGPrimitives.MOGGPrimitives.foreach`). `registration` is the
    intrinsic string the function is registered under (e.g.
    `"mogg.elemwise_for_each"` or `"mogg.call.materialize"`). Each
    `@register_internal` site produces exactly one `mogg.primitive.decl`;
    primitives must not be registered under multiple registration strings
    (the MLIR symbol table would reject the duplicate `sym_name`).

    `function` is a `MOGG_KernelFunctionAttr` that bundles the signature
    and the fn-level `ParamDeclAttr` list. Result-side slot-name mappings
    are not populated for primitives.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    StrAttr:$registration,
    MOGG_KernelFunctionAttr:$function
  );

  let assemblyFormat = [{
    $sym_name `registration` `=` $registration `function` `=` $function
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ShapeFunctionDeclOp
//===----------------------------------------------------------------------===//

def MOGG_ShapeFunctionDeclOp
    : MOGG_Op<"shape_func.decl", [SymbolName, SymbolVisibility, Symbol]> {
  let summary = "Declares the shape function for a kernel op registration";
  let description = [{
    Records the signature of the shape function for a given op registration.
    There is at most one `mogg.shape_func.decl` per registration string; the
    import pass enforces this and errors on duplicates.

    `sym_name` is the fully-qualified Mojo source name of the shape function.
    For a struct-internal `shape` method it is `"<struct_sym>.shape"` (e.g.
    `"MOGGBuiltins.MOGGBuiltins.AvgPool.shape"`).

    `registration` is the op registration string (e.g. `"mo.avg_pool"`).

    `function` is a `MOGG_KernelFunctionAttr` that bundles the signature and
    the fn-level `ParamDeclAttr` list. Result-side slot-name mappings are
    populated in strict mode (shape functions always return stdlib `IndexList`).
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    StrAttr:$registration,
    MOGG_KernelFunctionAttr:$function
  );

  let assemblyFormat = [{
    $sym_name `registration` `=` $registration `function` `=` $function
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Advanced-fusion codegen ops
//
// `MAPToMOGG` lowers the `map` dialect into the ops below. Everything ABOVE is
// the legacy MOGG op set, removed once advanced fusion (MAX_GC_USE_ADV_FUSION)
// is the default; new codegen ops are appended here so the cleanup is a clean
// cut at this line. Values are opaque handles (`!mogg.pointer_` / `!mogg.tensor_`
// / `!mogg.dim_`); shape/dtype/mut/io live on the creating op. No lowering is
// wired yet.
//===----------------------------------------------------------------------===//

def MOGG_KernelRegionOp : MOGG_LowerableOp<"_kernel", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExplicitDevice,
  MO_HasDeviceInterface,
  NoTerminator,
]> {
  let summary =
      "A codegen kernel: `!mo.tensor` inputs/outputs outside, `!mogg.pointer_` "
      "inside.";
  let description = [{
    The new-path counterpart of `mogg.generic.compiled.region` / `mogg.kernel`.
    Inputs are `!mo.tensor` (or, for an in-place kernel a `FuseMutableLoads`
    fused a buffer into, `!mo.buffer`) operands and outputs are `!mo.tensor`
    results; inside the body both become `!mogg.pointer_` block arguments
    (outputs turn into externally-allocated operands only later, in
    `MOGGToMGP`). The body's entry-block arguments are, in order: one
    `!mogg.pointer_` per input, one `!mogg.pointer_` per output, then the
    runtime `!mogg.dim_`s (dynamic dimensions), then one `!mogg.context`. Like
    `mogg.kernel` there is no terminator and no `mogg.output`: the output
    pointer block arguments are the outputs. `MAPToMOGG` lowers a
    `map.iter.opaque` into one of these.

    The optional `chain` operand/result models an in-place/mutating kernel,
    mirroring `mogg.kernel`'s and `map.iter.foreach`'s chain support: it is pure
    sequencing plumbing and never gets a body block argument on either side.
    When present, the output chain is always the last entry of `outputs`.

    Like `mogg.kernel.stub`, the op carries a `#M.device_ref` `device` attribute
    (the `MOExplicitDevice` trait) so device assignment and outlining resolve the
    kernel's execution device from the op itself rather than falling back to the
    host.

    `fusionInfo` carries the fusion-tree provenance forwarded from the source
    `map.iter.opaque`; absent only for a kernel region built by hand (e.g. in a
    test) rather than through `MAPToMOGG`.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_Buffer, MO_Opaque]>>:$inputs,
    Optional<MO_Chain>:$chain,
    OptionalAttr<MOGG_FusionInfoAttr>:$fusionInfo,
    MO_ParamDecls:$outputParamDecls,
    // Carried from `map.iter.foreach`/`map.iter.opaque`, which carry them from
    // the op they lower. Without them a kernel whose only observable effect is
    // a mutable store reports none and is dead.
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasReadEffect,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$hasWriteEffect
  );
  let results = (outs Variadic<AnyTypeOf<[MO_Tensor, MO_Chain, MO_Opaque,
                                          MOGG_ShapeFuncResult]>>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns the output chain, or null when there is none -- always the
    /// last entry of `outputs` when present (mirrors
    /// `IterForeachOp::getOutputChain()`/`KernelOp::maybeGetOutChain()`).
    ::mlir::Value getOutputChain();

    /// Lowers this kernel to a `mogg.generic.compiled.region`, registers its
    /// outputs against the graph, and returns the created region -- or a null
    /// `GenericCompiledRegionOp` on error (like a failed `LogicalResult`). The
    /// `LowerableToMGPInterface` `lower()` hook is a thin wrapper around this;
    /// the `mo.parallel` orchestrator calls it directly, once per launch, to
    /// collect the per-launch regions it then fuses into one async region.
    GenericCompiledRegionOp
    lowerToCompiledRegion(::M::MGP::MOGraphLoweringState &loweringState);
  }];
}

def MOGG_ShapeCreateOp : MOGG_Op<"_shape.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Build a codegen shape from a static shape attr + dynamic dims.";
  let description = [{
    New-path counterpart of `mogg.shape.__init__`: materializes a `!mogg.shape`
    from a static `#mosh.ape` and the `!mogg.dim_` handles for its dynamic
    dimensions (in shape order). The dims come from the enclosing `mogg._kernel`
    block arguments.
  }];
  let arguments = (ins Variadic<MOGG_Dim>:$dynamic_dims, MOSH_ShapeAttr:$shape);
  let results = (outs MOGG_Shape:$result);
  let assemblyFormat = [{
    `(` $dynamic_dims `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TensorCreateOp : MOGG_Op<"_tensor.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor from a pointer, shape and static metadata.";
  let description = [{
    New-path counterpart of `mogg.tensor.__init__`: reconstructs a `!mogg.tensor_`
    inside a kernel body from a data pointer and a shape. Element type,
    mutability, io-kind, and the static shape are all attributes (the opaque
    `!mogg.tensor_` and `!mogg.shape` types carry none of them). The `$device`
    records the device kind (cpu vs gpu) so emit can pick the preferred
    alignment locally instead of reaching out to the enclosing kernel stub.
  }];
  let arguments = (ins
    MOGG_Pointer:$ptr,
    MOGG_Shape:$shape,
    MOSH_ShapeAttr:$shapeAttr,
    M_DeviceRefAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io
  );
  // A `!mo.buffer` operand reconstructs as a mutable `!mogg.buffer_`, a
  // `!mo.tensor` as a `!mogg.tensor_`, so `getKernelOperandKind()` resolves
  // `Buffer` vs `Tensor` from the type alone. `MOGG_TensorCreateInterface`'s
  // verifier holds the shared `!mogg.buffer_ => mut` invariant.
  let results = (outs AnyTypeOf<[MOGG_Tensor, MOGG_Buffer]>:$tensor);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TensorCreateEpilogueOp : MOGG_Op<"_tensor.create.epilogue", [
  AttrSizedOperandSegments,
  NoTerminator,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "A fused-output tensor bound to its own epilogue body.";
  let description = [{
    New-path counterpart of `mogg.tensor.fused` for the output (store) side:
    produces a `!mogg.tensor_` (of `$shape`) that writes through its own
    epilogue body, rather than referencing a separate named lambda op. The
    body takes a variadic list of captured tensors plus an arbitrary list of
    extra operands (e.g. an index the callback is evaluated at); the body's
    own scalar parameters are the entry-block arguments. `MAPToMOGG` lowers a
    `map.iter.epilogue` into one of these. Element type / mutability /
    io-kind are attributes, and `$device` records the device kind (cpu vs
    gpu) so emit can pick the preferred alignment locally instead of
    reaching out to the enclosing kernel stub. Scaffolding: not yet wired.

    The body comes in two shapes. A STORE lambda owns its store: the body ends
    with a `mogg._tensor.store` into the store-target capture and has no
    terminator. A COMPUTE lambda leaves the store to the kernel: the body ends
    with `mogg.yield` of the value that store would have written, and the
    store-target capture survives unused so emit still has a slice to bind
    onto. `MAPToMOGG` picks the shape.

    `$computeOutputArg` records whether the consuming kernel declares the output
    this epilogue feeds as a `_FusedComputeOutputTensor`. It is a separate
    question from the body shape: a dtype-changing epilogue, or the
    `enable-compute-lambdas` opt-out, keeps a store body against a
    compute-capable argument. Emit reads it for one purpose, choosing between
    `_bind_to_fused_output` and `_bind_to_fused_compute_output`.
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    Variadic<MOGG_TensorLike>:$captures,
    Variadic<AnyType>:$args,
    Variadic<MOGG_Shape>:$shapeCaptures,
    MOGG_Pointer:$outputPtr,
    M_DeviceRefAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io,
    DefaultValuedAttr<BoolAttr, "false">:$computeOutputArg,
    // Present iff the consuming kernel declares this fused output against a
    // trait bound (`Fused & Output & StoreAccess` / `... & TransformAccess`).
    // Stamped at MAP lowering (where the epilogue -> call edge is explicit);
    // emit reads it to wrap the fusion functor in the conforming trait tensor
    // instead of binding it onto a `ManagedTensorSlice`. Absent for the legacy
    // `FusedOutputTensor` (MTS-API) lane.
    OptionalAttr<MOGG_TensorArgContractAttr>:$contract
  );
  let results = (outs MOGG_Tensor:$tensor);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $shape (`(` $args^ `:` type($args) `)`)?
    (`capturing` `(` $captures^ `:` type($captures) `)`)?
    (`shapes` `(` $shapeCaptures^ `)`)?
    `ptr` $outputPtr `:` type($outputPtr)
    `:` type($shape) `->` type($tensor)
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
}

def MOGG_TensorGetPtrOp : MOGG_Op<"_tensor.get_ptr", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Extract the raw `!mogg.pointer_` backing a codegen tensor.";
  let description = [{
    Reads the raw base pointer of `$tensor`'s `ManagedTensorSlice`. Used by
    `MAPToMOGG` to hand a `mogg._tensor.create.epilogue` the pointer of the
    tensor its body stores into (or, for a mixed-precision accumulating kernel,
    the pointer of the pre-cast accumulation buffer `AllocateAccumulatingBuffers`
    added), so emit can build the kernel's fused-output `ManagedTensorSlice` over
    real storage. An accumulating kernel then has a valid pointer to accumulate
    through, while a non-accumulating one simply ignores it and the lambda does
    the store.
  }];
  // A buffer as well as a tensor: an epilogue that stores into a fused
  // `mo.mutable.store`'s buffer has that buffer as its store target, and
  // `mogg._tensor.load`/`_tensor.store` already take either.
  let arguments = (ins MOGG_TensorLike:$tensor);
  let results = (outs MOGG_Pointer:$ptr);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TensorCreatePrologueOp : MOGG_Op<"_tensor.create.prologue", [
  AttrSizedOperandSegments,
  SingleBlock,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "A fused-input tensor bound to its own prologue body.";
  let description = [{
    Load-side counterpart of `mogg._tensor.create.epilogue`: produces a
    `!mogg.tensor_` (of `$shape`) the consuming kernel reads through its own
    prologue body, rather than from a separately materialized tensor. The
    body's entry-block arguments are the index the callback is evaluated at
    followed by one rebuilt tile per captured tensor, and it hands back the
    value the kernel should see via `mogg.yield`. `MAPToMOGG` lowers a
    `map.iter.prologue` into one of these. Element type / mutability / io-kind
    are attributes, and `$device` records the device kind (cpu vs gpu) so emit
    can pick the preferred alignment locally instead of reaching out to the
    enclosing kernel stub.
  }];
  let arguments = (ins
    MOGG_Shape:$shape,
    Variadic<MOGG_TensorLike>:$captures,
    // Runtime from/to shapes for a dynamic `mogg.index.reshape` in the body,
    // which cannot read them from the static signature. Same role as the
    // epilogue's: they become trailing `IndexList` fields on the fusion
    // functor.
    Variadic<MOGG_Shape>:$shapeCaptures,
    M_DeviceRefAttr:$device,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MO_IOKind:$io,
    // Present iff the consuming kernel declares this fused input against a trait
    // bound (`Fused & Input & LoadAccess`). Stamped at MAP lowering; emit reads
    // it to wrap the fusion functor in `_MoggInputLoadFusionTensor` instead of
    // binding it onto a `ManagedTensorSlice`. Absent for the legacy
    // `FusedInputTensor` (MTS-API) lane.
    OptionalAttr<MOGG_TensorArgContractAttr>:$contract
  );
  let results = (outs MOGG_Tensor:$tensor);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $shape (`capturing` `(` $captures^ `:` type($captures) `)`)?
    (`shapes` `(` $shapeCaptures^ `:` type($shapeCaptures) `)`)?
    `:` type($shape) `->` type($tensor)
    attr-dict-with-keyword $body
  }];
}

def MOGG_TensorStoreOp : MOGG_Op<"_tensor.store", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Store a scalar value into a codegen tensor at an index.";
  let description = [{
    Store path of a fusion callback body: writes `$value` into `$tensor` at
    `$index`.
  }];
  let arguments = (ins MOGG_TensorLike:$tensor, AnyType:$index, AnyType:$value);
  let assemblyFormat = [{
    $tensor `[` $index `]` `=` $value attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($index)) `,` qualified(type($value))
  }];
}

def MOGG_CallOpaqueOp : MOGG_Op<"_call.opaque", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Call a registered opaque kernel's execute function.";
  let description = [{
    New-path counterpart of `mogg.call.execute`: invokes the kernel named by
    `$callee` over `!mogg.tensor_` inputs/outputs. `$device_contexts` wires the
    callee's device context(s) explicitly instead of scavenging them from the
    enclosing kernel stub's block arguments at emit time. It is variadic: a
    single-device kernel takes one `!mogg.context`; a distributed kernel whose
    execute signature takes a `DeviceContextArray` takes one per device, which
    `ConformCallOperands` packs into a single `!mogg.tuple_` of contexts.
    `$hasEpilogueFusion` / `$hasPrologueFusion` record whether an output is a
    fused-epilogue tensor and whether an input is a fused-prologue tensor, so
    emit never inspects an operand's own producer for this. Not a terminator
    (the enclosing `mogg._kernel` body is terminated by `mogg.output`).
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$inputs,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Scalar, MOGG_Tuple,
                        MO_Opaque]>>:$outputs,
    Variadic<AnyTypeOf<[MOGG_Context, MOGG_Tuple]>>:$device_contexts,
    OptionalAttr<DictionaryAttr>:$kernel_param,
    DefaultValuedAttr<BoolAttr, "false">:$hasEpilogueFusion,
    DefaultValuedAttr<BoolAttr, "false">:$hasPrologueFusion,
    // Forwarded from the source op's `MO_SameVariadicOperandSize` trait so
    // `ConformCallOperands` can split a flat operand list into several
    // equal-length variadic groups (e.g. a bundled collective's N peer inputs
    // and N signal buffers). The trait is lost when the MO op becomes a call,
    // so it rides along as this attribute.
    DefaultValuedAttr<BoolAttr, "false">:$hasSameVariadicOperandSize
  );
  // An `!mo.opaque` result is allocated inside the kernel body (see
  // `MOGGKernelOpLoweringState`), so unlike a tensor it has no destination to
  // pass as an `$output`. It leaves through this result instead, and the
  // enclosing `mogg._kernel` forwards it with `mogg.output`.
  let results = (outs Optional<MO_Opaque>:$opaqueResult);
  let assemblyFormat = [{
    $callee (`inputs` `(` $inputs^ `:` type($inputs) `)`)?
            (`outputs` `(` $outputs^ `:` type($outputs) `)`)?
            (`context` `(` $device_contexts^ `:` type($device_contexts) `)`)?
            attr-dict (`->` type($opaqueResult)^)?
  }];
}

def MOGG_ScalarCreateOp : MOGG_Op<"_scalar.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Pack a single-element tensor operand into a scalar operand.";
  let description = [{
    New-path counterpart of `mogg.as_scalar`: `ConformCallOperands`
    inserts this to convert a `!mogg.tensor_` operand that a kernel expects as a
    scalar into a `!mogg.scalar_`. Element type is carried by the source tensor's
    creating op, not restated here.
  }];
  let arguments = (ins MOGG_Tensor:$tensor);
  let results = (outs MOGG_Scalar:$scalar);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TraitCreateOp : MOGG_Op<"_trait.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Re-present a codegen tensor as its conforming trait argument.";
  let description = [{
    Inserted by `ConformCallOperands` between a tensor-producing op and a
    `mogg._call.opaque` / `mogg._call.shape` whenever the callee declares that
    operand against a trait bound (`Tensor` / `DenseTensor` / `TileTensorable`).
    A raw `!mogg.tensor_` reconstructs as a `ManagedTensorSlice`, which conforms
    to none of those bounds; this op's emit wraps it in the conforming
    `_MoggDenseTensorArg` (execute lane) or `_MoggShapeTensorArg` (shape lane,
    `role = none`), keyed on `$contract`. The result keeps the operand's own
    `!mogg.tensor_`/`!mogg.buffer_` kind (a mutable-input trait argument
    reconstructs as a `!mogg.buffer_`), so the call op's operand list and its own
    emit are unchanged -- the wrap is a local, 1:1 lowering here, not
    context-sensitive logic at the call.

    `$device` records the execution device so emit picks the preferred alignment
    and device-passable flavor locally, exactly as `mogg._tensor.create` does,
    without reaching out to the enclosing kernel stub. It wraps a finished
    handle, so it works over any producer -- a direct `mogg._tensor.create` or a
    hoisted view whose layout is not row-major.
  }];
  let arguments = (ins
    MOGG_TensorLike:$tensor,
    M_DeviceRefAttr:$device,
    MOGG_TensorArgContractAttr:$contract
  );
  let results = (outs MOGG_TensorLike:$result);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_TupleCreateOp : MOGG_Op<"_tuple.create", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Pack a run of variadic tensor operands into a tuple operand.";
  let description = [{
    New-path counterpart of `mogg.as_tuple`: `ConformCallOperands`
    inserts this to pack the consecutive `!mogg.tensor_` operands that satisfy a
    single variadic kernel argument into one `!mogg.tuple_`. It also packs the
    per-device `!mogg.context` operands of a distributed kernel into a
    tuple-of-contexts, which emit renders as a `DeviceContextArray`.
  }];
  let arguments = (ins
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Buffer, MOGG_Context]>>:$elements,
    DefaultValuedAttr<MOGG_VariadicABIAttr,
        "::M::MOGG::VariadicABI::Plain">:$variadic_abi);
  let results = (outs MOGG_Tuple:$tuple);
  let assemblyFormat = [{
    `(` $elements `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_RebindOp : MOGG_Op<"_rebind", [Pure,
    DeclareOpInterfaceMethods<EmitsMojoInterface>]> {
  let summary = "A cast between an unpacked async handle and a codegen handle.";
  let description = [{
    Reconciles the type system during the async-unpack transition (e.g.
    `!mgp.buffer` -> `!mogg.pointer_`). For a `!mogg.pointer_` result it emits
    real Mojo, bridging the unpacked `OwnedByteBuffer` to the raw opaque data
    pointer the MAP-ABI stub expects; the `!mogg.context` / `!mogg.dim_` casts
    are code-generation no-ops (the unpacked handle already has the right Mojo
    type).

    TODO (debt): remove once `mogg.async.unpack` folds directly to the typed
    handle; sequence fusion currently expects `async.unpack` to fold, so a
    conversion op is needed to please the type system.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `to` type($output)
  }];
}

def MOGG_BufferConstantOp : MOGG_Op<"buffer.constant", [Pure]> {
  let summary = "A constant buffer, as a raw codegen pointer.";
  let description = [{
    Holds the bytes of a `mo.constant` that `FuseConstantIterOpaqueInputs`
    folded into a `map.iter.opaque` body, where there is no operand for it to
    arrive on. `MAPToMOGG` emits one of these in place of the `mo.constant` and
    reconstructs the tensor from its `!mogg.pointer_` result, exactly as it
    would from an input pointer.

    Unlike `mgp.buffer.constant` this carries the attribute as written, with no
    alignment: the MEF alignment depends on the pipeline's minimum CPU
    alignment, known only in `MOGGToMGP`. That pass rewrites this into an
    `mgp.buffer.constant` plus a `mogg._rebind` back to a pointer, running the
    value through `lowerElementsAttr` on the way and recording the blob in the
    enclosing region's `constantResources`.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs MOGG_Pointer:$pointer);
  let assemblyFormat = [{ $value attr-dict `:` type($pointer) }];
}

def MOGG_TensorLoadOp : MOGG_Op<"_tensor.load", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Load a scalar value from a codegen tensor at an index.";
  let description = [{
    New-path counterpart of `mogg.tensor.load`: the load half of a fusion
    callback body, reading `$tensor` at `$index`. Mirrors
    `mogg._tensor.store`'s shape minus the write side; not `Pure`, since a
    load shouldn't be freely reordered relative to a store through the same
    tensor any more than the store itself is.
  }];
  let arguments = (ins MOGG_TensorLike:$tensor, AnyType:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $tensor `[` $index `]` attr-dict `:`
    qualified(type($tensor)) `,` qualified(type($index)) `->` qualified(type($result))
  }];
}

def MOGG_ConstantScalarV2Op : MOGG_Op<"_constant.scalar", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "A scalar constant inlined into a fusion body as a literal.";
  let description = [{
    New-path counterpart of `mogg.constant.scalar`. It differs only in the
    result type: a builtin scalar (`f32`, `si32`, ...) rather than an
    `!mo.scalar`, which is the type `mogg._tensor.load` already produces, so
    an inlined constant is interchangeable with the load it replaces
    everywhere the new path consumes element values.

    Emits no statement of its own -- the value becomes a Mojo literal at each
    use -- splatted to the enclosing closure's SIMD width so it works as an
    elementwise operand in any position. Not `ConstantLike`: CSE dedups it,
    and the constant hoisting that trait brings is only wanted MAP-side,
    where `map.constant.scalar` carries it.

    Example:
    ```mlir
    %cst = mogg._constant.scalar {value = 2.000000e+00 : f32} : f32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$result);
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

def MOGG_CallForeachRegionOp : MOGG_Op<"_call.foreach", [
  NoTerminator,
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
]> {
  let summary = "Drive a parallel foreach loop over a fused elementwise body.";
  let description = [{
    New-path counterpart of `mogg.call.foreach`: dispatches to the Mojo
    foreach driver named by `$callee` (resolved once, at construction time,
    from the registered `mogg._call.foreach` primitive -- there is no
    per-kernel lookup the way `mogg._call.opaque` has, since this always
    calls the same generic driver) over `$shape` many indices. Unlike the
    opaque-kernel path, the region owns the full per-index computation
    directly: its entry block has one block argument per `$capturing` tensor
    (in order) followed by a trailing index block argument, and the body does
    its own load-compute-store round trip through `mogg._tensor.load` /
    `mogg.call.elementwise` / `mogg._tensor.store`. `MAPToMOGG` lowers a
    `map.iter.opaque` wrapping a `map.iter.foreach.body` into one of these.

    Captured tensors may have a different shape than `$shape` (e.g. a
    broadcast operand) and from each other, so this op deliberately has no
    verifier enforcing shape agreement. A capture is a `!mogg.buffer_`
    wherever in-place fusion folded a load's mutable buffer into a position a
    tensor used to hold. `$device` records the target device
    kind (cpu vs gpu), set once at construction time rather than read back
    from the enclosing kernel stub at emit time.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    MOGG_Shape:$shape,
    MOGG_Context:$context,
    Variadic<MOGG_TensorLike>:$capturing,
    M_DeviceRefAttr:$device
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $callee `shape` `(` $shape `)` `context` `(` $context `)`
            (`capturing` `(` $capturing^ `:` type($capturing) `)`)?
            attr-dict-with-keyword $body
  }];
}

def MOGG_CallShapeV2Op : MOGG_Op<"_call.shape", [
  DeclareOpInterfaceMethods<EmitsMojoInterface, ["isThrows"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "Call a registered shape function, returning its shape by value.";
  let description = [{
    New-path counterpart of `mogg.call.shape`: not a terminator, and yields
    its `IndexList` as an ordinary SSA result. `$kernel_param` carries
    `output_rank`, the rank of `$result`.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    UnitAttr:$isThrows,
    Variadic<AnyTypeOf<[MOGG_Tensor, MOGG_Scalar, MOGG_Tuple]>>:$operands,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs MOGG_ShapeFuncResult:$result);
  let assemblyFormat = [{
    $callee `(` $operands `:` type($operands) `)` `->` type($result) attr-dict
  }];
}

def MOGG_CallElementwiseV2Op : MOGG_Op<"_call.elementwise", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_KernelCallInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ParentOneOf<["MOGG::BindLambdaOp", "MOGG::CallForeachRegionOp", "MOGG::TensorCreateEpilogueOp", "MOGG::TensorCreatePrologueOp"]>,
  Pure,
]> {
  let summary = "Call the registered `::elementwise` function for a particular kernel";
  let description = [{
    Like `mogg.call.elementwise`, but also takes the current iteration index
    explicitly as an operand instead of relying on the kernel's registered
    signature to conditionally request one.

    Example:
    ```mlir
    mogg._call.elementwise["mo.add"](...)
    ```
  }];
  let arguments = (ins
    StrAttr:$kernelSymName,
    StrAttr:$moOpName,
    Variadic<AnyType>:$inputs,
    Optional<AnyType>:$index,
    OptionalAttr<DictionaryAttr>:$kernel_param
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `[` $kernelSymName `]` `(` $inputs (`,` $index^)? `)` attr-dict `:` type($inputs) (`,` type($index)^)? `->` type($outputs)
  }];
}

def MOGG_TensorCreateSliceOp : MOGG_Op<"_tensor.create.slice", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a strided slice view of another codegen tensor.";
  let description = [{
    Produces a `!mogg.tensor_` addressable directly by the consuming loop's
    own index, reading `$input`'s underlying storage through a start/step
    reindexing -- nothing is copied. `$outShape`/`$staticStart`/`$staticStep`
    carry the compile-time-known parts of the view's shape and bounds (`?`
    for a dimension that is only known at runtime); the input's own shape is
    read off `$input` directly rather than duplicated here.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    DenseI64ArrayAttr:$staticStart,
    DenseI64ArrayAttr:$staticStep,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Tensor:$starts,
    MOGG_Tensor:$stops,
    MOGG_Tensor:$steps
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $starts `,` $stops `,` $steps `)` attr-dict
    `:` `(` qualified(type($input)) `,` qualified(type($starts)) `,`
        qualified(type($stops)) `,` qualified(type($steps)) `)` `->`
    qualified(type($result))
  }];
}

def MOGG_TensorCreateBroadcastOp : MOGG_Op<"_tensor.create.broadcast", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a broadcast view of another codegen tensor.";
  let description = [{
    Produces a `!mogg.tensor_` addressable directly by the consuming loop's
    own index, reading `$input`'s underlying storage through a
    replicated-dimension reindexing (numpy broadcasting semantics) --
    nothing is copied. `$outShape` carries the view's shape; the input's own
    shape is read off `$input` directly rather than duplicated here.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `)` attr-dict
    `:` qualified(type($input)) `->`
    qualified(type($result))
  }];
}

def MOGG_TensorCreateTransposeOp : MOGG_Op<"_tensor.create.transpose", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a transpose view of another codegen tensor.";
  let description = [{
    Produces a `!mogg.tensor_` addressable directly by the consuming loop's
    own index, reading `$input`'s underlying storage through a
    dimension-reordering reindexing -- nothing is copied. `$outShape` and
    `$staticPermutations` carry the compile-time-known parts of the view's shape and
    permutation (`$staticPermutations[i]` names which `$input` dimension became output
    dimension `i`, `?` for a dimension whose source is only known at runtime);
    `$permutations` is the runtime permutation tensor, read only for the dimensions
    `$staticPermutations` leaves dynamic. The input's own shape is read off `$input`
    directly rather than duplicated here.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    DenseI64ArrayAttr:$staticPermutations,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Tensor:$permutations
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $permutations `)` attr-dict
    `:` `(` qualified(type($input)) `,` qualified(type($permutations)) `)` `->`
    qualified(type($result))
  }];
}

def MOGG_TensorCreateReshapeOp : MOGG_Op<"_tensor.create.reshape", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<MOGG_TensorCreateInterface>,
]> {
  let summary = "Build a codegen tensor that is a reshape view of another codegen tensor.";
  let description = [{
    Produces a `!mogg.tensor_` addressable directly by the consuming loop's
    own index, reindexing `$input`'s underlying storage via row-major
    de-linearization -- nothing is copied. `$outShape` carries the view's
    static shape (a `dyn` placeholder where a dim isn't known at compile
    time); `$shape` is the runtime `!mogg.shape` for it -- built the same way
    `mogg._tensor.create`'s own `$shape` is (`mogg._shape.create`), so an
    identical shape built elsewhere for the same kernel CSEs into one op --
    unlike broadcast/transpose, reshape computes its result's *stride*
    directly from the output shape's own values (row-major de-linearization
    has no other source of truth to fall back on, the way broadcast/transpose
    fall back to the input's own real stride), so a dim this op doesn't
    actually know at runtime would corrupt every element beyond the first row
    rather than just misreport shape metadata. The input's own shape is read
    off `$input` directly rather than duplicated here.
  }];
  let arguments = (ins
    MOSH_ShapeAttr:$outShape,
    TypeAttr:$elementType,
    BoolAttr:$mut,
    MOGG_TensorLike:$input,
    MOGG_Shape:$shape
  );
  let results = (outs MOGG_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MOGG_IndexReshapeOp : MOGG_Op<"index.reshape", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let summary = "Reindex an N-D index into an M-D index preserving row-major offset.";
  let description = [{
    Reindexes `$index` (a point in `$fromShape`) to the point in `$toShape`
    with the same row-major linear offset -- the per-index counterpart of
    `mogg._tensor.create.reshape`, used when a reshape is fused into an
    epilogue's STORE (transforming the store index) rather than materialized
    as a whole-tensor view. `MAPToMOGG` lowers an in-epilogue
    `map.index.reshape` into this (it is deliberately NOT hoisted, unlike the
    `.stride` view ops, since it depends on the loop index and feeds a store).

    `$fromShape`/`$toShape` carry the static shapes (`dyn` where a dim isn't
    known at compile time). `$dynamicShapes` supplies the runtime `!mogg.shape`
    for a shape that has a dynamic dim (row-major (de)linearization onto a
    dynamic shape has no other source of truth) -- empty when both shapes are
    fully static.

    Example:
    ```mlir
    %j = mogg.index.reshape(%i) {fromShape = #mosh.shape<[8, 8]>, toShape = #mosh.shape<[64]>} : (!map.index<2>) -> !map.index<1>
    ```
  }];
  let arguments = (ins
    AnyType:$index,
    MOSH_ShapeAttr:$fromShape,
    MOSH_ShapeAttr:$toShape,
    Variadic<MOGG_Shape>:$dynamicShapes
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
