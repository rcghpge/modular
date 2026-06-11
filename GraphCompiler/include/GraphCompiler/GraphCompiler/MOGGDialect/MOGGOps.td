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

include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGDialect.td"
include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGEnums.td"
include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGAttrs.td"
include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGTypes.td"
include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypes.td"
include "GraphCompiler/GraphCompiler/MODialect/MOTypes.td"
include "GraphCompiler/GraphCompiler/MODialect/MOAttrs.td"
include "GraphCompiler/GraphCompiler/MODialect/MOEnums.td"
include "GraphCompiler/GraphCompiler/MODialect/MOInterfaces.td"
include "GraphCompiler/GraphCompiler/MOSHDialect/MOSHAttrs.td"
include "GraphCompiler/GraphCompiler/Interfaces/LowerableToMGPInterface.td"
include "GraphCompiler/GraphCompiler/Interfaces/EmitsMojoInterface.td"
include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPOpInterfaces.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/LITDialect/LITTypes.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
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
    place of a context. The op returns an `!mgp.device_graph.node` handle naming
    the recorded node.

    Unlike `mogg.call.execute`, this is not a kernel terminator and is not
    wrapped in `mogg.with_trace`: it produces an SSA result (the node handle)
    that flows to the recording stub's `mogg.output`.

    Example:
    ```mlir
    %node = mogg.call.build_graph["linalg_kernels.__init__.Matmul"]
        inputs(%in, %wt : !mo.tensor<...>, !mo.tensor<...>)
        outputs(%out : !mo.tensor<...>)
        builder(%builder) -> !mgp.device_graph.node

    // Lowered to
    var node = Matmul.build_graph[...](out, in, wt, builder)
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

  let results = (outs MGP_DeviceGraphNodeType:$node);

  let assemblyFormat = [{
    `[` $kernelSymName `]` (`inputs` `(` $inputs^ `:` qualified(type($inputs)) `)`)?
                        (`outputs` `(` $outputs^ `:` qualified(type($outputs)) `)`)?
                        `builder` `(` $graphBuilder `)` `->` qualified(type($node)) attr-dict
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
  HasParent<"MOGG::BindLambdaOp">,
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

def MOGG_AsyncReadyOp : MOGG_Op<"async.ready", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
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
  DeclareOpInterfaceMethods<KernelArgumentConsumer>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let arguments = (ins MOGG_Async:$input);
  let results = (outs AnyType:$output);
  let summary = "Unpacks an async value and returns it as sync";
  let description = [{
    The values outside of a `mogg.generic.compiled.region` are asynchronous.
    This operation projects an `!mogg.async<T>` to its underlying `T`, modeling
    the fact that the async value is really a handle/pointer that must be
    unpacked before use. It is the inverse of `mogg.async.pack`.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizeMethod = 1;
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
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let arguments = (ins
    AnyType:$input,
    Optional<MGP_DeviceContextType>:$deviceCtx
  );
  let results = (outs MOGG_Async:$output);
  let summary = "Packs an sync value and returns it as async";
  let description = [{
    This is the opposite of the unpacking operation. It injects a synchronous
    `T` value into an `!mogg.async<T>`.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}


def MOGG_AsyncPackBorrowOp : MOGG_Op<"async.pack.borrow", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
  InferTypeOpAdaptor,
  Pure,
]> {
  let arguments = (ins
    AnyType:$input,
    AnyType:$mem
  );
  let results = (outs MOGG_Async:$output);
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
  HasParent<"MOGG::TryOp">,
  Terminator,
]> {
  let summary = "MOGG yield operation";
  let description = [{
    The "mogg.yield" operation specifies the output values for a mogg.try / except statement.
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
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
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
  DeclareOpInterfaceMethods<KernelTerminatorProvider>,
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

    This op gets lowered to a mojo call to `mogg_async_error`, which assigns
    the error to the corresponding AsyncValue.
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

def MOGG_AsyncPlaceholderAnyAsyncValueRefOp : MOGG_Op<"async.placeholder.AnyAsyncValueRef", [
  Pure,
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
]> {
  let results = (outs PointerOf<PointerOf<KGEN_NoneType>>:$result);
  let summary = "Returns an empty MLRT::AnyAsyncValueRef";
  let description = [{
    Required to serve as the borrowee for mogg.async.pack.borrow when the value
    is coming from a mgp.buffer.constant. The content of the placeholder is
    initialized to be nullptr.
  }];

  let assemblyFormat = [{attr-dict}];
}

def MOGG_AsyncUnpackAnyAsyncValueRefOp : MOGG_Op<"async.unpack.AnyAsyncValueRef", [
  DeclareOpInterfaceMethods<EmitsMojoInterface>,
  Pure,
]> {
  let summary = "Returns the memory storage handle of a tensor or buffer.";
  let description = [{
    For a given MGP device buffer or tensor (or its `!mogg.async<T>` wrapper),
    returns the AnyAsyncValueRef that represents the memory storage. This is
    required as the borrowee to mogg.async.pack.borrow. The result is always an
    AnyAsyncValueRef handle, independent of the input type.
  }];

  let arguments = (ins MOGG_AsyncOf<[MGP_DeviceBufferType, MGP_DeviceTensorType]>:$value);
  let results = (outs PointerOf<PointerOf<KGEN_NoneType>>:$result);

  let builders = [
    // The result is always an AnyAsyncValueRef handle (`!kgen.pointer<pointer<
    // none>>`), so callers need only supply the source value.
    OpBuilder<(ins "::mlir::Value":$value), [{
      build($_builder, $_state,
            ::M::KGEN::PointerType::get(::M::KGEN::PointerType::get(
                ::M::KGEN::NoneType::get($_builder.getContext()))),
            value);
    }]>,
  ];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizeMethod = 1;
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

def MOGG_KernelDeclOp : MOGG_Op<"kernel.decl", [Symbol]> {
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
    OptionalAttr<MOGG_KernelFunctionAttr>:$execute,
    OptionalAttr<MOGG_KernelFunctionAttr>:$elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$tile_elementwise,
    OptionalAttr<MOGG_KernelFunctionAttr>:$update_view,
    OptionalAttr<MOGG_KernelFunctionAttr>:$build_graph,
    ArrayAttr:$struct_params,
    BoolAttr:$is_user_defined
  );

  let assemblyFormat = [{
    $sym_name `registration` `=` $registration
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

def MOGG_PrimitiveDeclOp : MOGG_Op<"primitive.decl", [Symbol]> {
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

def MOGG_ShapeFunctionDeclOp : MOGG_Op<"shape_func.decl", [Symbol]> {
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
    StrAttr:$registration,
    MOGG_KernelFunctionAttr:$function
  );

  let assemblyFormat = [{
    $sym_name `registration` `=` $registration `function` `=` $function
    attr-dict
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_TD
