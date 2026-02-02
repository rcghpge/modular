//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "GenericML/GraphCompiler/MODialect/MOEnums.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
  : Op<MODialect, mnemonic, [MO_BufferOpConformance] # traits>;

// Base class for ops that work with tensors in the MO dialect.
class MO_TensorOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [MO_TensorOpInterface, MO_HasDeviceInterface] # traits>;

// Base class for ops that work with tensors but do not have a notion of device.
class MO_DeviceAgnosticTensorOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [MO_TensorOpInterface] # traits>;

// Base class for ops working with control flow
class MO_ControlOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<mnemonic, [MO_ControlOpInterface,
                           MO_PreservedDuringKernelLowering,
                           MO_ExecutesOnHost] # traits>;

// Base class for ops working with mutable buffers
class MO_MutableOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [DeclareOpInterfaceMethods<MO_MutableOpInterface>,
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                     MO_HasDeviceInterface] # traits>;

// Base class for ops working with mutable buffers that do not have a notion of device
class MO_DeviceAgnosticMutableOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [DeclareOpInterfaceMethods<MO_MutableOpInterface>,
                     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>] # traits>;

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

def MO_IndexToTensorOp : MO_Op<"index.to_tensor", [
  Pure,
  MO_PreservedDuringKernelLowering,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "`index` to scalar `!mo.tensor` casting operation";
  let description = [{
    Example:

    ```mlir
      %c: index
      %scalarT = mo.index.to_tensor(%c) -> !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins SI64:$input);
  let results = (outs MO_RankedScalarIndexTensor:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//
def MO_TransferOp : MO_DeviceAgnosticTensorOp<"transfer", [
  Pure,
  MO_PreservedDuringKernelLowering,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Device transfer operation";
  let description = [{
    This op represents a possible copy or aliasing operation to make the
    contents of the operand tensor available on the (virtual) device of the
    result tensor.

    It is valid for the source and destination devices to be identical. If the
    `alwaysElideSameDeviceCopy` flag is not set, it is implementation defined as
    to whether the result tensor is a copy or alias of the operand tensor; if
    this flag is true, transfers to the same device never result in a copy.

    Example:

    ```mlir
      %arg : !mo.tensor<[N, 8], f32, gpu:3>
      %onOtherDevice = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:3> to <"gpu", 1>
      %onHost = mo.transfer %arg : !mo.tensor<[N, 8], f32, gpu:0> to <"cpu", 1>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$alwaysElideSameDeviceCopy,
    MO_Chain:$inChain
  );
  let results = (outs MO_Tensor:$result, MO_Chain:$outChain);

  let assemblyFormat = [{
    `[` $inChain `]` $input `:` qualified(type($input)) `to`
    custom<TransferResult>(ref(type($input)), type($result)) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    OpBuilder<(ins "TensorValue":$input, "DeviceRefAttr":$destDevice,
      "ChainValue":$inChain, CArg<"bool", "true">:$alwaysElideSameDeviceCopy)>,
  ];
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_DeviceAgnosticTensorOp<"graph", [
  HasParent<"ModuleOp">,
  IsolatedFromAbove, SingleBlockImplicitTerminator<"OutputOp">,
  RegionKindInterface, HasOnlyGraphRegion,
  // needed to leverage KGEN parameter infra
  DeclInterface,
  MO_PreservedDuringKernelLowering,
  Symbol,
]> {
  let summary = "Computation graph operation";
  let description = [{
    This op represents a computation graph that consists of:
    - input data and their types
    - output types
    - other ops representing computations on input and intermediate data
    - a terminating output op that returns the outputs

    Example:

    ```mlir
      mo.graph @example<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        // ... intermediate computations ...
        %res : !mo.tensor<[D3], f32>
        mo.output<D3> %arg : !mo.tensor<[D3], f32>
      }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<KGEN_FuncTypeGeneratorType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParameters,
    KGEN_ParamDeclArrayAttr:$resultParameters,
    IntProp<"int64_t">:$counter,
    UnitAttr:$isSubgraph
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<GraphOp>($sym_name, $signature, $functionType, $inputParameters,
                    $resultParameters, attr-dict, $body, $counter)
  }];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
    OutputOp getOutputOp();

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the callable region of this function.
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// Return a new dimension parameter of `IndexType`, with a name that starts
    /// with `prefix` and differs from existing parameter names inside this
    /// graph op.
    KGEN::ParamDeclAttr getNewDimParam(const Twine &prefix);

    /// Return a new dimension parameter of `MOSH::ShapeType`, with a name that
    /// starts with `prefix` and differs from existing parameter names inside
    /// this graph op.
    KGEN::ParamDeclAttr getNewShapeParam(const Twine &prefix);

    /// Return a new dimension parameter of given type, with a name that starts
    /// with `prefix` and differs from existing parameter names inside this
    /// graph op.
    KGEN::ParamDeclAttr getNewParamOfType(const Twine &prefix, Type type);

    /// Return a new name that's unique from existing parameters inside this
    /// graph op.
    std::string getNewParamName(const Twine &prefix);

    /// Returns the #M.device_spec_collection annotation for this graph op, or
    /// null if no such annotation.
    DeviceSpecCollectionAttr getDeviceSpecs();

    /// Sets the #M.device_spec_collection annotation for this graph op,
    /// overwriting any existing annotation.
    void setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs);

    /// Sets the #M.device_spec_collection annotation for this graph op,
    /// overwriting any existing annotation.
    void setDeviceTargetInfo(TargetInfoAttr targetInfo);

    /// Sets the #M.device_spec_collection annotation for this graph op,
    /// overwriting any existing annotation.
    TargetInfoAttr getDeviceTargetInfo();

    /// Implement the DeclInterface methods with our own parameter
    /// implementation in MO.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  let builders = [
    /// Create a new mo.graph with an empty body.
    OpBuilder<(ins
      "StringRef":$name,
      "TypeRange":$inputTypes,
      "TypeRange":$resultTypes,
      CArg<"bool", "false">:$isSubgraph
    )>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def MO_CallOp : MO_ControlOp<"call", [
  CallOpInterface,
  DeclareOpInterfaceMethods<MO_ConditionallyInPlaceInterface>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, ["getInChains", "getOutChains", "getInChainsMutable"]>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  HasParent<"MO::GraphOp">,
  MO_ExecutesOnHost
]> {
  let summary = "Call a computation graph";
  let description = [{
    This op calls a computation graph.

    Example:

    ```mlir
      %res = mo.call @gelu(%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    FlatSymbolRefAttr:$callee,
    DefaultValuedAttr<StrAttr, "\"\"">:$prefix,
    MO_ParamDecls:$outputParamDecls,

    // Needed to satisfy CallOpInterface
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) $callee `(` operands `)` attr-dict `:`
       functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
      /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      this->setCalleeAttr(cast<FlatSymbolRefAttr>(cast<SymbolRefAttr>(callee)));
    }
  }];

  let hasVerifier = true;
}


//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MO_OutputOp : MO_ControlOp<"output", [
  Terminator, ReturnLike, HasParent<"MO::GraphOp">,
  DeclareOpInterfaceMethods<ParamOpInterface>
]>, NoParamDecls {
  let summary = "Graph output operation";
  let description = [{
    This op specifies the output parameters and values for a `mo.graph`. The
    op takes variable number of operands and produces no results. The operand
    number and types must match the signature of the `mo.graph` that contains
    the op (after substituting the bindings to signature output type).

    Examples:

    ```mlir
      mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
        mo.output %arg0 : !mo.tensor<?, f32>
      }

      mo.graph @with_params<D1 -> D2>(
          %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
        mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
      }
    ```

    Note that in the parameterized example, the output operands type annotation
    and the graph's output type don't necessarily need to match textually, but
    the compiler will eventually verify that they do.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
  );

  let assemblyFormat = [{
    `` custom<OutputValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : MO_ControlOp<"if", [
  DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getEntrySuccessorRegions", "getSuccessorInputs"]>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  HasOnlyGraphRegion,
  NoRegionArguments,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlockImplicitTerminator<"MO::YieldOp">,
  DeclInterface,
]> {
  let summary = "`if` control flow operation";
  let description = [{
    The `mo.if` op takes an `i1` condition, a 'then' block and an 'else'
    block. If the condition is true, the 'then' block is run and the op
    returns the results of that block, otherwise the "else" block is
    run and its values are returned.

    The blocks have access to all outer values. The blocks must return
    values using the `mo.yield' op, and the returned values must match the
    types given in the `mo.if` result signature.

    Example:

    ```mlir
      %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
        %v1 = mo.add(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                ) -> !mo.tensor<?, f32>
        mo.yield %v1 : !mo.tensor<?, f32>
      } else {
        %v2 = mo.sub(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                  ) -> !mo.tensor<?, f32>
        mo.yield %v2 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins
      AnyTypeOf<[MO_CondLike, MO_BoolScalar]>:$cond,
      MO_ParamDecls:$outputParamDecls
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    Block *getThenBlock();
    Block *getElseBlock();

    /// Implement the DeclInterface methods with our own parameter
    /// implementation in MO.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `(` $cond `:` qualified(type($cond)) `)` (`->` type($results)^)?
    $thenRegion `else` $elseRegion attr-dict
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : MO_ControlOp<"while", [
  DeclareOpInterfaceMethods<LoopLikeOpInterface,
    ["getRegionIterArgs", "getInitsMutable", "getYieldedValuesMutable"]>,
  DeclareOpInterfaceMethods<RegionBranchOpInterface,
    ["getEntrySuccessorOperands", "getSuccessorInputs"]>,
  DeclareOpInterfaceMethods<MO_MutableOpInterface, ["getInChains", "getOutChains", "getInChainsMutable"]>,
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlock,
  DeclInterface,
]> {
  let summary = "`while` control flow operation";
  let description = [{
    The `mo.while` operation takes "cond" and "body" blocks. While the "cond"
    block evaluates to the true condition, the "body" block is executed.

    The "cond" and "body" blocks have access to both the values listed in the
    `mo.while` signature and any other outer values.

    - The "cond" block must return a `!mo.tensor<[], bool>` result
    followed by values whose types match the inputs of the `mo.while` signature.

    - The "body" block must return results with the same types as the `mo.while`
    signature, again using a `mo.yield`. The signature must include an "as"
    clause for every input so as to rename it to a fresh symbol.

    The results of the `mo.while` op are the operands of the
    `mo.while.condition` op when the condition operand evaluates to false.

    Example:

    ```mlir
      %x: !mo.tensor<[2], f32>
      %y: !mo.tensor<[], f32>
      %res = mo.while (%x as %inner_x: !mo.tensor<[2], f32>) {
        %zero = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
        %shape = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
        %mean0 = mo.mean (%inner_x, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
        %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
        %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
        mo.while.condition(%cond) %inner_x : !mo.tensor<[2], f32>
      } do {
        %new_x = mo.add(%inner_x, %inner_x) : !mo.tensor<[2], f32>
        mo.yield %new_x : !mo.tensor<[2], f32>
      }
      %res: !mo.tensor<[2], f32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    WhileConditionOp getCondTerminator();
    YieldOp getBodyTerminator();

    Block *getCondBlock();
    Block *getBodyBlock();

    Block::BlockArgListType getCondArguments();
    Block::BlockArgListType getBodyArguments();

    /// Implement the DeclInterface methods with our own parameter
    /// implementation in MO.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  // The assemblyFormat is almost:
  //   custom<ParenOperandList>($inputs) attr-dict $condRegion `do` $bodyRegion
  // However since inputs are used as both referential and binding occurrences
  // we must instead use a custom parser.
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def MO_YieldOp : MO_ControlOp<"yield", [
  Terminator,
  ReturnLike,
  Pure,
  ParentOneOf<["IfOp", "WhileOp", "ParallelOp"]>,
  DeclareOpInterfaceMethods<ParamOpInterface>
]>, NoParamDecls {
  let hasVerifier = 1;
  let summary = "`yield` operation";
  let description = [{
    This op specifies the output values for control flow blocks. The op
    takes variable number of operands and produces no results.

    Example:

    ```mlir
      mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
        mo.yield %arg0 : !mo.tensor<?, f32>
      } else {
        mo.yield %arg1 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins
    Variadic<MO_AnyNonBufferType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
);

  let assemblyFormat = [{
    `` custom<OutputValues>($parameters) attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// WhileConditionOp
//===----------------------------------------------------------------------===//

def MO_WhileConditionOp : MO_ControlOp<"while.condition", [
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
    ["getSuccessorRegions"]>,
  HasParent<"WhileOp">,
  MO_PreservedDuringKernelLowering,
  Pure,
  Terminator,
]> {
  let summary = "loop continuation condition";
  let description = [{
    This op takes the continuation condition of the parent `mo.while`. The op
    takes variable number of operands which must match the operands of the
    parent `mo.while`.

    See the mo.while operation for an example.
  }];

  let arguments = (ins
    MO_CondLike:$condition,
    Variadic<MO_AnyNonBufferType>:$args
  );

  let assemblyFormat = [{
    `(` $condition `)` attr-dict ($args^ `:` type($args))?
  }];
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

def MO_ParallelOp : MO_ControlOp<"parallel", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface, ["getInChains", "getOutChains", "getInChainsMutable"]>,
  HasOnlyGraphRegion,
  RecursiveMemoryEffects,
  RegionKindInterface,
  SingleBlock,
  DeclInterface,
  SingleBlockImplicitTerminator<"MO::YieldOp">,
]> {
  let summary = "`parallel` region operation";
  let description = [{
    The `mo.parallel` operation takes a single "body" block, which is executed
    in parallel for each set of inputs.

    The results of the `mo.parallel` op are the operands of the
    `mo.yield` op from each iteration, which must all have the same type.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let extraClassDeclaration = [{
    /// Implement the DeclInterface methods with our own parameter
    /// implementation in MO.
    ArrayRef<ParamDeclAttr> getInputParams();
    SmallVector<ParamDeclAttr> getAllParams();
  }];

  // Custom assembly format: mo.parallel %arg in (%a, %b, ... : type) { ... }
  // Multiple inputs map to a single block argument that processes each in parallel.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def MO_RebindOp : MO_ControlOp<"rebind", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Parameter rebind operation";
  let description = [{
    This op represents the "rebinding" of a type to another type. This is
    typically used for rebinding the shape of !mo.tensor types to other
    !mo.tensor types to get things to type-check due to strict shape-related
    verifiers.

    Rebinding is similar to casting, except no data conversion takes place.
    It is assumed that the two types ultimately prove to be the same at runtime.

    Therefore, in cases where it is not statically provable, the left and right
    runtime types are the same, it is expected we will insert runtime assertions
    of some sort. Note, this operation does not do this, and any assertions
    must be inserted separately.

    Note, rebinds which use parameters they declare essentially "rename"
    the associated dimensions which is a useful tool. These types of rebinds
    are guaranteed to be removed by shape inference unless they are used to
    name unknown dims (denoted by `?`) from 3P dialects like those in PT.

    Examples:
    ```mlir
      // something like ASSERT N == 3 AND Sh == {3, 1} if we aren't statically
      // sure of this fact.
      %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
      %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[3, 1], f32>

      // Renames dims. K == 3, M == 1
      %3 = mo.rebind<() -> K, M>(%0) :
        !mo.tensor<[3, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantExternalOp
//===----------------------------------------------------------------------===//

def MO_ConstantExternalOp : MO_TensorOp<"constant.external", [
    MO_ConstantLike,
    Pure,
    MO_PreservedDuringKernelLowering,
    MO_ExplicitDevice,
]> {
  let summary = "External constant operation";
  let description = [{
    Represents an undefined reference to an "external" constant.
    This constant's backing data is undefined at graph compile time, but all
    other attributes of the tensor are statically known.
    In a given mo.graph, mo.constant.external ops should be uniquely named.

    Currently, the `device` attribute on the op determines which device the
    weights registry pointers reside on.
    This should be checked at runtime.
    The device attribute and result type can mismatch only when the `device`
    attribute is on the host.
    This is to support mmap'ed weights, a common use case for external
    constants.

    TODO(MSDK-1060): implement the runtime check by storing device alongside
    runtime pointer in the weights registry.
    And emit an explicit mo.transfer in the graph API instead of implicitly
    inserting an HtoD copy in converting mo.constant.external to
    mgp.buffer.constant.external.

    If the `hasAlias` attribute is set to true, the external constant might be
    updated through external alias between graph invocation, and it will not
    be lifted to the init phase.

    Example:

    ```mlir
    %weight = mo.constant.external {
      align = 16 : ui64, device = #M.device_ref<"gpu", 0>, name = "foo"
    } : !mo.tensor<[4096, 4096], bf16>
  }];

  let arguments = (ins
    StrAttr:$name,
    UI64Attr:$align,
    M_DeviceRefAttr:$device,
    DefaultValuedAttr<BoolAttr, "false">:$hasAlias,
    DefaultValuedAttr<BoolAttr, "false">:$isPlaceholder
  );
  let results = (outs MO_Tensor:$result);

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_TensorOp<"constant", [
    ConstantLike,
    MO_ConstantLike,
    Pure,
    MO_PreservedDuringKernelLowering,
    MO_SingleDevice,
]> {
  let summary = "Constant operation";
  let description = [{
    This op allows storing literal values inside the graph.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
    ```
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[M_ArrayElementsAttr,
                  Builtin_DenseResourceElementsAttr,
                  M_AlignedBytesAttr]>]
    >:$value);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasVerifier = 1;
  let hasFolder = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    /// Override the default builder to add the device attribute from the
    /// result type if provided.
    OpBuilder<(ins "Type":$resultType, "ElementsAttr":$value)>,

    /// Create a new rank-1 tensor mo.constant of int32 value.
    OpBuilder<(ins "ArrayRef<int32_t>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int32 value.
    OpBuilder<(ins "int32_t":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-1 tensor mo.constant of int64 value.
    OpBuilder<(ins "ArrayRef<int64_t>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int64 value.
    OpBuilder<(ins "int64_t":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-1 tensor mo.constant of any float type.
    OpBuilder<(ins "ArrayRef<APFloat>":$values, "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of given float type.
    OpBuilder<(ins "APFloat":$value, "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-1 tensor mo.constant of any int type.
    OpBuilder<(ins "ArrayRef<APInt>":$values, "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of given int type.
    OpBuilder<(ins "APInt":$value, "Type":$elementType,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of f32 type.
    OpBuilder<(ins "float":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of f64 type.
    OpBuilder<(ins "double":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-1 tensor mo.constant of boolean value.
    OpBuilder<(ins "ArrayRef<bool>":$values,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of boolean value.
    OpBuilder<(ins "bool":$value,
      CArg<"DeviceRefAttr", "{}">:$deviceRef)>,
  ];
}


def MO_ConstantScalarOp : MO_TensorOp<"constant.scalar", [
  ConstantLike,
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_HostOnly
]> {
  let summary = "Scalar constant operation";
  let description = [{
    Same as `mo.constant`, but specialized to scalar types.

    Example:

    ```mlir
      %0 = mo.constant.scalar { value = 3 : si64 } : !mo.scalar<si64>
    ```
  }];

  let arguments = (ins
      AnyAttrOf<[
                  Builtin_IntegerAttr,
                  BoolAttr,
                  Builtin_FloatAttr,
                  ]>:$value);
  let results = (outs MO_Scalar:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
  let hasVerifier = true;

  let builders = [
    OpBuilder<(ins "TypedAttr":$value)>
  ];
}



//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

// For ops where each output depends only on the corresponding indices in the inputs
class MO_ElementWiseOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits #
     [Pure, MO_SameOperandsAndResultShape, MO_ElementWiseLike, MO_SingleDevice,
      DeclareOpInterfaceMethods<MO_ParamDeclaration>]>
{
  let results = (outs MO_Tensor:$result);
}

// Like MO_ElementWiseOp, but the dtypes and shapes match for all results/operands
class MO_ElementWiseMonoTypeOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_SameOperandsAndResultType]> {
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

class MO_ElementWiseUnaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseUnaryMonoTypeOp<string name, list<Trait> traits = []>
    : MO_ElementWiseMonoTypeOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseUnaryFloatMonoTypeOp<string name, list<Trait> traits = []>
    : MO_ElementWiseUnaryMonoTypeOp<name, traits> {
  let arguments = (ins MO_FloatTensor:$input);
}

class MO_ElementWiseBinaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ElementWiseBinaryMonoTypeOp<string name, list<Trait> traits = []>
    : MO_ElementWiseMonoTypeOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ComparisonOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure, MO_SameOperandsElementType,
                            MO_SameOperandsAndResultShape, MO_SingleDevice,
                            DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class MO_ComparisonUnaryOp<string name, list<Trait> traits = []>
    : MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x);
}

class MO_ComparisonBinaryOp<string name, list<Trait> traits = []>
    : MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_LogicalBinaryOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [
  Commutative, Pure, MO_SameOperandsAndResultType, MO_ElementWiseBinary,
  MO_SingleDevice, DeclareOpInterfaceMethods<MO_ParamDeclaration>
]> {
  let arguments = (ins MO_BooleanMask:$input_x, MO_BooleanMask:$input_y);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_AddOp : MO_ElementWiseBinaryMonoTypeOp<"add", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Add operation";
  let description = [{
    Returns `x + y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_AbsOp : MO_ElementWiseUnaryMonoTypeOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_DivOp : MO_ElementWiseBinaryMonoTypeOp<"div", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Divide operation";
  let description = [{
    Returns `x / y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ExpOp : MO_ElementWiseUnaryMonoTypeOp<"exp"> {
  let summary = "Exponential function operation";
  let description = [{
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ErfOp : MO_ElementWiseUnaryMonoTypeOp<"erf"> {
  let summary = "(Gaussian) error function operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}


def MO_LogOp : MO_ElementWiseUnaryMonoTypeOp<"log"> {
  let summary = "Natural logarithm operation";
  let description = [{
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_Log1pOp : MO_ElementWiseUnaryMonoTypeOp<"log1p"> {
  let summary = "Natural logarithm of `1 + x` operation ";
  let description = [{
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MaxOp : MO_ElementWiseBinaryMonoTypeOp<"max", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Maximum operation";
  let description = [{
    Returns `max(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MinOp : MO_ElementWiseBinaryMonoTypeOp<"min", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Minimum operation";
  let description = [{
    Returns `min(x, y)`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ModOp : MO_ElementWiseBinaryMonoTypeOp<"mod"> {
  let summary = "Modulus operation";
  let description = [{
    Returns `x mod y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], si32>
      %rhs: !mo.tensor<[2, 3], si32>
      %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], si32>
    ```
  }];
}

def MO_MulOp : MO_ElementWiseBinaryMonoTypeOp<"mul", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Multiplication operation";
  let description = [{
    Returns `x * y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_NegativeOp : MO_ElementWiseUnaryMonoTypeOp<"negative", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Negation operation";
  let description = [{
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_SignedTensor:$input);
}

def MO_PowOp : MO_ElementWiseBinaryOp<"pow", [Pure]> {
  let summary = "Power operation";
  let description = [{
    Computes `x ** y`, where `x` and `y` are input tensors.

    Examples:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], f32>
      ) : !mo.tensor<[2, 3], f32>

      %x: !mo.tensor<[2, 3], f32>
      %y_int: !mo.tensor<[2, 3], si32>
      %res = mo.pow(
          %x: !mo.tensor<[2, 3], f32>,
          %y: !mo.tensor<[2, 3], si32>
      ) : !mo.tensor<[2, 3], f32>
      %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input_x `:` qualified(type($input_x))
    `,` $input_y `:` qualified(type($input_y)) `)` attr-dict
    `:` qualified(type($result))
  }];
}

// TODO(#22958): Consider relaxing constraints on trig. functions
def MO_SinOp : MO_ElementWiseUnaryFloatMonoTypeOp<"sin"> {
  let summary = "Sine operation";
  let description = [{
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}

def MO_CosOp : MO_ElementWiseUnaryFloatMonoTypeOp<"cos"> {
  let summary = "Cosine operation";
  let description = [{
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}

def MO_AtanhOp : MO_ElementWiseUnaryFloatMonoTypeOp<"atanh"> {
  let summary = "Atanh operation";
  let description = [{
    Returns `atanh(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = mo.atanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_FloatTensor:$input);
}


def MO_CumSumOp : MO_TensorOp<"cumsum", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_HostOnly,
]> {
  let summary = "Cumulative summation operation";
  let description = [{
    Returns the cumulative summation of input tensors along an axis. By default,
    it copies the first element as is. If the `exclusive` attribute is set to 1,
    then the first element is excluded. The `reverse` attribute causes the
    summation to be done in the opposite direction of the axis.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis: !mo.tensor<[], i64>
    %res = mo.cumsum(%arg, %axis) {exclusive = 1 : index, reverse = 0 : index} : (
      !mo.tensor<[2, 3], f32>., !mo.tensor<[], i64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_RankedScalarIndexTensor:$axis,
                       IndexAttr:$exclusive,
                       IndexAttr:$reverse);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input `,` $axis `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_CeilOp : MO_ElementWiseUnaryFloatMonoTypeOp<"ceil", [Idempotent]> {
  let summary = "Ceiling operation";
  let description = [{
    Returns the smallest largest integer greater than `x`, where `x` is input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_FloorOp : MO_ElementWiseUnaryFloatMonoTypeOp<"floor", [Idempotent]> {
  let summary = "Floor operation";
  let description = [{
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TruncOp : MO_ElementWiseUnaryFloatMonoTypeOp<"trunc", [Idempotent]> {
  let summary = "Truncation operation";
  let description = [{
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RoundOp : MO_ElementWiseUnaryFloatMonoTypeOp<"round", [Idempotent]> {
  let summary = "Rounding operation";
  let description = [{
    Returns the elementwise nearest integer, with ties going towards the
    nearest even number.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryMonoTypeOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RsqrtOp : MO_ElementWiseUnaryFloatMonoTypeOp<"rsqrt"> {
  let summary = "Reciprocal square root operation";
  let description = [{
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SqrtOp : MO_ElementWiseUnaryFloatMonoTypeOp<"sqrt"> {
  let summary = "Square root operation";
  let description = [{
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_CastOp : MO_ElementWiseUnaryOp<"cast", [
  Pure,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Casting operation";
  let description = [{
    Returns the input tensor, cast to the specified element type.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], i32>
      %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;

  let builders = [
    /// Cast DType interface
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

def MO_IsNaNOp : MO_ComparisonUnaryOp<"is_nan"> {
  let summary = "NaN test operation";
  let description = [{
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_IsInfOp : MO_ComparisonUnaryOp<"is_inf"> {
  let summary = "Infinity test operation";
  let description = [{
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_EqualOp : MO_ComparisonBinaryOp<"equal", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Equality test operation";
  let description = [{
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                    !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterOp : MO_ComparisonBinaryOp<"greater", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                      !mo.tensor<[2, 3], f32>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_GreaterEqualOp : MO_ComparisonBinaryOp<"greater_equal", [
  DeclareOpInterfaceMethods<MO_Staticization>
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_NotEqualOp : MO_ComparisonBinaryOp<"not_equal", [
  Commutative,
  DeclareOpInterfaceMethods<MO_Staticization>
]> {
  let summary = "Comparison operation";
  let description = [{
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_AndOp : MO_LogicalBinaryOp<"and"> {
  let summary = "Conjunction operation";
  let description = [{
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_OrOp : MO_LogicalBinaryOp<"or"> {
  let summary = "Disjunction operation";
  let description = [{
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_NotOp : MO_TensorOp<"not", [
  Pure, MO_SameOperandsAndResultType, MO_ElementWiseUnary,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice
]> {
  let summary = "Logical negation operation";
  let description = [{
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$input);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_XorOp : MO_LogicalBinaryOp<"xor"> {
  let summary = "Exclusive disjunction operation";
  let description = [{
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                  !mo.tensor<[2, 3], bool>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

// TODO: Use an AllElementTypesMatch<["x", "y", "result"]> trait for MO_Tensor.
def MO_SelectOp : MO_TensorOp<"select", [
  Pure,
  MO_SameOperandsAndResultShape,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice
]>{
  let summary = "Select operation";
  let description = [{
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$cond, MO_Tensor:$x, MO_Tensor:$y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MO_SliceOp : MO_TensorOp<"slice", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["start", "stop", "step"]>
]> {
  let summary = "Slice operation";
  let description = [{
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesVectorIfRanked:$start,
                       MO_IndicesVectorIfRanked:$stop,
                       MO_IndicesVectorIfRanked:$step,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_SliceDimOp : MO_TensorOp<"slice_dim", [
  MO_SingleDeviceWithHostOperands<["start", "stop", "step"]>,
  MO_ViewLike,
]> {
  let summary = "Slice dim operation";
  let description = [{
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The slice will be taken using `start` / `stop` / `step`
    along the `dim` dimension.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?, ?], f32>
      %start: !mo.tensor<[1], si64> // [1]
      %stop: !mo.tensor<[1], si32>  // [-3]
      %step: !mo.tensor<[1], si64>  // [5]
      // equivalent to this in numpy: `input[1:-3:5, :, :]`
      %res = mo.slice_dim[0](%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10, 10], f32>,
        !mo.tensor<[1], si64>,
        !mo.tensor<[1], si32>,
        !mo.tensor<[1], si64>
      ) -> !mo.tensor<[?, ?, ?], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input,
                       IndexAttr:$axis,
                       MO_RankedScalarIndexTensor:$start,
                       MO_RankedScalarIndexTensor:$stop,
                       MO_RankedScalarIndexTensor:$step);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `[` $axis `]``(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_SubOp : MO_ElementWiseBinaryMonoTypeOp<"sub", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Subtraction operation";
  let description = [{
    Returns `x - y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TanhOp : MO_ElementWiseUnaryFloatMonoTypeOp<"tanh"> {
  let summary = "Hyperbolic tangent operation";
  let description = [{
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Softmax-like ops
//===----------------------------------------------------------------------===//

class MO_SoftmaxLikeOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure,
                                  AllDTypesMatch<["input", "result"]>,
                                  AllShapeAttrsMatch<["input", "result"]>,
                                  MO_SingleDeviceWithHostOperands<["axis"]>,
                                  DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let arguments = (ins MO_Tensor:$input, MO_RankedScalarIndexTensor:$axis);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_LogSoftmaxOp : MO_SoftmaxLikeOp<"logsoftmax"> {
  let summary = "Logarithmic softmax operation";
  let description = [{
    Returns `log(softmax(x, axis))`, where `x` is input tensor, and `axis` is
    the axis along which `softmax` is applied.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.logsoftmax(%arg, %axis) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SoftmaxOp : MO_SoftmaxLikeOp<"softmax"> {
  let summary = "Softmax operation";
  let description = [{
    Returns `exp(input) / sum(exp(input))`, where `x` is input tensor.

    The `sum` reduction is applied along `axis`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %axis: !mo.tensor<[], si64>
      %res = mo.softmax(%arg, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

class MO_ReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
    name, traits # [MO_ReductionTraits,
      MO_SingleDeviceWithHostOperands<["axis"]>,
    ]>
{
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;

  let builders = [
    /// Build reduction op and automatically infer output type if it is not given.
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$axis, CArg<"TensorType", "{}">:$outputTy)>,
    OpBuilder<(ins "TensorValue":$input, "int64_t":$axis, CArg<"TensorType", "{}">:$outputTy)>
  ];
}

// Reduction operations where the input and result type match
class MO_MonotypeReductionOp<string name, list<Trait> traits = []> :
  MO_ReductionOp<name, traits # [AllDTypesMatch<["input", "result"]>]>;

def MO_MeanOp : MO_MonotypeReductionOp<"mean"> {
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
      %res = mo.mean(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMulOp : MO_MonotypeReductionOp<"reduce.mul", [
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
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
      %res = mo.reduce.mul(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceAddOp : MO_MonotypeReductionOp<"reduce.add"> {
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
      %res = mo.reduce.add(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
  let hasFolder = 1;
}

def MO_ReduceMinOp : MO_MonotypeReductionOp<"reduce.min"> {
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
      %res = mo.reduce.min(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMaxOp : MO_MonotypeReductionOp<"reduce.max"> {
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
      %res = mo.reduce.max(%arg, %axis) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Sliding window ops
//===----------------------------------------------------------------------===//

def MO_AvgPoolOp : MO_TensorOp<"avg_pool", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
]>{
  let summary = "Average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. `strides`, `dilations`, `padding`) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_AvgPoolCeilOp : MO_TensorOp<"avg_pool_ceil_mode_true", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
]>{
  let summary = "Ceiling-mode average pooling operation";
  let description = [{
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    BoolAttr:$count_boundary,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolOp : MO_TensorOp<"max_pool", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
]>{
  let summary = "Maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolCeilOp : MO_TensorOp<"max_pool_ceil_mode_true", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["filter_shape", "strides", "dilations", "paddings"]>,
]>{
  let summary = "Ceiling-mode maximum pooling operation";
  let description = [{
    Computes max pooling with the given filter shape, strides, and dilations.

    The op supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ConvOp : MO_TensorOp<"conv", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["strides", "dilations", "paddings", "num_groups"]>,
]>{
  let summary = "Convolution operation";
  let description = [{
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    `strides`, `dilations`, and `padding` must be of rank 1, or unranked.
    If the input has static rank, all hyperparameters with static shape must
    have sizes of `input_rank - 2`, except padding, which must have
    size `2 * (input_rank - 2)`. Individual elements in the hyperparameters
    apply to corresponding dimensions of the input (after ignoring the batch
    and channel dimensions), with padding representing a before/after pair for
    each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match if statically known.

    `num_groups` must be a ranked scalar. The number of input and output
    channels must both be divisible by the number of groups `num_groups`.

    This op currently only supports strides and padding on the input.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>}
        : %!mo.tensor<[], si64>
      %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_RankedScalarIndexTensor:$num_groups,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls
  );
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    ` ` `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
                    `paddings` `=` $paddings `,`
                    `num_groups` `=` $num_groups `]` `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_ConvTransposeOp : MO_TensorOp<"conv_transpose", [
  MO_ViewLike,
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["strides", "dilations", "paddings", "output_paddings"]>,
]>{
  let summary = "Transposed convolution operation";
  let description = [{
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %op = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %res = mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ShapeLikeIfRanked:$output_paddings,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$input_layout,
    DefaultValuedAttr<StrAttr, "">:$filter_layout,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)` attr-dict
    ` ` `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
        `paddings` `=` $paddings `,` `output_paddings` `=` $output_paddings`]` `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

class MO_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<mnemonic,
    traits # [MO_SameFirstOperandAndResultsElementType]
> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BroadcastToOp : MO_ShapeModifierOp<"broadcast_to", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_ViewLike,
  MO_SingleDevice
]> {
  let summary = "Broadcast operation";
  let description = [{
    Broadcasts the input tensor to the specified shape.

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %to = mo.constant {
        value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %res = mo.broadcast_to(%from, %to) : (
        !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let builders = [
    /// Create a ShapeOf + BroadcastToOp with a "BroadcastLike" interface.
    /// The second argument is not the shape to broadcast to, but a tensor
    /// whose shape we want to broadcast to.
    /// The third optional parameter should represent the shape of the
    /// second arg (e.g. from ShapeOfOp). If not given a ShapeOfOp will be
    /// inserted. It is provided to avoid Op duplication before CSE.
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$broadcastLike, CArg<"TensorValue", "{}">:$shapeOfInput)>,
  ];
}

def MO_AddSingletonDimOp : MO_TensorOp<"add_singleton_dim", [
  Pure,
  MO_ViewLike,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "Add a singleton dimension to a tensor shape";
  let description = [{
    Adds a dimension of `1` to a shape at the given axis.

    Example:
    ```mlir
      mo.add_singleton_dim[1](%res): (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
    ```
  }];
  let arguments = (ins MO_RankedTensor:$input, IndexAttr:$axis);
  let results = (outs MO_RankedTensor:$result);
  let assemblyFormat = [{
    `[` $axis `]``(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_SplitDimOp : MO_TensorOp<"split_dim", [
  Pure,
  MO_ViewLike,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "Splits a single dimensions of a tensor into two";
  let description = [{
    Splits tensor at `axis` into two dimensions. Example:
    Input=[N, K], Axis=0

    Output=[S1, S2, K], where S1 = N / S2.

    Value of S2 is taken from the output shape.

    Example:
    ```mlir
      %out = mo.split_dim[0](%res): (!mo.tensor<[4, 9], f32>) -> !mo.tensor<[2, 2, 9], f32>
    ```
  }];
  let arguments = (ins MO_RankedTensor:$input, IndexAttr:$axis);
  let results = (outs MO_RankedTensor:$result);
  let assemblyFormat = [{
    `[` $axis `]``(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_MergeDimOp : MO_TensorOp<"merge_dim", [
  Pure,
  MO_ViewLike,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "Merge two dimensions of a tensor into one";
  let description = [{
    Merges two adjacent dimensions of a tensor into one. Example:
    Input=[A, B, C, D], Axis=1

    Output=[A, B*C, D].

    We merge axis i and i+1 into one dimension.

    Example:
    ```mlir
      %out = mo.merge_dim[1](%res): (!mo.tensor<[1, 2, 3, 4], f32>) -> !mo.tensor<[1, 6, 4], f32>
    ```
  }];
  let arguments = (ins MO_RankedTensor:$input, IndexAttr:$axis);
  let results = (outs MO_RankedTensor:$result);
  let assemblyFormat = [{
    `[` $axis `]``(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ReshapeOp : MO_ShapeModifierOp<"reshape", [
  MO_ViewLike,
  MO_PreservedDuringKernelLowering,
  DeclareOpInterfaceMethods<MO_Staticization>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_ExecutesOnHost
]> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same underlying data, but different shape.

    The first argument is the tensor to reshape.  The second tensor is the
    shape to reshape the first tensor to.  The second tensor may contain a
    single "-1" element, which signifies that that dimension should be
    automatically computed.

    Example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```

    Auto-sizing example:

    ```mlir
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape = mo.constant {
        value = #M.dense_array<3, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_SqueezeShapeOp : MO_ShapeModifierOp<"squeeze_shape", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_HostOnly
]> {
  let summary = "Squeeze-shape operation";
  let description = [{
    Calculates the shape from squeeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N - M`. The
    indices represent the 0-based index of dimensions in the original rank `N`
    tensor.

    The indicated indices must represent dimensions of size 1. If
    an index does not point to a dimension to size 1, an error is thrown
    instead.

    This operator supports negative indexing with python-like semantics.
    That is all indices must be in [-N, N), if an index is < 0, it is as if
    we added `N` to it.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLike:$input_shape,
    MO_ShapeLike:$remove_indices,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let hasCanonicalizeMethod = 1;
}

def MO_UnsqueezeShapeOp : MO_ShapeModifierOp<"unsqueeze_shape", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_HostOnly
]> {
  let summary = "Unsqueeze-shape operation";
  let description = [{
    Calculates the shape from unsqueeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N + M`.

    The indices in the given list map to the new vector of length `N + M` where
    the indicated dimensions are replaced with `1`. The remaining dimension in
    the original input shape vector are copied over in the non-1 dimensions.

    This operator supports negative indexing.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLike:$input_shape,
    MO_ShapeLike:$padding_indices,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let hasCanonicalizeMethod = 1;
}

def MO_ShapeOfOp : MO_TensorOp<"shape_of", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExecutesOnHost
]> {
  let summary = "Shape operation";
  let description = [{
    Returns the shape of a tensor.

    Examples:

    ```mlir
      // statically ranked
      %arg1: !mo.tensor<[1, 2, 3], f32>
      %shape1 = mo.shape_of(%arg1) : !mo.tensor<[3], si64>

      // dynamically ranked
      %arg2: !mo.tensor<?, f32>
      %shape2 = mo.shape_of(%arg2) : !mo.tensor<[?], si64>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let builders = [
    /// Create a shape-of of op on the given SSA value.
    OpBuilder<(ins "TensorValue":$input)>,
    OpBuilder<(ins "TensorValue":$input, "int64_t":$width)>,
  ];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_BroadcastShapeOp : MO_TensorOp<"broadcast_shape", [
  Commutative,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDevice
]> {
  let summary = "Broadcast-shape operation";
  let description = [{
    Returns the shape that two shapes would broadcast to under the numpy rules:

    1. Make the shapes the same rank by adding 1 to the front
       of the shorter shape
    2. Ensure each dimension is equal, either by matching them or promoting a 1
       to a larger number from the other shape

    Examples:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [4, 2, 3]

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, `shape1` will compute to [10, 2, 5]
  }];

  let arguments = (ins MO_ShapeLike:$input_x,
                       MO_ShapeLike:$input_y,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

/// Used to model MO ops that accept both MO_Tensor and MOSH_ShapeType.
class MOShape_Op<string mnemonic, list<Trait> traits = []>
    : Op<MODialect, "shape." # mnemonic, traits> {
  string cppNamespace = MODialect.cppNamespace # "::Shape";
}

def MOShape_ToTensorOp : MOShape_Op<"to_tensor", [
  Pure,
  MO_PreservedDuringKernelLowering,
]> {
  let summary = "Casting operation";
  let description = [{
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasCanonicalizeMethod = 1;
}

def MOShape_FromTensorOp : MOShape_Op<"from_tensor", [
  Pure,
  MO_PreservedDuringKernelLowering,
]> {
  let summary = "Casting operation";
  let description = [{
    Casts the input shape value to a shape-like tensor.

    Example:

    ```mlir
      %sh: !mosh.ape
      %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
  }];

  let arguments = (ins MO_ShapeLike:$input);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

class MOStatic_TensorOp<string mnemonic, list<Trait> traits = []>
    : MO_TensorOp<"static." # mnemonic, traits> {
  string cppNamespace = MODialect.cppNamespace # "::Static";
}

class MOStatic_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
    : MOStatic_TensorOp<mnemonic, traits # [Pure, MO_SameFirstOperandAndResultsElementType, MO_SingleDevice]> {
  string cppNamespace = MODialect.cppNamespace # "::Static";

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];
}

def MOStatic_BroadcastToOp : MOStatic_ShapeModifierOp<"broadcast_to", [
    DeclareOpInterfaceMethods<MO_Staticization>, MO_ViewLike
  ]> {
  let summary = "Broadcasting operation";
  let description = [{
    Broadcasts the input tensor to the result tensor. The shape of the input and
    result tensors must not be unknown or contain unknown dimensions, but can be
    parametric.

    This op only has limited compile-time check on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.broadcast_to` with a constant shape-like tensor instead).

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
      %from: !mo.tensor<[3], f32>
      %res1 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
      kgen.param.declare N = <...>
      %res2 = mo.static.broadcast_to(%from)
        : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MOStatic_ReshapeOp : MOStatic_ShapeModifierOp<"reshape", [
  MO_ViewLike,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Reshape operation";
  let description = [{
    Returns a tensor with the same underlying data, but different shape. The
    shape of the input and result tensors must not be unknown or contain unknown
    dimensions, but can be parametric. We do not allow inferred dimensions
    (e.g. -1 in MO_ReshapeOp).

    The op has no compile-time or runtime checks on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.reshape` with a constant shape-like tensor instead).

    Example:

    ```mlir
      %from: !mo.tensor<[2, 3], f32>
      %res = mo.static.reshape(%from)
        : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_ConcatOp : MO_TensorOp<"concat", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_Staticization>,
  MO_SingleDeviceWithHostOperands<["axis"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>
]> {
  let summary = "Concatenation operation";
  let description = [{
    Concatenates the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor.

    The following constraints apply to the inputs/outputs:

    - The input tensors and output tensors all has the same shape except along
      the concatenation dimension `axis`.
    - The size of the concatenation dimension in output tensor have be the sum
      of sizes of the concatenation dimension in input tensors.
    - The element type of the input and output tensors must match.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarIndexTensor:$axis,
    Variadic<MO_Tensor>:$inputs,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_SplitOp : MO_TensorOp<"split", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["splitSizes", "axis"]>,
]> {
  let summary = "Splitting operation";
  let description = [{
    Splits the input tensor into multiple tensors along a given dimension.

    `mo.split` splits the tensor `input` into multiple output tensors.
    The number of output tensors is equal to the number of elements in
    `splitSizes`, which is a rank-1 tensor of integers.
    Each of the output tensors has the same shape as `input` except along the
    split dimension `axis`, where the size is given by the corresponding
    element in `splitSizes`.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 8], f32>
      %splitSizes = mo.constant {
        value = #M.dense_array<3, 5> : tensor<2xsi64>
      } : !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>
      } : !mo.tensor<[], si64>
      %res:2 = mo.split[%axis: !mo.tensor<[], si64>](%input, %splitSizes) : (
        !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
      ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$splitSizes,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $input `,` $splitSizes `)` attr-dict `:` `(`
      qualified(type($input)) `,` qualified(type($splitSizes))
    `)` `->` `(` type($results) `)`
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_TransposeOp : MO_TensorOp<"transpose", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["perm"]>
]> {
  let summary = "Transposition operation";
  let description = [{
    Returns a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$perm,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MO_TileOp : MO_TensorOp<"tile", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_HostOnly
]> {
  let summary = "Tiling operation";
  let description = [{
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$repeats,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MO_PadConstantOp : MO_TensorOp<"pad.constant", [
  MO_PadLikeTraits,
  MO_SingleDeviceWithHostOperands<["paddings", "constant"]>
]> {
  let summary = "Constant-mode padding operation";
  let description = [{
    Pads the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assumes input has rank `N`, the `paddings` tensor should have
    shape `(2 * N)`, where each consecutive pair of elements has the form
    `[before, after]`, indicating how many of `constant` to add before and after
    the contents of `input` in that dimension. The size of each dimension D of
    the padded output is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant {
        value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_RankedScalarTensor:$constant,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_PadReflectOp : MO_TensorOp<"pad.reflect", [
  MO_PadLikeTraits,
  MO_HostOnly
]> {
  let summary = "Reflection-mode padding operation";
  let description = [{
    Pads the `input` tensor by reflecting it according to the `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_PadRepeatOp : MO_TensorOp<"pad.repeat", [
  MO_PadLikeTraits,
  MO_HostOnly
]> {
  let summary = "Repeat-mode padding operation";
  let description = [{
    Pads the `input` tensor by repeating border values according to `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def MO_GatherOp : MO_TensorOp<"gather", [
  DeclareOpInterfaceMethods<MO_Staticization>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDeviceWithHostOperands<["axis"]>
]> {
  let summary = "`gather` operation";
  let description = [{
    Gathers slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %axis = mo.constant {
        value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.gather(%input, %indices, %axis) : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

// TODO(#20442): Remove with generic gather fusion.
def MO_GatherSumOp : MO_TensorOp<"gather_sum", [MO_HostOnly]> {
  let summary = "`gather-sum` operation";
  let description = [{
    A temporary composite op that composes a gather with sum reduction.
    The gather axis is 0 and reduction axis is 1.
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices);
  let results = (outs MO_Tensor:$result);
}

def MO_GatherNdOp : MO_TensorOp<"gather_nd", [
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDevice
]> {
  let summary = "Multi-dimensional `gather` operation";
  let description = [{
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension stores the index whereas
    the outer dimensions act like batch dimensions. The size of the last
    dimension is at most the rank of the input. When the dimension size is less
    than the rank of the input, slices of the input are gathered, starting from
    the leftmost dimension.

    ```
    output_shape = (
          input.shape[:batch_dims]
        + indices.shape[batch_dims:-1]
        + data.shape[batch_dims + indices.shape[-1]:]
    )
    ```

    ```mlir
      %input = mo.constant {device = #M.device_ref<"cpu", 0>, value =
        #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> :
        tensor<2x2x4xsi64>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} :
        !mo.tensor<[3], si64>

      %result = mo.gather_nd(%input, %indices) {batchDims = 0} :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices,
                       // TODO(#21324): Support dynamic batchDims
                       IndexAttr:$batchDims,
                       MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def MO_ScatterOp : MO_TensorOp<"scatter", [MO_ScatterLikeTraits, MO_HostOnly]> {
  let summary = "`scatter` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

class MO_ScatterReductionOp<string name, list<Trait> traits = []>
  : MO_TensorOp<"scatter." # name, traits # [MO_ScatterLikeTraits, MO_HostOnly]>
{
  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterAddOp : MO_ScatterReductionOp<"add"> {
  let summary = "`scatter-add` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMaxOp : MO_ScatterReductionOp<"max"> {
  let summary = "scatter-max operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMinOp : MO_ScatterReductionOp<"min"> {
  let summary = "`scatter-min` operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMulOp : MO_ScatterReductionOp<"mul"> {
  let summary = "scatter-mul operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

def MO_ScatterNdOp : MO_TensorOp<"scatter_nd", [MO_ScatterNdLikeTraits, MO_SingleDevice]> {
  let summary = "Multi-dimensional `scatter` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

class MO_ScatterNdReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
  "scatter_nd." # name, traits # [MO_ScatterNdLikeTraits, MO_SingleDevice]
> {
  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterNdAddOp : MO_ScatterNdReductionOp<"add"> {
  let summary = "Multi-dimensional `scatter-add` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMaxOp : MO_ScatterNdReductionOp<"max"> {
  let summary = "Multi-dimensional `scatter-max` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMinOp : MO_ScatterNdReductionOp<"min"> {
  let summary = "Multi-dimensional `scatter-min` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMulOp : MO_ScatterNdReductionOp<"mul"> {
  let summary = "Multi-dimensional `scatter-mul` operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NonMaximumSuppressionOp
//===----------------------------------------------------------------------===//

def MO_NonMaximumSuppressionOp : MO_TensorOp<"non_maximum_suppression",
    [DeclareOpInterfaceMethods<MO_ParamDeclaration>,
     MO_HostOnly]> {
  let summary = "Non-max suppression operation";
  let description = [{
   Filters out boxes that have high intersection-over-union (IOU).

   `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
   coordinates of any diagonal pair of box corners and the coordinates can be
   provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

    Example:

    ```mlir
      %boxes : !mo.tensor<[1, 6, 4], f32>
      %scores : !mo.tensor<[1, 1, 6], f32>
      %maxOutputBoxesPerClass : !mo.tensor<[], si64>
      %iouThreshold : !mo.tensor<[], si64>
      %scoreThreshold : !mo.tensor<[], si64>
      %res = mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold) : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], si64>
    ```
  }];
  // TODO(#21506): Allow any ranked tensors as inputs.
  let arguments = (ins MO_Rank3Tensor:$boxes,
                       MO_Rank3Tensor:$scores,
                       MO_RankedScalarIndexTensor:$maxOutputBoxesPerClass,
                       MO_RankedScalarFloatTensor:$iouThreshold,
                       MO_RankedScalarFloatTensor:$scoreThreshold,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Rank2IndexTensor: $output);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

def MO_ArgNonzeroOp : MO_TensorOp<"arg_nonzero", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_HostOnly
]> {
  let summary = "Nonzero-value indices operation";
  let description = [{
    Returns a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
      } : !mo.tensor<[3, 3], si32>
      %1 = mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_I64Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ArgMaxOp / ArgMinOp
//===----------------------------------------------------------------------===//

// Base class for ArgMin and ArgMax.
class MO_ArgMinMaxOp<string mnemonic, list<Trait> traits = []>
    : MO_ReductionOp<mnemonic, traits> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_IndicesTensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MO_ArgMaxOp : MO_ArgMinMaxOp<"arg_max"> {
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
      %1 = mo.arg_max(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_ArgMinOp : MO_ArgMinMaxOp<"arg_min"> {
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
      %1 = mo.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

//===----------------------------------------------------------------------===//
// RandomNormalOp
//===----------------------------------------------------------------------===//

def MO_RandomNormalOp : MO_TensorOp<"random.normal", [
  MO_SingleDeviceWithHostOperands<["shape", "mean", "variance", "seed"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  AllDTypesMatch<["mean", "variance"]>,
]> {
  let summary = "Normal distribution operation";
  let description = [{
    Returns a tensor with shape `shape` populated with random
    values from a normal distribution, with the mean of the distribution equal
    to `mean` and the standard deviation equal to `variance`.

  Example:
    ```mlir
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %mean = mo.constant {
        value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
      %variance = mo.constant {
        value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
      %seed = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64> } : !mo.tensor<[], si64>
      %res = mo.random.normal(%size, %mean, %variance, %seed) :
            (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
            !mo.tensor<[], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$shape,
    MO_RankedScalarFloatTensor:$mean,
    MO_RankedScalarFloatTensor:$variance,
    MO_RankedScalarIndexTensor:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RandomUniformOp
//===----------------------------------------------------------------------===//

def MO_RandomUniformOp : MO_TensorOp<"random.uniform", [
  MO_SingleDeviceWithHostOperands<["shape", "lowerBound", "upperBound", "seed"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  AllDTypesMatch<["lowerBound", "upperBound", "result"]>,
]> {
  let summary = "Uniform distribution operation";
  let description = [{
    Returns a tensor with shape `shape` populated with random values from a
    uniform distribution on the half-open interval [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant {
      value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %lowerBound = mo.constant {
      value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
    %upperBound = mo.constant {
      value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
    %seed = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64> } : !mo.tensor<[], si64>
    %res = mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[], si64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_IndicesTensor:$shape,
    MO_RankedScalarTensor:$lowerBound,
    MO_RankedScalarTensor:$upperBound,
    MO_RankedScalarIndexTensor:$seed,
    MO_ParamDecls:$outputParamDecls
  );

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

def MO_RangeOp : MO_TensorOp<"range", [
  MO_SameOperandsAndResultElementType,
  MO_SingleDeviceWithHostOperands<["start", "limit", "step"]>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  DeclareOpInterfaceMethods<MO_Staticization>,
]> {
  let summary = "Range operation";
  let description = [{
    Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant {
        value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
      %step = mo.constant {
        value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
      %res = mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant {
        value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
      %stepInt = mo.constant {
        value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
      %limitInt = mo.constant {
        value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
      %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarTensor:$start,
    MO_RankedScalarTensor:$limit,
    MO_RankedScalarTensor:$step,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Resize op
//===----------------------------------------------------------------------===//


def MO_ResizeLinearOp : MO_TensorOp<"resize.linear",
  [DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
   MO_HostOnly
]> {
  let summary = "Linear resize operation";
  let description = [{
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    When set to true, the antialias attribute causes an antialiasing filter to be applied
    when downscaling.
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    BoolAttr:$antialias,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}


def MO_ResizeNearestOp : MO_TensorOp<"resize.nearest",
  [DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
   MO_HostOnly
]> {
  let summary = "Nearest-neighbor resize operation";
  let description = [{
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_CoordinateTransformModeAttr:$coordinate_transform_mode,
    DefaultValuedAttr<I64Attr, "0">:$round_mode,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}


def MO_ResizeBicubicOp : MO_TensorOp<"resize.bicubic",
  [DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
   MO_SingleDeviceWithHostOperands<["size"]>
]> {
  let summary = "Bicubic resize operation";
  let description = [{
    Resizes a tensor to a new shape using the bicubic interpolation algorithm.

    Bicubic interpolation uses a 4x4 pixel neighborhood and cubic polynomials
    to produce smoother results than linear interpolation. This implementation
    uses Keys' cubic convolution with a = -0.5.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 3, 448, 448> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = mo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
  }];
  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesTensor:$size,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// ROI align op
//===----------------------------------------------------------------------===//

def MO_RoiAlignOp : MO_TensorOp<"roi_align", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_HostOnly
]> {
    let summary = "ROI alignment operation";
    let description = [{
      ROI align consumes an input tensor and regions of interest in which to apply pooling.

      Example:
      ```mlir
        %inp: !mo.tensor<[1, 10, 10, 1], f32>
        %rois: !mo.tensor<[1, 5], f32>
        %output_height = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
        %spatial_scale = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
        %sampling_ratio = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

        %res = mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
          {aligned = false,  mode = "AVG"}
          : (!mo.tensor<[1, 10, 10, 1], f32>,
            !mo.tensor<[1, 5], f32>,
            !mo.tensor<[], si64>,
            !mo.tensor<[], si64>,
            !mo.tensor<[], f32>,
            !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
      ```

    }];
    let arguments = (ins
      MO_Rank4Tensor:$input,
      MO_Rank2Tensor:$rois,
      MO_RankedScalarIndexTensor:$output_height,
      MO_RankedScalarIndexTensor:$output_width,
      MO_RankedScalarFloatTensor:$spatial_scale,
      MO_RankedScalarFloatTensor:$sampling_ratio,
      BoolAttr:$aligned,
      Builtin_StringAttr:$mode,
      MO_ParamDecls:$outputParamDecls);

    let results = (outs MO_Rank4Tensor:$result);

    let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:` functional-type(operands, results)
    }];
    // TODO: add verifier that checks that 2nd rois dimension is 4
  }

//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//

class MO_DotBase<string name, list<Trait> traits = []> :
    MO_TensorOp<name, traits # [
    Pure,
    MO_MatmulLike,
      DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    MO_SingleDevice
]> {
  let description = [{
    Base class for all generic dot products (i.e. tensor contractions with two
    inputs), for example matmuls.
  }];
}

def MO_MatmulOp : MO_DotBase<"matmul"> {
  let summary = "Matrix-matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two 2D tensors.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[10, 20], f32>
      %rhs: ... !mo.tensor<[20, 5], f32>
      %res = mo.matmul(%lhs, %rhs) : (
        !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
      ) -> !mo.tensor<[10, 5], f32>
    ```
  }];

  let arguments = (
    ins MO_Rank2Tensor:$input_a,
    MO_Rank2Tensor:$input_b,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<BoolAttr, "false">:$packed_b
  );
  let results = (outs MO_Rank2Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_BatchMatmulOp : MO_TensorOp<"batch_matmul", [
  Pure,
  MO_MatmulLike,
  DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
  MO_SingleDevice,
]> {
  let summary = "Batched matrix-matrix multiplication operation";
  let description = [{
    Performs matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.

    Example:

    ```mlir
      %lhs: ... !mo.tensor<[3, 4, 5], f32>
      %rhs: ... !mo.tensor<[3, 5, 6], f32>
      %res = mo.batch_matmul(%lhs, %rhs) :
        mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
      ) -> !mo.tensor<[3, 4, 6], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input_a,
    MO_Tensor:$input_b,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let builders = [
    /// Create matmul with default attribute flags off.
    OpBuilder<(ins "Type":$type, "Value":$input_a, "Value":$input_b, "ArrayRef<KGEN::ParamDeclAttr>":$decls)>,
  ];

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Linear Algebra ops
//===----------------------------------------------------------------------===//

class MO_LinalgOp<string mnemonic, list<Trait> traits = []>
    : MO_TensorOp<"linalg." # mnemonic, traits # [Pure]> {
  string cppNamespace = MODialect.cppNamespace # "::LinAlg";
}

// TODO: Use an AllElementTypesMatch<["input", "result"]> trait for MO_Tensor.
def MO_BandPartOp : MO_LinalgOp<"band_part", [
      DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    MO_SingleDeviceWithHostOperands<["num_lower", "num_upper", "exclude"]>
    ]> {
  let summary = "Matrix band part operation";
  let description = [{
    Copies a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matrices.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

    Please explicitly note that with negative values, this kernel returns the
    entire lower or upper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant {
        value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %num_upper = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %exclude = mo.constant {
        value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
      %res = mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$num_lower,
    MO_RankedScalarIndexTensor:$num_upper,
    MO_CondLike:$exclude,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape function invocation
//===----------------------------------------------------------------------===//

def MO_InvokeShapeFuncOp : MO_TensorOp<"invoke_shape_func", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_ExecutesOnHost
]> {
  let summary = "Shape function call operation";
  let description = [{
    Invokes a shape function with given name and bind its return values to the
    output parameters. Each argument must be a shape-like or integer scalar
    tensor. The `kgenParams` will be passed along to the shape function as is.

    Example:

    ```mlir
      %input : !mo.tensor<[D1, D2], f32>
      %axis  : !mo.tensor<[], si64>
      %input_shape = mo.shape_of(%input)
      : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si64>

      mo.invoke_shape_func["shape_func_name"]<() -> D1, D2>(%input_shape, %axis)
      : (!mo.tensor<[2], si64>, !mo.tensor<[], si64>)
    ```

    `dataDeptTensors` Is an instruction to the lowering that the inputs at
    these indices should be treated as data dependent tensors. These tensors
    include things like dynamic "axis" in reductions or "steps" in slice. Any
    tensor which needs to have its values read to compute the shape for the op
    falls into this category.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_List, MO_Opaque]>>:$inputs,
    StrAttr:$shapeFuncName,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<DictionaryAttr>:$kgenParams,
    OptionalAttr<ArrayAttr>:$dataDeptTensors
  );

  let assemblyFormat = [{
    `[` $shapeFuncName `]` `` custom<ParamDecls>($outputParamDecls)
    `(` $inputs `)` attr-dict `:` `(` type($inputs) `)`
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Layout Transform
//===----------------------------------------------------------------------===//

def MO_LayoutTransformOp : MO_TensorOp<"layout.transform", [MO_HostOnly]> {
  let summary = "Layout transformation operation";
  let description = [{
    This op transforms the layout of input tensor to the layout specified in
    the output type. The `kgenParams` will be passed along as named parameters
    to the underlying mojo kernel.

    It requires both the input and output tensor types to contain layout
    annotations.
  }];

  let arguments = (ins MO_Tensor:$input, OptionalAttr<DictionaryAttr>:$kgenParams);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Temp layer norm ops
//===----------------------------------------------------------------------===//

def MO_LayerNormOp : MO_TensorOp<"layer_norm", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_SingleDeviceWithHostOperands<["epsilon"]>
]> {
  let summary = "Layer normalization operation";
  let description = [{
    Layer normalization operation which operates on the last dimension of
    `input`:

      meanInput = mean(input)
      varInput = var(input)
      result = (input - meanInput) / sqrt(varInput + epsilon) * gamma + beta.

    We expect gamma and beta to be shape [channels], where channels is the size
    of the last input dimension.
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_Tensor:$gamma,
                       MO_Tensor:$beta,
                       MO_Tensor:$epsilon,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// DebugPrintOp
//===----------------------------------------------------------------------===//
def MO_DebugPrintOp : MO_TensorOp<"debug.print", [
  MemoryEffects<[MemWrite]>,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "Printing operation";
  let description = [{
    Prints a debug string. If a label attribute is supplied the string is printed with that label.
    Otherwise just the string is printed. For debugging and testing only.

    Example:
    ```mlir
      %ch0: !mo.chain
      %ch1 = mo.debug.print(%ch0) {value = "message", label = "label"}
    ```
  }];

  let arguments = (ins MO_Chain: $inChain, StrAttr: $value,
                   DefaultValuedAttr<StrAttr, "\"\"">: $label);
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict
  }];
}


//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

def MO_DebugTensorPrintOp : MO_TensorOp<"debug.tensor.print", [
  MemoryEffects<[MemWrite]>,
  MO_SingleDeviceWithHostOperands<[], ["input"]>,
  MO_PreservedDuringKernelLowering
]> {
  let summary = "Printing operation";
  let description = [{
    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      %ch0: !mo.chain
      %ch1 = mo.debug.tensor.print(%ch0, %arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
  }];

  let arguments = (ins MO_Chain: $inChain, MO_Tensor: $input,
                   DefaultValuedAttr<StrAttr, "\"\"">: $label);
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($input))
  }];
}

def MO_DebugTensorUnsafePrintOp : MO_TensorOp<"debug.tensor.unsafe.print", [
  MemoryEffects<[MemWrite]>,
  MO_SingleDeviceWithHostOperands<[], ["input"]>,
  MO_PreservedDuringKernelLowering
]> {
  let summary = "Printing operation";
  let description = [{
    Deprecated use mo.debug.tensor.print instead to ensure proper sequencing
    of print operations.

    This is just kept around due to the Mojo Graph API not supporting chains at
    the time of writing.

    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
      %arg: !mo.tensor<[5], f32>
      mo.debug.tensor.print(%arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
  }];

  let arguments = (ins MO_Tensor: $input,
                   DefaultValuedAttr<StrAttr, "\"\"">: $label);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` qualified(type($input))
  }];
}

// Class for top-k/bottom-k operators
class MO_SelectKLikeOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<
      mnemonic, [
              DeclareOpInterfaceMethods<MO_ParamDeclaration>,
        DeclareOpInterfaceMethods<MO_ShapeMaterialization>,
        MO_SingleDeviceWithHostOperands<["k", "axis", "sorted"]>
      ] # traits> {
  let arguments = (ins MO_Tensor:$input,
                       MO_RankedScalarIndexTensor:$K,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_CondLike:$sorted,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$values,
                      MO_I64Tensor:$indices);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}


def MO_TopKOp : MO_SelectKLikeOp<"top_k"> {
  let summary = "Top-K operation";
  let description = [{
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis. Returned values along the axis are always sorted
    (stable).

    axis: The axis to compute the largest values over.
      The axis must be in [-rank, rank).
    k: The number of values to compute.
    sorted: Whether to return the values and indices sorted or not.

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant() { value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.top_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

def MO_BottomKOp : MO_SelectKLikeOp<"bottom_k"> {
  let summary = "Bottom-K operation";
  let description = [{
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
  }];
}

//===----------------------------------------------------------------------===//
// List ops
//===----------------------------------------------------------------------===//

def MO_ListAppendOp : MO_TensorOp<"list.append", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_HostOnly
]> {
  let summary = "List appending operation";
  let description = [{
    Creates a new list that's the result of appending an element to the end of a
    copy of the input list (i.e., the op is value semantics and doesn't affect
    the input list).

    Example:

    ```mlir
      %list:   !mo.list<!mo.tensor<[2, 2], f32>>
      %tensor: !mo.tensor<[2, 3], f32>

      %tensor = mo.list.append(%list, %tensor) : (
        !mo.list<!mo.tensor<[2, 2], f32>>, !mo.tensor<[2, 3], f32>
      ) -> !mo.list<!mo.tensor<[2, ?], f32>>
    ```
  }];

  let arguments = (ins MO_List:$input,
                       MO_Tensor:$element);
  let results = (outs MO_List:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ListCreateOp : MO_TensorOp<"list.create", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "List creation operation";
  let description = [{
    Creates a new list which contains the given inputs (currently limited to
    tensors).

    Example:

    ```mlir
      %arg0: !mo.tensor<[2, 3], f32>
      %arg1: !mo.tensor<[2, 5], f32>

      %list = mo.list.create(%arg0, %arg1) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
      ) -> !mo.list<!mo.tensor<[2, ?], f32>>
    ```
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs MO_List:$result);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
}

def MO_ListGetOp : MO_TensorOp<"list.get", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "List access operation";
  let description = [{
    Retrieves the element at given (base zero) index of the list. The index can be
    negative, with numpy indexing semantics, i.e., `-1` means the last element,
    `-2` means the second to last element, etc.

    Example:

    ```mlir
      %list: !mo.list<!mo.tensor<[2, 2], f32>>
      %idx:  !mo.tensor<[], si32>

      %tensor = mo.list.get(%list, %idx) : (
        !mo.list<!mo.tensor<[2, 2], f32>>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins MO_List:$input, MO_RankedScalarIndexTensor:$index);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ListInsertOp : MO_TensorOp<"list.insert", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "List insertion operation";
  let description = [{
    Creates a new list that's the result of inserting an element at given (base
    zero) index of a copy of the list (i.e., the op is value semantics and
    doesn't affect the input list).

    The index can be negative, with numpy indexing semantics, i.e., `-1` means
    the last element, `-2` means the second to last element, etc.

    It will push any existing element at the index to after the inserted element
    (which will now be at the index). If the index equals the length of the
    list, the op effectively appends the element to the list.

    Example:

    ```mlir
      %list:   !mo.list<!mo.tensor<[2, 2], f32>>
      %tensor: !mo.tensor<[2, 3], f32>
      %index:  !mo.tensor<[], si32>

      %tensor = mo.list.insert(%list, %tensor, %index) : (
        !mo.list<!mo.tensor<[2, 2], f32>>, !mo.tensor<[2, 3], f32>, !mo.tensor<[], si32>
      ) -> !mo.list<!mo.tensor<[2, ?], f32>>
    ```
  }];

  let arguments = (ins MO_List:$input,
                       MO_Tensor:$element,
                       MO_RankedScalarIndexTensor:$index);
  let results = (outs MO_List:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ListConcatOp : MO_TensorOp<"list.concat", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "List concatenation operation";
  let description = [{
    Creates a new list that's the result of concatenating the two input lists
    together.

    Example:

    ```mlir
      %lhs:   !mo.list<!mo.tensor<[2, 2], f32>>
      %rhs:   !mo.list<!mo.tensor<[2, 2], f32>>

      %tensor = mo.list.concat(%lhs, %rhs) : (
        !mo.list<!mo.tensor<[2, 2], f32>>, !mo.list<!mo.tensor<[2, 2], f32>>
      ) -> !mo.list<!mo.tensor<[2, 2], f32>>
    ```
  }];

  let arguments = (ins MO_List:$lhs,
                       MO_List:$rhs);
  let results  = (outs MO_List:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ListSliceOp : MO_TensorOp<"list.slice", [
  Pure,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "List slice operation";
  let description = [{
    Creates a new list that's the result of applying the equivalent of Python's
    slice operation: list[start:end:step]

    The start/end can be negative, with numpy indexing semantics, i.e., `-1` means
    the last element, `-2` means the second to last element, etc.

    Steps can be negative as well in which case start to end
    are walked in reverse order.

    Example:

    ```mlir
      %list:   !mo.list<!mo.tensor<[2, 2], f32>>
      %index:  !mo.tensor<[], si32>
      %start:  !mo.tensor<[], si32>
      %step:  !mo.tensor<[], si32>

      %tensor = mo.list.slice(%list, %tensor, %index) : (
        !mo.list<!mo.tensor<[2, 2], f32>>,
        !mo.tensor<[], si32>,  !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.list<!mo.tensor<[2, ?], f32>>
    ```
  }];

  let arguments = (ins MO_List:$input,
                       MO_RankedScalarIndexTensor:$start,
                       MO_RankedScalarIndexTensor:$end,
                       MO_RankedScalarIndexTensor:$step);
  let results = (outs MO_List:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_CustomOp : MO_TensorOp<"custom", [
  DeclareOpInterfaceMethods<MO_MutableOpInterface, ["getInChains", "getOutChains", "getInChainsMutable"]>,
  DeclareOpInterfaceMethods<MO_ConditionallyInPlaceInterface>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_ExplicitDevice,
]> {
  let summary = "User-defined operation";
  let description = [{
    The `symbol` attribute specifies which underlying Mojo kernel implements
    this operation. This kernel must be decorated with the appropriate decorator
    with the exact same symbol string.

    The `function` attribute specifies which labeled function the operation
    refers to. Examples: `mogg.shape`.

    Example:

    ```mlir
      %0 = mo.custom {symbol = "test_custom_op"}(%arg0)
        (!mo.tensor<[?], f32>) -> !mo.tensor<[?], f32>
    ```

    Corresponding kernel definition:

    ```mojo
      @register_internal("test_custom_op")
      fn foo(...):
        pass
    ```

    Also allows the definition of custom kernels that have side-effects on
    mo.buffer values via the use of chains.
    (Currently limited to NDBuffer based kernels).

    Nothing needs to change about the kernel definition, but now the custom op
    must take a `mo.buffer` operand for the value it wants to mutate and
    produce an output chain as its final result and take an input chain as its
    final operand.

    Example:

    ```mlir
    %ch1 = mo.custom {symbol = "test_mutable_op"}(%arg0, %ch0) :
      (!mo.buffer<[D1, D2], f32>, !mo.chain) -> !mo.chain
    ```
    ```mojo
      @register_internal("test_custom_op")
      fn foo(...):
        pass
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       StrAttr:$symbol,
                       DefaultValuedAttr<M_DeviceRefAttr, "\"cpu\", 0">:$device,
                       DefaultValuedAttr<DictionaryAttr, "{}">:$parameters,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) attr-dict `(` operands `)` `:`
       functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    /// Returns the sole input chain (if present) and std::nullopt otherwise.
    std::optional<ChainValue> maybeGetInChain();

    /// Returns the sole output chain (if present) and std::nullopt otherwise.
    std::optional<ChainValue> maybeGetOutChain();

    /// Collects all operands of type !mo.buffer.
    mlir::SmallVector<Value> getBufferOperands();
  }];
}

def MO_ConcatFromListOp : MO_TensorOp<"concat_from_list", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  Pure,
  MO_HostOnly
]> {
  let summary = "List concatenation operation";
  let description = [{
    Concatenates tensors in the input list along a given dimension.

    This op concatenates tensors in the `input` list into an output tensor. The
    input tensors and output tensors all have the same shape except along the
    concatenation dimension `axis`. The size of the concatenation dimension in
    output tensor would have be the sum of sizes of the concatenation dimension
    in input tensors.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %list: !mo.list<!mo.tensor<[2, 3], f32>>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %res = mo.concat_from_list[%axis: !mo.tensor<[], si64>](%list) :
        !mo.list<!mo.tensor<[2, 3], f32>> -> !mo.tensor<[2, ?], f32>
    ```
  }];

  let arguments = (ins MO_List:$input,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type(results))
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_AssertOp : MO_Op<"assert"> {
  let summary = "Assert that a single boolean value is true.";

  let description = [{
    Asserts that a single boolean value is true.

    Currently bare assert operations are *not* used in MO graphs,
    rather they are wrapped first in a KGEN MLIR op attr, a KGEN apply
    attr and finally materialized as an actual operation via the KGENParamDeclare op.

    In order to properly materialize assert logic for a particular op and ensure
    the execution dependencies are properly tracked you must:

    1. Create each individual assert via calls to AssertOp's emitStaticCall method
    2. Supply all the asserts (really KGENParamDeclare ops) to the materializeAllAssertLogic
       function which will handle linking the individual assertions together via !mo.chains
       and materialize the remaining structures needed to properly track the execution
       dependencies from the asserts to their compute op.

    Example:

    ```mlir
    // When using emitStaticCall and materializeAllAssertLogic
    %X: mo.tensor<[D1, D2]>,
    %Y: mo.tensor<[D3, D4]>

    ...

    kgen.param.declare CH0:
      <apply(:(i1) -> !mo.chain "mo.assert"{message = "Error 1" : !kgen.string}, eq(D1, D3))>

    kgen.param.declare CH1:
      <apply(:(!mo.chain, i1) -> !mo.chain "mo.assert"{message = "Error 2" : !kgen.string}, CH0, eq(D2, D4))>

    %chain1 = mosh.param.to_value = <CH0>
    %chain2 = mosh.param.to_value = <CH1>

    %guard_chain = mo.chain.create(%chain1, %chain2)

    %Xp, Yp = mo.guard[%guard_chain](%X, %Y)

    ...

    %Z = mo.op(%Xp, %Yp)
    ```
  }];

  let arguments = (ins Optional<MO_Chain>:$inChain,
                       I1:$cond,
                       AttrOfType<KGEN_StringType>:$message);
  let results = (outs MO_Chain:$chain);

  let builders = [
    OpBuilder<(ins "Value":$cond, "const llvm::Twine &":$message)>,
  ];

  let assemblyFormat = "(`[`$inChain^`]`)? $cond `,` custom<StringParam>($message) attr-dict";
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Create a static call to apply(mo.assert, cond, message)
    static TypedAttr emitStaticCall(TypedAttr cond, const Twine &message);

    /// Links a sequence of assertions together with the op they're meant to precede by:
    ///
    /// 1: Linking the sequence of asserts together via their chains
    /// 2: Converting the chains produced by each individual assert into SSA values via mosh.param.to_value
    /// 3: Tying these chains all together via mo.chain.create
    /// 4: Creating a guard op consuming the chain from (3) along with all operands of op
    /// 5: Replacing all operands for op with those produced by the mo.guard op introduced in (4)
    static void materializeAllAssertLogic(mlir::RewriterBase &rewriter,
                                          Operation *op,
                                          ArrayRef<KGEN::ParamDeclareOp> asserts);
  }];
}

def MO_GuardOp : MO_Op<"guard", [
  AllTypesMatch<["inputs", "results"]>,
  InferTypeOpAdaptor,
  MO_PreservedDuringKernelLowering,
  Pure
]> {
  let summary = "SSA-value guard operation";
  let description = [{
    Consumes a mo.chain operation and a variadic number of inputs and is
    semantically equivalent to a copy of the variadic inputs.

    This can be used in conjunction with the !mo.chain returning ops
    (currently only mo.assert) in order to set up execution dependencies
    between the chain producing op and any appropriate downstream operations.

    Example:

    ```mlir
      %X: !mo.tensor<[D1, D2], f32>
      %Y: !mo.tensor<[D2, D3], f32>

      %chain : !mo.chain

      %guarded:2 = mo.guard[%chain](%X, %Y)

      %z = mo.matmul(%guarded#0, %guarded#1)
    ```
  }];

  let arguments = (ins MO_Chain:$chain,
                       Variadic<MO_AnyNonBufferType>:$inputs);

  let results = (outs Variadic<MO_AnyNonBufferType>:$results);

  let assemblyFormat = [{
    `[` $chain `]` `(` $inputs `)` attr-dict `:` type($inputs)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def MO_ChainCreateOp : MO_Op<"chain.create", [
  Commutative,
  Pure,
  MO_HasDeviceInterface,
  MO_PreservedDuringKernelLowering,
  MO_ExecutesOnHost
]> {
  let summary = "Create a new chain conditioned on a variadic number of inputs";
  let description = [{
    This operation consumes an arbitrary number of values and produces a chain.
    Can be used for the sequencing for side-effecting ops when they might
    depend on multiple other ops producing chains.

    ```mlir

    %ch0 = mo.assert ...
    %x = mo.matmul ....

    %ch1 = mo.chain.create(%ch0: !mo.chain, %x: mo.tensor)
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs MO_Chain:$result);
  let hasCustomAssemblyFormat = 1;

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Buffer ops
//===----------------------------------------------------------------------===//

def MO_BufferTransferOp : MO_DeviceAgnosticMutableOp<"buffer.transfer", [
  AllDTypesMatch<["src", "dst"]>,
  AllShapeAttrsMatch<["src", "dst"]>,
  MO_PreservedDuringKernelLowering,
]> {
  let summary = "Transfer data between two buffers";
  let description = [{
    This operation transfers data from a source buffer to a destination buffer.
    Both buffers must have the same shape and data type. The operation takes an input
    chain and produces an output chain to sequence the transfer with other operations.

    Example:
    ```mlir
    %outChain = mo.buffer.transfer[%inChain](%src, %dst) : !mo.buffer<[2,3], f32, gpu:0>, !mo.buffer<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Buffer:$src,
    MO_Buffer:$dst,
    MO_Chain:$inChain
  );
  let results = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($src)) `,` qualified(type($dst))
  }];
}


// TODO(GEX-1292): Support parameterization interfaces for this op
// Not currently an issue since parameterization for mo.buffer operands that are
// graph inputs works and this op will only be used in a pass where the shape
// attr of the results of this op are derived from an existing mo.buffer value.
def MO_BufferCreateOp : MO_Op<"buffer.create", [
  Pure,
  MO_PreservedDuringKernelLowering
]> {
  let summary = "Creates an uninitialized buffer on a specific device.";
  let description = [{
    This operation creates an uninitialized buffer with the specified shape and data type on a given device.
    The buffer is not initialized with any values, and the operation is intended for use cases where the buffer
    will be filled with data later in the computation.

    Example:
    ```mlir
    %buf = mo.buffer.create : !mo.buffer<[20, 20], f32, gpu:0>
    ```
  }];

  let results = (outs MO_Buffer:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def MO_MutableLoadOp : MO_MutableOp<"mutable.load", [
  AllDTypesMatch<["inBuffer", "outTensor"]>,
  AllShapeAttrsMatch<["inBuffer", "outTensor"]>,
  MO_SingleDeviceWithHostOperands</*hostOperands*/[], /*deviceOperands*/["inBuffer"]>,
  // Note: outTensor is a result, not an operand. The trait will generate
  // a device constraint for it by default.
  MO_PreservedDuringKernelLowering
]>
{
  let summary = "Take a chain and a buffer semantic tensor and returns a chain and a value semantic tensor.";
  let description = [{
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.store.

    This is semantically equivalent to a copy from `inBuffer` to `outTensor`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic output of this operation has more than one use the
    operation becomes ineligibile for fusion with compute operations.
  }];

  let arguments = (ins MO_Buffer:$inBuffer, MO_Chain:$inChain);
  let results   = (outs MO_Tensor:$outTensor, MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
}

def MO_MutableStoreOp : MO_MutableOp<"mutable.store", [
  AllDTypesMatch<["inBuffer", "inTensor"]>,
  AllShapeAttrsMatch<["inBuffer", "inTensor"]>,
  MO_SingleDeviceWithHostOperands<
    /*hostOperands*/[],
    /*deviceOperands*/["inBuffer", "inTensor"]>,
]>
{
  let summary = "Takes a chain, a buffer semantic tensor, and a value semantic tensor and returns a chain";
  let description = [{
    Allows modelling of in-place operations in MO in conjunction with mo.mutable.load.

    This is semantically equivalent to a copy from `inTensor` to `inBuffer`

    The output chain of this operation is only allowed to have at most one use.

    If the value semantic tensor input of this operation has more than one use
    the operation becomes ineligibile for fusion with compute operations.
  }];

  let arguments = (ins MO_Buffer:$inBuffer, MO_Tensor:$inTensor, MO_Chain:$inChain);
  let results   = (outs MO_Chain:$outChain);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_MutableStoreSliceOp : MO_MutableOp<"mutable.store.slice", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  MO_SingleDeviceWithHostOperands<
    /*hostOperands*/["start", "stop", "step"],
    /*deviceOperands*/["inBuffer", "slice"]>,
]>
{
  let summary = "Stores a tensor to a slice of a buffer semantic tensor.";
  let description = [{
    Stores the  tensor `slice` to a subset of the elements in `inBuffer`.

    The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
    ```mlir
      %buffer: !mo.buffer<[20, 20], f32>
      %slice: !mo.tensor<[D0, D1], f32>
      %ch: !mo.chain

      %start: !mo.tensor<[2], si64> // [1,  -6]
      %stop: !mo.tensor<[2], si32>  // [-3, -3]
      %step: !mo.tensor<[2], si64>  // [5,   1]

      // equivalent to this in numpy: `buffer[1:-3:5, -6:-3:1] = slice`
      %ch' = mo.mutable.store.slice(%ch, %buffer, %slice, %start, %stop, %step)
    ```

    Both consumes and produces a chain. The output chain is allowed to have at
    most one use.
  }];
  let arguments = (ins MO_Buffer:$inBuffer,
                       MO_Tensor:$slice,
                       MO_IndicesVectorIfRanked:$start,
                       MO_IndicesVectorIfRanked:$stop,
                       MO_IndicesVectorIfRanked:$step,
                       MO_Chain:$inChain,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Chain:$outChain);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Collective ops
//===----------------------------------------------------------------------===//

class MO_DistributedOp<string name, list<Trait> traits = []> : MO_TensorOp<name, [
  DeclareOpInterfaceMethods<MO_MutableOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  MO_SameVariadicOperandSize
] # traits> {
  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results   = (outs Variadic<MO_Tensor>:$outputs);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_AllReduceSumOp : MO_DistributedOp<"distributed.allreduce.sum", [
  MO_ExplicitDevice
]> {
  let summary = "Allreduce sum across N devices for a single device.";
  let description = [{
    Allreduce takes a single input tensor from this device and performs a sum
    reduction across all devices. Peer input addresses are obtained via signal
    buffer payloads at runtime (P2P access required). This op instance executes
    on a specific device (specified by the device attribute) and produces the
    output for that device.

    Multiple instances of this op are created (one per device) to enable
    multi-threaded execution.
  }];
  let hasVerifier = 1;
  // Variadic allreduce has a folder since it can take 1 input.
  let hasFolder = 1;
  let arguments = (ins
    MO_Tensor:$input,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    M_DeviceRefAttr:$device
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_ReduceScatterSumOp : MO_DistributedOp<"distributed.reducescatter.sum", [
  MO_ExplicitDevice
]> {
  let summary = "ReduceScatter sum across N devices for a single device.";
  let description = [{
    ReduceScatter takes in inputs each coming from a different device, and
    partitions the reduction such that each device receives a disjoint subset
    of the result. This op instance executes on a specific device (specified by
    the device attribute) and produces the output for that device.

    Multiple instances of this op are created (one per device) to enable
    multi-threaded execution.
  }];
  let hasVerifier = 1;
  // Variadic ReduceScatter has a folder since it can take 1 input.
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain,
    M_DeviceRefAttr:$device
  );
  let results = (outs
    MO_Tensor:$output,
    MO_Chain:$outChain
  );
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}


def MO_AllGatherOp : MO_DistributedOp<"distributed.allgather", [
  MO_HasDeviceInterface,
  MO_MultiDevice,
]> {
  let summary = "AllGather across N devices (variadic).";
  let description = [{
    AllGather takes in inputs each coming from a different device and collects
    the data into an output tensor along the 0th dimension. The output is
    replicated across the same devices.
  }];
  let hasVerifier = 1;
  // Variadic allgather has a folder since it can take 1 input.
  let hasFolder = 1;
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
}

def MO_MatmulAllReduceOp : MO_DistributedOp<"distributed.matmul_allreduce", [
  MO_HasDeviceInterface,
  MO_MultiDevice,
]> {
  let summary = "Split + overlap Matmul + Allreduce sum";
  let description = [{
    Execute a multi-gpu Matmul + AllReduce.
    This op encode multiple GPU kernels.
    This needs to be a single op to handle kernel overlap / PDL correctly.
    Mega kernel that execute a multi-gpu Matmul + AllReduce.
    The computation is potentially split on the row / col axis
    (For H100+ and large enough dimensions).
    This way we can overlap some of the operations that are independent.
  }];
  let arguments = (ins
    Variadic<MO_Tensor>:$inputs,
    Variadic<MO_Tensor>:$weights,
    Variadic<MO_Buffer>:$signalBuffers,
    MO_Chain:$inChain
  );
  let results = (outs
    Variadic<MO_Tensor>:$outputs,
    MO_Chain:$outChain
  );
}

//===----------------------------------------------------------------------===//
// "Custom" GPU only ops
//===----------------------------------------------------------------------===//

// TODO: Make into custom ops. Currently there's no way to mark custom ops as
// GPU only, because device specificity is tied to individual ops. We will need
// to support CPU only and GPU only custom ops, but that's not possible at the
// moment.

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
