//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPS_TD
#define GENERICML_MGPDIALECT_MGPOPS_TD

include "GenericML/MGPDialect/MGPOpInterfaces.td"
include "GenericML/MGPDialect/MGPTensorSpec.td"
include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "GenericML/MGPDialect/MGPAttrs.td"
include "GenericML/GraphCompiler/MOGGDialect/MOGGInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOAttrs.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.td"
include "GraphRT/GRTDialect/GRTMayAliasInterface.td"
include "GraphRT/MEFDialect/MEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MGP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MGPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MGP_Op<"context.create"> {
  let summary = "Creates a context with given number of state slots.";
  let description = [{
    Creates a context with the given number of 'state slots'.

    Example:
    ```mlir
    %ctx = mgp.context.create 4
    %ctx : !mgp.context
    ```
  }];

  let arguments = (ins UI64Attr:$numSlots);
  let results = (outs MGP_ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}

//===----------------------------------------------------------------------===//
// StateSlotCreateOp
//===----------------------------------------------------------------------===//

def StateSlotCreateOp : MGP_Op<"state_slot.create"> {
  let summary = [{
    Creates a default state slot within the context at the given slot index.
  }];
  let description = [{
    Creates a default state slot at the given slot index with room for the
    given number of cached buffers (which generally represent operator
    outputs). The op returns a chain indicating when it has completed execution.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch0 : !mef.chain

    // Creates a default state slot at index 42 with room for
    // two buffers.
    %ch1 = mgp.state_slot.create %ch0, %ctx[42], 2
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names. Thankfully this matches our desired argument order.
  let arguments = (ins ChainType:$inChain,
                   MGP_ContextType:$theContext,
                   UI64Attr:$numBuffers,  // 0
                   UI64Attr: $slot);      // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $slot `]` `,` $numBuffers attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CheckRuntimeOp
//===----------------------------------------------------------------------===//

def CheckRuntimeOp : MGP_Op<"check_runtime"> {
  let summary = "Reconcile the devices available at runtime with those assumed "
                "at compile time.";
  let description = [{
    This op probes the execution environment to confirm all the computation
    devices required by the model are available. The model requirements are
    expressed as an `#M.device_spec_collection` attribute. The resulting
    device specification map (a C++ `DeviceSpecMap`) is made available for use
    by other primitives within the given `!mgp.context` slot. An error is
    raised if a requirement cannot be met.

    Example:
    ```mlir
    %ch0 : !mef.chain
    %ctx : !mgp.context
    %ch1 = mgp.check_runtime %ch0, %cxt[3],
             <host = <"cpu", 0>,
              devices = [
               <ref = <"cpu", 0>,
                target = <triple="x86_64-unknown-linux-gnu", cpu="znver3", features="+avx2">>,
               <ref = <"cuda", 0>,
                target = <triple="nvptx64-nvidia-cuda", features="+compute_80">>]>
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aSlot,                       // 0
                       M_DeviceSpecCollectionAttr:$bDevices); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `]` `,` $bDevices attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecCreateOp : MGP_Op<"tensor_spec.create", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a CPU tensor spec from a mix of runtime and static dimensions and
    a statically known dtype.
  ]};
  let description = [{
    Creates a CPU tensor spec of statically known rank with dimensions taken
    from either runtime operands or statically known attributes. The element
    type must be known statically. Operands must be given for each unknown
    dimension in order. It is valid for all dimensions to be known statically,
    in which case no operands are required.

    Tensor specs are represented as M::TensorSpecs in the Modular runtime.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mgp.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
    %spec : !mgp.tensor_spec<?x?xf32>

    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<32x4xf32>
    %spec : !mgp.tensor_spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by MEF.
  let arguments = (ins Variadic<Index>:$runtimeDims,
                   UI64ArrayElementsAttr:$aRawDims,  // 0
                   UI8Attr:$bRawDType);              // 1
  let results = (outs MGP_TensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims attr-dict
    `:` custom<RawDimsAndDType>($aRawDims, $bRawDType)
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "TensorSpecType":$spec,
                   CArg<"ValueRange", "{}">:$runtimeDims)>
  ];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateDynamicOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecCreateDynamicOp : MGP_Op<"tensor_spec.create.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a CPU tensor spec from a runtime DType and a mix of runtime and
    static dimensions.
  }];
  let description = [{
    Creates a CPU tensor spec of statically known rank with element type taken
    at runtime by a DType operand, and dimensions taken from either runtime
    operands or statically known attributes. Operands must be given for each
    unknown dimension in order. It is valid for all dimensions to be known
    statically, in which case only an element type needs to be given.

    Tensor specs are represented as M::TensorSpecs in the Modular runtime.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index
    %dtype : !mgp.dtype

    %spec = mgp.tensor_spec.create.dynamic %dtype, %dim0, %dim1 : !mgp.tensor_spec<?x?x?>
    %spec : !mgp.tensor_spec<?x?x?>

    %spec = mgp.tensor_spec.create %dtype : !mgp.tensor_spec<32x4x?>
    %spec : !mgp.tensor_spec<32x4x?>
    ```
  }];

  // NOTE: The attribute effectively encodes the shape of the result's tensor
  // spec type in a form understandable by MEF.
  let arguments = (ins MGP_DTypeType:$runtimeDType,
                   Variadic<Index>:$runtimeDims,
                   UI64ArrayElementsAttr:$rawDims);
  let results = (outs MGP_TensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDType `,` $runtimeDims attr-dict
                          `:` custom<RawDims>($rawDims) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecWithDTypeOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecWithDTypeOp : MGP_Op<"tensor_spec.with_dtype", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a CPU tensor spec from a copy of an existing spec but with
    statically known dtype.
  ]};
  let description = [{
    Creates a CPU tensor spec as a copy of an existing spec but with a
    statically known dtype. The existing spec must have an unknown dtype.

    Example:
    ```mlir
    %shape : !mgp.tensor_spec<?x3xunk>
    %spec = mgp.tensor_spec.with_dtype %shape : !mgp.tensor_spec<?x3xf32>
    %spec : !mgp.tensor_spec<?x3xf32>
    ```
  }];

  let arguments = (ins MGP_TensorSpecType:$tensorSpec,
                   UI8Attr:$rawDType);
  let results = (outs MGP_TensorSpecType:$result);
  let assemblyFormat = [{
    $tensorSpec attr-dict `:` custom<RawDType>(type($tensorSpec), $rawDType)
  }];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecEqualStaticOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecEqualStaticOp : MGP_Op<"tensor_spec.equal.static", [Pure]> {
  let summary = "Returns true if a tensor spec matches static expectations";
  let description = [{
    Mostly used to validate user provided tensors by checking if the tensor has
    the same shape as the program expects. This is necessary because a user can
    provide inputs of different shapes which are not always validated by
    frameworks. Only statically known information is considered in the
    comparison.

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<2x?xf32>
    // Will return true if first dimension is 2 and data type is f32.
    %valid = mgp.tensor_spec.equal.static %spec : !mgp.tensor_spec<?x?xf32>
    ```
  }];

  let arguments = (ins MGP_TensorSpecType: $spec,
                   UI64ArrayElementsAttr:$rawDims);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $spec attr-dict `:` custom<TensorSpecTypeWithRawDims>(type($spec), $rawDims)
  }];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDimsDynamicOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecDimsDynamicOp : MGP_Op<"tensor_spec.dims.dynamic", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Returns the runtime values of the dimensions of CPU tensor spec
    which are not statically known.
  ]};
  let description = [{
    Returns the actual runtime value of each dimension in a CPU tensor spec
    where that dimension is not known statically. It is valid for the tensor
    spec to be fully static, in which case no results are returned (and the op
    is a no-op).

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<?x?xf32>
    %dim0, %dim1 = mgp.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mgp.tensor_spec<32x?xf32>
    %dim = mgp.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<32x?xf32>
    %dim : index

    %spec : !mgp.tensor_spec<32x4xf32>
    mgp.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<32x4xf32> // no-op
    ```
  }];

  let arguments = (ins MGP_TensorSpecType: $spec,
                   UI64ArrayElementsAttr:$rawDims);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    $spec attr-dict `:` custom<TensorSpecTypeWithRawDims>(type($spec), $rawDims)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecSizeOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecSizeOp : MGP_Op<"tensor_spec.size"> {
  let summary = [{
    Returns the byte size needed for a CPU buffer for a tensor with
    the given tensor spec.
  }];
  let description = [{
    Returns the number of bytes needed to represent the CPU buffer backing a
    tensor of the given tensor spec. Generally the tensor spec's type will
    contain at least one unknown dimension and/or an unknown element type
    (otherwise the size is already known).

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<32x?xf32>
    %size = mgp.tensor_spec.size %spec : !mgp.tensor_spec<32x?xf32>
    %size : index
    ```
  }];

  let arguments = (ins MGP_TensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecNumElemsOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecNumElemsOp : MGP_Op<"tensor_spec.num_elems"> {
  let summary = [{
    Returns the number of elements in a CPU tensor with the given
    tensor spec.
  }];
  let description = [{
    Returns the number of elements in a CPU tensor of the given tensor spec.
    Generally the tensor spec's type will contain at least one unknown
    dimension (otherwise the number of elements is already known).

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<32x?xf32>
    %elems = mgp.tensor_spec.num_elems %spec : !mgp.tensor_spec<32x?xf32>
    %elems : index
    ```
  }];

  let arguments = (ins MGP_TensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDTypeOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecDTypeOp : MGP_Op<"tensor_spec.dtype"> {
  let summary = "Returns the DType of the given CPU tensor spec.";
  let description = [{
    Returns the runtime DType of the given CPU tensor spec. Generally the
    tensor spec's type will have an unknown element type (otherwise the result
    is already known).

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<32x?xunk>
    %dtype = mgp.tensor_spec.dtype %spec : !mgp.tensor_spec<32x?xunk>
    %dtype : !mgp.dtype
    ```
  }];

  let arguments = (ins MGP_TensorSpecType: $spec);
  let results = (outs MGP_DTypeType:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// DebugBoolPrintOp
//===----------------------------------------------------------------------===//

def DebugBoolPrintOp : MGP_Op<"debug.bool.print"> {
  let summary = "Prints a debug representation of argument boolean. For "
                "debugging and testing only.";
  let description = [{
    Prints a debug representation of argument boolean. For lit unit tests.

    Example:
    ```mlir
    %b : i1
    %ch0 : !mef.chain
    %ch1 = mgp.debug.bool.print %ch0, %b : i1
    %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain, I1: $cond);
  let results = (outs ChainType: $outChain);
  let assemblyFormat = "$inChain `,` $cond attr-dict `:` qualified(type($cond))";
}


//===----------------------------------------------------------------------===//
// DebugCpuSpinWaitOp
//===----------------------------------------------------------------------===//

def DebugCpuSpinWaitOp : MGP_Op<"debug.cpu.spin_wait"> {
  let summary = "Spins as a sub-task for given delay. Entirely CPU bound.";
  let description = [{
    Spins as a sub-task for given delay in microseconds. Entirely CPU bound.
    Can be used to test for linear scaling w.r.t. number of threads on the
    LLCL runtime. Default delay is 100us.

    Example:
    ```mlir
    %ch0 : !mef.chain
    %ch1 = mgp.debug.cpu.spin_wait %ch0, 30
    %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       DefaultValuedAttr<UI64Attr, "100">:$delayUs);
  let results = (outs ChainType: $outChain);
  let assemblyFormat = "$inChain (`,` $delayUs^)? attr-dict";
}

//===----------------------------------------------------------------------===//
// ToGRTBoolOp
//===----------------------------------------------------------------------===//

// Checks if a MGP_CpuBufferType has one of the specified ranks.
class HasBufferSizePred<list<int> sizes> :
  Or<!foreach(size, sizes,
    CPred<
     [{cast<MGP::CpuBufferType>($_self).getRawSize().getValue() == }] # size>)>;

def MGP_ScalarBuffer : ConfinedType<MGP_CpuBufferType, [HasBufferSizePred<[1]>],
    "scalar buffer"> {
  let cppClassName = "::M::MGP::CpuBufferType";
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : MGP_Op<"assert", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>]> {
  let summary = "Assert that a boolean condition is true.";
  let description = [{
    Return an error with the given message if the condition is false.

    Example:
    ```mlir
    %ch = mgp.assert %cond, "condition is not true"
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins I1:$cond, StrAttr:$message);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = "$cond `,` $message attr-dict";
}

//===----------------------------------------------------------------------===//
// ExecuteGenericOp
//===----------------------------------------------------------------------===//

def ExecuteGenericOp : MGP_Op<"generic.execute"> {
  let summary = "Executes a function loaded into the CPU Runtime";
  let description = [{
    Executes a function already loaded into the compiled kernel context at the
    given slot/module index within the `!mgp.context`. This will execute the function using
    the generic API. The primitive will pass in raw async values for the func
    to unpack internally.

    This execution will always return an output chain, in addition to the results
    of the kernel.  The chain can be used for error handling or side-effect dependencies.
  }];

  let arguments = (ins MGP_ContextType:$theContext,
                       ChainType:$inChain, // Needed to ensure that the execute happens after module creation
                       Variadic<AnyType>:$args,
                       UI64Attr:$aRuntimeSlot,                        // 0
                       UI64Attr:$bModuleIndex,                        // 1
                       UI64Attr:$cFunctionIndex,                      // 2
                       StrAttr:$dSummary,                             // 3
                       DefaultValuedAttr<BoolAttr, "false">:$eIsAsync,// 4
                       DefaultValuedAttr<BoolAttr, "false">:$fHasCuda,// 5 // This should be removed once we have device context
                       UI64Attr:$gCudaSlot                            // 6
                       );
  let results = (outs MGP_DeviceChainType: $outChain, Variadic<AnyType>:$outResults);
}

//===----------------------------------------------------------------------===//
// ListAppend
//===----------------------------------------------------------------------===//

def MGP_ListAppendOp : MGP_Op<"list.append", [
  // The input list elements and appended element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List append Operation.";
  let description = [{
    Creates a new list that's the result of appending an element to the end of a
    copy of the input list (i.e., the op is value semantics and doesn't affect
    the input list).

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.tensor<3x4xf32>,
    %t = mgp.list.append(%lst, %tns) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x4xf32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       AnyType:$element);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListConcat
//===----------------------------------------------------------------------===//

def MGP_ListConcatOp : MGP_Op<"list.concat", [
  // The input elements from the two lists are shallow copied into the result list
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List concat Operation.";
  let description = [{
    Creates a new list that's the result of concatenating the two input lists
    together.

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.list<!mgp.tensor<3x4xf32>>,
    %t = mgp.list.concat(%lst, %tns) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.list<!mgp.tensor<3x4xf32>>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$lhs,
                       MGP_List:$rhs);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListCreate
//===----------------------------------------------------------------------===//

def ListCreateOp : MGP_Op<"list.create", [
  // The inputs are shallow copied into the output list
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "Create a new list with given elements.";
  let description = [{
    Creates a new `!mgp.list` with the inputs as elements in the same order.

    Example:
    ```mlir
    %t1 : !mgp.tensor<3x5xf32>,
    %t2 : !mgp.tensor<3x10xf32>
    %l = mgp.list.create(%t1, %t2) : (
      !mgp.tensor<3x5xf32>, !mgp.tensor<3x10xf32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs MGP_List:$result);
  let assemblyFormat = [{
    ` ` $inputs ` ` attr-dict `:` functional-type($inputs, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListGet
//===----------------------------------------------------------------------===//

def MGP_ListGetOp : MGP_Op<"list.get", [
  // The retrieved element is shallow copied
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "List get Operation.";
  let description = [{
    Retrive the element at given (base zero) index of the list. The index can be
    negative, with numpy indexing semantics, i.e., `-1` means the last element,
    `-2` means the second to last element, etc.

    Example:

    ```mlir
    %lst : !mgp.list<!mgp.tensor<3x?xf32>>,
    %idx : !mgp.tensor<si32>
    %t = mgp.list.get(%lst, %idx) :
      !mgp.list<!mgp.tensor<3x?xf32>>, !mgp.tensor<si32>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       // TODO: Should be IndexType
                       MGP_DeviceTensorType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` qualified(type(operands))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListInsert
//===----------------------------------------------------------------------===//

def MGP_ListInsertOp : MGP_Op<"list.insert", [
  // The input list elements and inserted element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List insert Operation.";
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
    %lst : !mgp.list<!mgp.tensor<3x3xf32>>,
    %tns : !mgp.tensor<3x4xf32>,
    %idx : !mgp.tensor<si32>
    %t = mgp.list.insert(%lst, %tns, %idx) : (
      !mgp.list<!mgp.tensor<3x3xf32>>, !mgp.tensor<3x4xf32>, !mgp.tensor<si32>
    ) -> !mgp.list<!mgp.tensor<3x?xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       AnyType:$element,
                       // TODO: Should be IndexType
                       MGP_DeviceTensorType:$index);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ListInsert
//===----------------------------------------------------------------------===//

def MGP_ListSliceOp : MGP_Op<"list.slice", [
  // The input list elements and inserted element are all shallow copied into
  // the output list.
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "List slice operation.";
  let description = [{
    Creates a new list that's the result of slicing the input list.

    The slicing semantics follow those of Python

    ```mlir
      %lst   : !mgp.list<!mgp.tensor<3x3xf32>>,
      %start : !mgp.tensor<si32>
      %end   : !mgp.tensor<si32>
      %step  : !mgp.tensor<si32>
      %t = mgp.list.insert(%lst, %start, %end, %step) : (
        !mgp.list<!mgp.tensor<3x3xf32>>,
        !mgp.tensor<si32>, !mgp.tensor<si32>, !mgp.tensor<si32>
      ) -> !mgp.list<!mgp.tensor<3x3xf32>>
    ```
  }];

  let arguments = (ins MGP_List:$input,
                       MGP_DeviceTensorType:$start,
                       MGP_DeviceTensorType:$end,
                       MGP_DeviceTensorType:$step);
  let results = (outs MGP_List:$result);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

def GuardOp : MGP_Op<"guard", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]>
{
  let summary = "SSA-value guard operation";

  let description = [{
    This operation consumes a chain and a variadic list of values and forwards
    the non-chain inputs through.

    This can be used with !mef.chain returning ops in order to establish
    execution dependencies between the chain producing op and any downstream
    operations.

    Example:

    ```mlir
      %X: !mgp.tensor<?x?xf32>
      %Y: !mgp.tensor<?x?xf32>

      %spec1:2 = mgp.tensor_spec.dims.dynamic %0 : !mgp.tensor_spec<?x?xf32>
      %spec2:2 = mgp.tensor_spec.dims.dynamic %2 : !mgp.tensor_spec<?x?xf32>

      %cond = mip.cmp.eq %spec1#1, %spec2#0

      %chain = mgp.assert %cond, "D1 != D2"
      %Xp, %Yp = mgp.guard %chain, %X, %Y : !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>
    ```
  }];

  let arguments = (ins ChainType:$chain,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    ` ` operands ` ` attr-dict `:` qualified(type($inputs))
  }];

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelFrameworkAttr : ConfinedAttr<StrAttr, [
    AttrConstraint<
      CPred<"CompiledFrameworkLabel::isValidFrameworkName("
            "cast<StringAttr>($_self).getValue())">,
      "must be a recognized framework name.">]>;

def ModelOp : MGP_Op<"model", [HasParent<"ModuleOp">,
                               IsolatedFromAbove, RegionKindInterface,
                               HasOnlyGraphRegion]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    This op creates an executable model from single-block regions
    representing its 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      (!mgp.opaque, ...) -> (!mef.chain, !mgp.context, !mgp.opaque, ...)
    where:
      mgp.opaques are device contexts

    The 'execute' entry point has the implicit signature:
      (!mef.chain, !mgp.context, dev0, ... devn, in1, ..., inn) ->
      (!mef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is given as the op's explicit signature. And dev0, ..., devn are the
    devices passed to init.

    The Modular runtime implicitly calls 'init' when the model executable is
    first loaded, and re-uses the resulting context value for each subsequent
    call to 'execute'.

    This op is represented by a subclass of M::Model in the Modular runtime.
    The actual class is determined from the `frameworkName` attribute, which
    tools can inspect to determine which API to establish to mediate
    between framework-specific values and representations used by the model.

    Additional attributes may be supplied. Currently the following attribute
    names have special meaning:
     - argument_names
     - result_names

    Example:
    ```mlir
    // At the top-level
    mgp.model @example_model : tfl { attr="value" }
      init (%cpu: !mgp.opaque) {
        %ch = mef.chain.new
        %ctx = mgp.context.create 0
        mef.output %ch, %ctx, %cpu : !mef.chain, !mgp.context, !mgp.opaque
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %cpu: mgp.opaque, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        mef.output %ch, %arg : !mef.chain, tensor<?x4x16xf32>
      }
    ```
  }];

  // CAUTION: There are three constraints on these attribute names and their
  // order:
  //  - 'name' must match the standard symbol name attribute name (as given
  //    by mlir::SymbolTable::getSymbolAttrName()). Note however that we could
  //    probably choose a more consistent name and implement
  //    DerivedAttributeOpInterface to recover name if we really wanted
  //    that.
  //  - The attributes are accessed by index within M::Module subclasses and
  //    MEF metadata extraction, and that index matches the MLIR canonical
  //    asciibetical order.
  //  - The extra attributes must appear last since optional.
  let arguments = (ins M_DeviceRefArrayAttr:$devices,                   // 0
                       ModelFrameworkAttr:$frameworkName,               // 1
                       TypeAttrOf<FunctionType>:$inputOutputTypes,      // 2
                       SymbolNameAttr:$name,                            // 3
                       OptionalAttr<DictionaryAttr>:$theExtraAttrs);    // 4
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }
    MutableArrayRef<BlockArgument> getExecuteInputs() { return getExecuteBlock()->getArguments().drop_front(2+getDevices().size());}
    MutableArrayRef<BlockArgument> getExecuteDevices() { return getExecuteBlock()->getArguments().drop_front(2).take_front(getDevices().size());}
    MutableArrayRef<BlockArgument> getExecuteImplicitArguments() { return getExecuteBlock()->getArguments().take_front(2+getDevices().size());}
    BlockArgument getExecuteContext() { return getExecuteBlock()->getArgument(1); }
  }];

  let builders = [
    /// Creates a model with empty 'init' and 'execute' bodies and
    /// with `inputTypes` and `outputTypes` for the op's explicit signature.
    /// `inputArgLocs` specifies the location information for the
    /// explicit arguments to the 'execute' entry point, and thus should
    /// have the same length as `inputTypes`.
    OpBuilder<(ins "StringAttr":$name,
                   "StringRef":$frameworkName,
                   "ArrayRef<DeviceRef>":$devices,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$outputTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>
  ];

  let assemblyFormat = [{
    $name `:` custom<ClassName>($frameworkName)
      ($theExtraAttrs^)?
      `devices` $devices
      `init` custom<InitRegion>($init)
      `execute` custom<ExecuteRegion>($inputOutputTypes, $execute, ref($devices))
      attr-dict
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

include "GenericML/MGPDialect/MGPDeviceOps.td"

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
