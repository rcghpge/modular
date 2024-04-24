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
//
// We have a family of 6 buffer allocation primitives, depending on choice of
//  {static size vs dynamic size vs dynamic tensor spec} x {alloc vs re-alloc}
//
//  CpuBufferAllocStaticOp: Alloc with statically known size
//    mgp.cpu.buffer.alloc.static.deprecated    %ch,            64
//  CpuBufferAllocDynamicOp: Alloc with dynamic size
//    mgp.cpu.buffer.alloc.dynamic   %ch,            %size
//  CpuBufferReallocStaticOp: Re-alloc with statically known size
//    mgp.cpu.buffer.realloc.static.deprecated  %ch, %ctx[5,1], 64
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic size
//    mgp.cpu.buffer.realloc.dynamic.deprecated %ch, %ctx[5,1], %size
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CpuBufferAllocStaticOp
//===----------------------------------------------------------------------===//

// TODO(#32165): DEPRECATED
def CpuBufferAllocStaticOp : MGP_Op<"cpu.buffer.alloc.static.deprecated", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface>,
    DeclareOpInterfaceMethods<MGP_StaticAllocOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>]> {
  let summary = [{
    Allocates an uninitialized buffer on the CPU with size taken from
    statically known constant.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    Allocates an uninitialized buffer on the CPU whose byte array is of the
    given statically-known size. An optional static alignment constraint may
    also be given.

    Note that any alignment constraint is not propagated into the result type.

    The result buffer will be 'owning', and the underlying memory will be
    reclaimed once the last reference to the result goes out of scope.

    Example:
    ```mlir
    %ch : !mef.chain

    %buf = mgp.cpu.buffer.alloc.static.deprecated %ch, 64, align 8
    %buf : !mgp.cpu.buffer<64>  // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.static.deprecated %ch, 64
    %buf : !mgp.cpu.buffer<64>  // 64 bytes with no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                   UI64Attr:$aSize,                                         // 0
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign); // 1
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $aSize (`,` `align` $bRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool CpuBufferAllocStaticOp::isTrivial() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocDynamicOp
//===----------------------------------------------------------------------===//

// TODO(#32165): DEPRECATED
def CpuBufferAllocDynamicOp : MGP_Op<"cpu.buffer.alloc.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface>,
    DeclareOpInterfaceMethods<MGP_DynamicAllocOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>]> {
  let summary = [{
    Allocates an uninitialized buffer on the CPU with size taken from index
    operand.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    Allocates an uninitialized buffer on the CPU whose byte array is of the
    given operand size. An optional static alignment constraint may also be
    given.

    See `cpu.buffer.alloc.static.deprecated` for additional details.

    Example:
    ```mlir
    %ch : !mef.chain
    %size : index

    %buf = mgp.cpu.buffer.alloc.dynamic %ch, %size align 8
    %buf : !mgp.cpu.buffer  // aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.dynamic %ch, %size
    %buf : !mgp.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                   Index:$runtimeSize,
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $runtimeSize (`,` `align` $rawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocStaticOp
//===----------------------------------------------------------------------===//

// TODO(#32165): DEPRECATED
def CpuBufferReallocStaticOp : MGP_Op<"cpu.buffer.realloc.static.deprecated", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>]> {
  let summary = [{
    (Re)allocates a cached CPU buffer on the CPU with size taken from
    statically known constant.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer. The size must be known
    statically.  An optional static alignment constraint may also be given.

    This op accepts a chain argument so that access to the slot state can
    be guaranteed to come after construction and initialization of the
    state slot at the slot index.

    Though the cache entry will be an 'owning' reference to the buffer,
    the result of this op is always a 'borrowing' reference to the same
    underlying memory. The compiler must respect the following rules to avoid
    use-after-free errors and mutation-via-aliases at runtime:
     1. The 'one shot' rule: For all control flow paths, at most one
        mgp.cpu.buffer.realloc.* can execute for the same slot and buffer
        indexes.
        This rule is easy to follow since no two operations in the graph should
        share the same slot, and buffer indexes should correspond to op result
        indexes.
     2. The 'lifetime' rule: The lifetime of the context must exceed the
        lifetime of the returned buffer. This rule is easy to follow since the
        context is held by the Modular runtime M::ExecutableModel, which must
        outlive both all calls to execute and all uses of aliasing results of
        execute.

    Note that though the buffer will be cached, the contents of the buffer need
    not be preserved between calls to `mgp.cpu.buffer.realloc.*` in successive
    invocations of the model 'execute' body. This rule allows the runtime
    to avoid some buffer copies when extracting the buffer from a tensor
    which itself was created using a cached buffer. In other words, the cache
    is designed to avoid re-allocating buffers rather than re-calculating the
    contents of buffers.

    Note that alignment constraints are ignored when deciding if a cached buffer
    may be reused. In particular, in:
    ```mlir
       init {
         ...
         %buf1 = mgp.cpu.buffer.realloc.static.deprecated %ch, %ctx[2, 0], 48, align 8
       }
       execute (...) -> ... {
         ...
         %buf2 = mgp.cpu.buffer.realloc.static.deprecated %ch, %ctx[2, 0], 48, align 16
       }
    ```
    the second realloc will always reuse the buffer from the first realloc.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch : !mef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.static.deprecated %ch, %ctx[42, 2], 64, align 8
    %buf : !mgp.cpu.buffer<64>   // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.realloc.static.deprecated %ch, %ctx[42, 2], 64
    %buf : !mgp.cpu.buffer<64>  // 64 bytes with no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                   MGP_ContextType:$theContext,
                   UI64Attr:$aSlot,                                         // 0
                   UI64Attr:$bBuffer,                                       // 1
                   UI64Attr:$cSize,                                         // 2
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$dRawAlign); // 3
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `,` $bBuffer `]` `,` $cSize
    (`,` `align` $dRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocDynamicOp
//===----------------------------------------------------------------------===//

// TODO(#32165): DEPRECATED
def CpuBufferReallocDynamicOp : MGP_Op<"cpu.buffer.realloc.dynamic.deprecated", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>]> {
  let summary = [{
    (Re)allocates a cached CPU buffer on the CPU with size taken from index
    operand.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer. The size is given by
    an operand. An optional static alignment constraint may also be given.

    See `cpu.buffer.realloc.static.deprecated` for additional details and restrictions.

    Example:
    ```mlir
    %ctx : !mgp.context
    %size : index
    %ch : !mef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.dynamic.deprecated %ch, %ctx[42, 2], %size, align 8
    %buf : !mgp.cpu.buffer      // 8 bytes aligned

    %buf = mgp.cpu.buffer.realloc.dynamic.deprecated %ch, %ctx[42, 2], %size
    %buf : !mgp.cpu.buffer<64>  // no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                   MGP_ContextType:$theContext,
                   Index:$runtimeSize,
                   UI64Attr:$aSlot,                                         // 0
                   UI64Attr:$bBuffer,                                       // 1
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$cRawAlign); // 3
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `,` $bBuffer `]` `,` $runtimeSize
    (`,` `align` $cRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}


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

def CpuTensorSpecCreateOp : MGP_Op<"cpu.tensor_spec.create", [Pure,
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

    %spec = mgp.cpu.tensor_spec.create %dim0, %dim1 : !mgp.tensor_spec<?x?xf32>
    %spec : !mgp.tensor_spec<?x?xf32>

    %spec = mgp.cpu.tensor_spec.create : !mgp.tensor_spec<32x4xf32>
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

def CpuTensorSpecCreateDynamicOp : MGP_Op<"cpu.tensor_spec.create.dynamic", [
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

    %spec = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim0, %dim1 : !mgp.tensor_spec<?x?x?>
    %spec : !mgp.tensor_spec<?x?x?>

    %spec = mgp.cpu.tensor_spec.create %dtype : !mgp.tensor_spec<32x4x?>
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

def CpuTensorSpecWithDTypeOp : MGP_Op<"cpu.tensor_spec.with_dtype", [
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
    %spec = mgp.cpu.tensor_spec.with_dtype %shape : !mgp.tensor_spec<?x3xf32>
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

def CpuTensorSpecEqualStaticOp : MGP_Op<"cpu.tensor_spec.equal.static", [Pure]> {
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
    %valid = mgp.cpu.tensor_spec.equal.static %spec : !mgp.tensor_spec<?x?xf32>
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

def CpuTensorSpecDimsDynamicOp : MGP_Op<"cpu.tensor_spec.dims.dynamic", [
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
    %dim0, %dim1 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mgp.tensor_spec<32x?xf32>
    %dim = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<32x?xf32>
    %dim : index

    %spec : !mgp.tensor_spec<32x4xf32>
    mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.tensor_spec<32x4xf32> // no-op
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

def CpuTensorSpecSizeOp : MGP_Op<"cpu.tensor_spec.size"> {
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
    %size = mgp.cpu.tensor_spec.size %spec : !mgp.tensor_spec<32x?xf32>
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

def CpuTensorSpecNumElemsOp : MGP_Op<"cpu.tensor_spec.num_elems"> {
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
    %elems = mgp.cpu.tensor_spec.num_elems %spec : !mgp.tensor_spec<32x?xf32>
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

def CpuTensorSpecDTypeOp : MGP_Op<"cpu.tensor_spec.dtype"> {
  let summary = "Returns the DType of the given CPU tensor spec.";
  let description = [{
    Returns the runtime DType of the given CPU tensor spec. Generally the
    tensor spec's type will have an unknown element type (otherwise the result
    is already known).

    Example:
    ```mlir
    %spec : !mgp.tensor_spec<32x?xunk>
    %dtype = mgp.cpu.tensor_spec.dtype %spec : !mgp.tensor_spec<32x?xunk>
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
     [{$_self.cast<MGP::CpuBufferType>().getRawSize().getValue() == }] # size>)>;

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
                       Variadic<AnyType>:$args,
                       UI64Attr:$aRuntimeSlot,       // 0
                       UI64Attr:$bModuleIndex,       // 1
                       UI64Attr:$cFunctionIndex      // 2
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
// ModelOp
//===----------------------------------------------------------------------===//

def ModelFrameworkAttr : ConfinedAttr<StrAttr, [
    AttrConstraint<
      CPred<"CompiledFrameworkLabel::isValidFrameworkName("
            "$_self.cast<StringAttr>().getValue())">,
      "must be a recognized framework name.">]>;

def ModelOp : MGP_Op<"model", [HasParent<"ModuleOp">, Symbol,
                               IsolatedFromAbove, RegionKindInterface,
                               HasOnlyGraphRegion]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    This op creates an executable model from single-block regions
    representing its 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      () -> (!mef.chain, !mgp.context)

    The 'execute' entry point has the implicit signature:
      (!mef.chain, !mgp.context, in1, ..., inn) ->
      (!mef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is given as the op's explicit signature.

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
      init {
        %ch = mef.chain.new
        %ctx = mgp.context.create 0
        mef.output %ch, %ctx : !mef.chain, !mgp.context
      }
      execute (%ch: !mef.chain, %ctx: !mgp.context, %arg: tensor<?x4x16xf32>)
              -> (!mef.chain, tensor<?x10xf32>) {
        mef.output %ch, %arg : !mef.chain, tensor<?x4x16xf32>
      }
    ```
  }];

  // CAUTION: There are three constraints on these attribute names and their
  // order:
  //  - 'sym_name' must match the standard symbol name attribute name (as given
  //    by mlir::SymbolTable::getSymbolAttrName()). Note however that we could
  //    probably choose a more consistent name and implement
  //    DerivedAttributeOpInterface to recover sym_name if we really wanted
  //    that.
  //  - The attributes are accessed by index within M::Module subclasses and
  //    MEF metadata extraction, and that index matches the MLIR canonical
  //    asciibetical order.
  //  - The extra attributes must appear last since optional.
  let arguments = (ins ModelFrameworkAttr:$frameworkName,               // 0
                       TypeAttrOf<FunctionType>:$inputOutputTypes,      // 1
                       SymbolNameAttr:$sym_name,                        // 2
                       OptionalAttr<DictionaryAttr>:$theExtraAttrs);    // 3
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }
  }];

  let builders = [
    /// Creates a model with empty 'init' and 'execute' bodies and
    /// with `inputTypes` and `outputTypes` for the op's explicit signature.
    /// `inputArgLocs` specifies the location information for the
    /// explicit arguments to the 'execute' entry point, and thus should
    /// have the same length as `inputTypes`.
    OpBuilder<(ins "StringAttr":$sym_name,
                   "StringRef":$frameworkName,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$outputTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>
  ];

  let assemblyFormat = [{
    $sym_name `:` custom<ClassName>($frameworkName)
      ($theExtraAttrs^)?
      `init` $init
      `execute` custom<ExecuteRegion>($inputOutputTypes, $execute)
      attr-dict
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

include "GenericML/MGPDialect/MGPDeviceOps.td"

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
