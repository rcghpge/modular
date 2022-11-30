//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPOPS_TD
#define GENERICML_MOPDIALECT_MOPOPS_TD

include "GraphRT/BEFDialect/BEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "GenericML/MOPDialect/MOPDialect.td"
include "GenericML/MOPDialect/MOPTypes.td"

// Base class for ops in this dialect.
class MOP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MOPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CpuBufferConstantOp
//===----------------------------------------------------------------------===//

def CpuBufferConstantOp : MOP_Op<"cpu.buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a buffer hosted on the CPU whose data is aliased with
    the bytes in the given attribute.
  }];
  let description = [{
    Creates a CPU-hosted buffer whose byte array is aliased with the byte array
    in the given attribute. The attribute must support contiguous iteration
    over `uint8_t`. The result buffer type will have a size matching the
    attribute size.

    Example valid attributes include `AlignedBytesAttr` (in the M dialect),
    `DenseUI8ResourceElementsAttr` and `DenseI8Array`. Any alignment specified
    by the `AlignedBytesAttr` and `DenseUI8ResourceElementsAttr` attributes
    will be preserved through to runtime.

    The result buffer will 'borrow' the underlying data. In debug builds
    the result buffer will also me marked 'immutable'.

    Example:
    ```mlir
    %buf = mop.cpu.buffer.constant #m.aligned_buffer<8: "0x00010203">
    %buf : !mop.cpu.buffer<4>  // at least 8 byte aligned in storage

    %buf = mop.cpu.buffer.constant dense_resource<label>:!m.array<4xui8>
    %buf : !mop.cpu.buffer<4>  // respecting alignment encoded in blob at label

    %buf = mop.cpu.buffer.constant array<ui8: 0, 1, 2, 3>
    %buf : !mop.cpu.buffer<4>  // no specific alignment in storage
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the optional alignment constraint for the buffer, if any.
    Optional<uint64_t> getOptAlign();
  }];
}


//===----------------------------------------------------------------------===//
//
// We have a family of 6 buffer allocation primitives, depending on choice of
//  {static size vs dynamic size vs dynamic tensor spec} x {alloc vs re-alloc}
//
//  CpuBufferAllocStaticOp: Alloc with statically known size
//    mop.cpu.buffer.alloc.static        %ctx      64
//  CpuBufferAllocDynamicOp: Alloc with dynamic size
//    mop.cpu.buffer.alloc.dynamic       %ctx      %size
//  CpuBufferAllocSpecOp: Alloc with dynamic tensor spec
//    mop.cpu.buffer.alloc.spec          %ctxt     %spec
//  CpuBufferReallocStaticOp: Re-alloc with statically known size
//    mop.cpu.buffer.realloc.static  %ch %ctx[5,1] 64
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic size
//    mop.cpu.buffer.realloc.dynamic %ch %ctx[5,1] %size
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic tensor spec
//    mop.cpu.buffer.realloc.spec    %ch %ctx[5,1] %spec
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CpuBufferAllocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocStaticOp : MOP_Op<"cpu.buffer.alloc.static", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
    reclaimed once the last reference to the result goes out of scope. In
    debug builds the buffer will also be marked 'mutable'.

    Example:
    ```mlir
    %ctx : !mop.context

    %buf = mop.cpu.buffer.alloc.static %ctx 64 align 8
    %buf : !mop.cpu.buffer<64>  // 64 bytes aligned to 8 bytes

    %buf = mop.cpu.buffer.alloc.static %ctx 64
    %buf : !mop.cpu.buffer<64>  // 64 bytes with no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MOP_ContextType:$theContext,
                       UI64Attr:$aSize,                                         // 0
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign); // 1
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $aSize (`align` $bRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocDynamicOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocDynamicOp : MOP_Op<"cpu.buffer.alloc.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Allocates an uninitialized buffer on the CPU with size taken from index
    operand.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    Allocates an uninitialized buffer on the CPU whose byte array is of the
    given operand size. An optional static alignment constraint may also be
    given.

    See `cpu.buffer.alloc.static` for additional details.

    Example:
    ```mlir
    %ctx : !mop.context
    %size : index

    %buf = mop.cpu.buffer.alloc.dynamic %ctx %size align 8
    %buf : !mop.cpu.buffer  // aligned to 8 bytes

    %buf = mop.cpu.buffer.alloc.dynamic %ctx %size
    %buf : !mop.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins MOP_ContextType:$theContext,
                       Index:$runtimeSize,
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $runtimeSize (`align` $rawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocSpecOp : MOP_Op<"cpu.buffer.alloc.spec", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Allocates an uninitialized buffer on the CPU with size taken from tensor
    spec operand.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    Allocates an uninitialized buffer on the CPU whose byte array is of the
    byte size of the given tensor spec operand. An optional static alignment
    constraint may also be given.

    See `cpu.buffer.alloc.static` for additional details.

    Example:
    ```mlir
    %ctx : !mop.context
    %spec : !mop.tensor_spec<?x64xf32>

    %buf = mop.cpu.buffer.alloc.spec %ctx %spec align 8 : !mop.cpu.tensor_spec<?x64xf32>
    %buf : !mop.cpu.buffer  // aligned to 8 bytes

    %buf = mop.cpu.buffer.alloc.spec %ctx %spec : !mop.cpu.tensor_spec<?x64xf32>
    %buf : !mop.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins MOP_ContextType:$theContext,
                       MOP_CpuTensorSpecType:$runtimeSpec,
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $runtimeSpec (`align` $rawAlign^)? attr-dict
    `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocStaticOp : MOP_Op<"cpu.buffer.realloc.static", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
    underlying memory. (In debug builds both refs will also be marked
    'mutable'). The compiler must respect the following rules to avoid
    use-after-free errors and mutation-via-aliases at runtime:
     1. The 'one shot' rule: For all control flow paths, at most one
        mop.cpu.buffer.realloc.* can execute for the same slot and buffer
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
    not be preserved between calls to `mop.cpu.buffer.realloc.*` in successive
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
         %buf1 = mop.cpu.buffer.realloc.static %ch %ctx[2, 0] 48 align 8
       }
       execute (...) -> ... {
         ...
         %buf2 = mop.cpu.buffer.realloc.static %ch %ctx[2, 0] 48 align 16
       }
    ```
    the second realloc will always reuse the buffer from the first realloc.

    Example:
    ```mlir
    %ctx : !mop.context
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mop.cpu.buffer.realloc.static %ch %ctx[42, 2] 64 align 8
    %buf : !mop.cpu.buffer<64>   // 64 bytes aligned to 8 bytes

    %buf = mop.cpu.buffer.realloc.static %ch %ctx[42, 2] 64
    %buf : !mop.cpu.buffer<64>  // 64 bytes with no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$ch,
                       MOP_ContextType:$theContext,
                       UI64Attr:$aSlot,                                         // 0
                       UI64Attr:$bBuffer,                                       // 1
                       UI64Attr:$cSize,                                         // 2
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$dRawAlign); // 3
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $ch $theContext `[` $aSlot `,` $bBuffer `]` $cSize
    (`align` $dRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocDynamicOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocDynamicOp : MOP_Op<"cpu.buffer.realloc.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

    See `cpu.buffer.realloc.static` for additional details and restrictions.

    Example:
    ```mlir
    %ctx : !mop.context
    %size : index
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mop.cpu.buffer.realloc.dynamic %ch %ctx[42, 2] %size align 8
    %buf : !mop.cpu.buffer      // 8 bytes aligned

    %buf = mop.cpu.buffer.realloc.dynamic %ch %ctx[42, 2] %size
    %buf : !mop.cpu.buffer<64>  // no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$ch,
                       MOP_ContextType:$theContext,
                       Index:$runtimeSize,
                       UI64Attr:$aSlot,                                         // 0
                       UI64Attr:$bBuffer,                                       // 1
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$cRawAlign); // 3
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $ch $theContext `[` $aSlot `,` $bBuffer `]` $runtimeSize
    (`align` $cRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocSpecOp : MOP_Op<"cpu.buffer.realloc.spec", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    (Re)allocates a cached CPU buffer on the CPU with size taken from operand.
  }];
  let description = [{
    One of a family of 6 buffer allocation primitives.

    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer. The size is given by
    an operand. An optional static alignment constraint may also be given.

    See `cpu.buffer.realloc.static` for additional details and restrictions.

    Example:
    ```mlir
    %ctx : !mop.context
    %spec : !mop.cpu.tensor_spec<?x64xf32>
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mop.cpu.buffer.realloc.spec %ch %ctx[42, 2] %spec align 8 : !mop.cpu.tensor_spec<?x64xf32>
    %buf : !mop.cpu.buffer      // 8 bytes aligned

    %buf = mop.cpu.buffer.realloc.spec %ch %ctx[42, 2] : %spec !mop.cpu.tensor_spec<?x64xf32>
    %buf : !mop.cpu.buffer<64>  // no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$ch,
                       MOP_ContextType:$theContext,
                       MOP_CpuTensorSpecType:$runtimeSpec,
                       UI64Attr:$aSlot,                                         // 0
                       UI64Attr:$bBuffer,                                       // 1
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$cRawAlign); // 3
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $ch $theContext `[` $aSlot `,` $bBuffer `]` $runtimeSpec
    (`align` $cRawAlign^)? attr-dict `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferCopyOp
//===----------------------------------------------------------------------===//

def CpuBufferCopyOp : MOP_Op<"cpu.buffer.copy", [
    AllTypesMatch<["buffer", "result"]>]> {
  let summary = "Copies the contents of a CPU buffer.";
  let description = [{
    This op returns a copy of a CPU buffer. The result will be 'owning'.
    In debug builds the result will also be 'mutable'.

    Generally this op should only be needed when testing the use
    of `mop.cpu.buffer.realloc` using a `!mop.context` which is created
    within the test itself. By rule 2 of `mop.cpu.buffer.realloc` the
    result of `mop.cpu.buffer.realloc` cannot outlive the context itself,
    so a copy is needed to return that buffer to the test harness
    for printing etc. It takes `!mop.context` as an argument to
    make sure that the context survives until the copy is performed.

    Any alignment constraint on the operand buffer is preserved in
    the result.

    Example:
    ```mlir
    %ctx : !mop.context
    %buf1 : !mop.cpu.buffer
    %buf2 = mop.cpu.buffer.copy %ctx %buf1 : !mop.cpu.buffer
    %buf2 : !mop.cpu.buffer  // with same alignment as %buf1
    ```
  }];

  let arguments = (ins MOP_ContextType:$theContext, MOP_CpuBufferType:$buffer);
  let results = (outs MOP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $buffer attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferSetOp
//===----------------------------------------------------------------------===//

def CpuBufferSetOp : MOP_Op<"cpu.buffer.set", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Set each byte of the given buffer to the given value (default = 0).
  }];
  let description = [{
    Sets each byte of the allocated buffer to the given value (default = 0).

    Example:
    ```mlir
    %buf : !mop.cpu.buffer<>

    // Set each byte of the buffer to 0.
    %ch = mop.cpu.buffer.set %buf : !mop.cpu.buffer<>
    // Set each byte of the buffer to 1.
    %ch = mop.cpu.buffer.set %buf 1 : !mop.cpu.buffer<>
    ```
  }];

  let arguments = (ins MOP_CpuBufferType:$buffer,
                       DefaultValuedAttr<UI8Attr, "0">:$value);
  let results = (outs ChainType:$result);
  let assemblyFormat = [{
    $buffer ($value^)? attr-dict `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MOP_Op<"context.create"> {
  let summary = "Creates a context with given number of state slots.";
  let description = [{
    Creates a context with the given number of 'state slots'.

    Example:
    ```mlir
    %ctx = mop.context.create 4
    %ctx : !mop.context
    ```
  }];

  let arguments = (ins UI64Attr:$numSlots);
  let results = (outs MOP_ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}

//===----------------------------------------------------------------------===//
// StateSlotCreateOp
//===----------------------------------------------------------------------===//

def StateSlotCreateOp : MOP_Op<"state_slot.create"> {
  let summary = [{
    Creates a default state slot within the context at the given slot index.
  }];
  let description = [{
    Creates a default state slot at the given slot index with room for the
    given number of cached buffers (which generally represent operator
    outputs). The op returns a chain indicating when it has completed execution.

    Example:
    ```mlir
    %ctx : !mop.context

    // Creates a default state slot at index 42 with room for
    // two buffers.
    %ch = mop.state_slot.create %ctx[42] 2
    %ch : !bef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names. Thankfully this matches our desired argument order.
  let arguments = (ins MOP_ContextType:$theContext,
                       UI64Attr:$numBuffers,  // 0
                       UI64Attr: $slot);      // 1
  let results = (outs ChainType:$result);
  let assemblyFormat = "$theContext `[` $slot `]` $numBuffers attr-dict";
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecCreateOp : MOP_Op<"cpu.tensor_spec.create", [
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

    %spec = mop.cpu.tensor_spec.create %dim0 %dim1 : !mop.cpu.tensor_spec<?x?xf32>
    %spec : !mop.cpu.tensor_spec<?x?xf32>

    %spec = mop.cpu.tensor_spec.create : !mop.cpu.tensor_spec<32x4xf32>
    %spec : !mop.cpu.tensor_spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by BEF.
  let arguments = (ins Variadic<Index>:$runtimeDims,
                       UI64ArrayElementsAttr:$aRawDims,  // 0
                       UI8Attr:$bRawDType);              // 1
  let results = (outs MOP_CpuTensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims attr-dict
    `:` custom<RawDimsAndDType>($aRawDims, $bRawDType)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateDynamicOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecCreateDynamicOp : MOP_Op<"cpu.tensor_spec.create.dynamic", [
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
    %dtype : !mop.dtype

    %spec = mop.cpu.tensor_spec.create.dynamic %dtype %dim0, %dim1 : !mop.cpu.tensor_spec<?x?x?>
    %spec : !mop.cpu.tensor_spec<?x?x?>

    %spec = mop.cpu.tensor_spec.create %dtype : !mop.cpu.tensor_spec<32x4x?>
    %spec : !mop.cpu.tensor_spec<32x4x?>
    ```
  }];

  // NOTE: The attribute effectively encodes the shape of the result's tensor
  // spec type in a form understandable by BEF.
  let arguments = (ins MOP_DTypeType:$runtimeDType, Variadic<Index>:$runtimeDims,
                       UI64ArrayElementsAttr:$rawDims);
  let results = (outs MOP_CpuTensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDType $runtimeDims attr-dict
                          `:` custom<RawDims>($rawDims) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDimsDynamicOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecDimsDynamicOp : MOP_Op<"cpu.tensor_spec.dims.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
    %spec : !mop.cpu.tensor_spec<?x?xf32>
    %dim0, %dim1 = mop.cpu.tensor_spec.dims.dynamic %spec : !mop.cpu.tensor_spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mop.cpu.tensor_spec<32x?xf32>
    %dim = mop.cpu.tensor_spec.dims.dynamic %spec : !mop.cpu.tensor_spec<32x?xf32>
    %dim : index

    %spec : !mop.cpu.tensor_spec<32x4xf32>
    mop.cpu.tensor_spec.dims.dynamic %spec : !mop.cpu.tensor_spec<32x4xf32> // no-op
    ```
  }];

  let arguments = (ins MOP_CpuTensorSpecType: $spec,
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

def CpuTensorSpecSizeOp : MOP_Op<"cpu.tensor_spec.size"> {
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
    %spec : !mop.cpu.tensor_spec<32x?xf32>
    %size = mop.cpu.tensor_spec.size %spec : !mop.cpu.tensor_spec<32x?xf32>
    %size : index
    ```
  }];

  let arguments = (ins MOP_CpuTensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecNumElemsOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecNumElemsOp : MOP_Op<"cpu.tensor_spec.num_elems"> {
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
    %spec : !mop.cpu.tensor_spec<32x?xf32>
    %elems = mop.cpu.tensor_spec.num_elems %spec : !mop.cpu.tensor_spec<32x?xf32>
    %elems : index
    ```
  }];

  let arguments = (ins MOP_CpuTensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDTypeOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecDTypeOp : MOP_Op<"cpu.tensor_spec.dtype"> {
  let summary = "Returns the DType of the given CPU tensor spec.";
  let description = [{
    Returns the runtime DType of the given CPU tensor spec. Generally the
    tensor spec's type will have an unknown element type (otherwise the result
    is already known).

    Example:
    ```mlir
    %spec : !mop.cpu.tensor_spec<32x?xunk>
    %dtype = mop.cpu.tensor_spec.dtype %spec : !mop.cpu.tensor_spec<32x?xunk>
    %dtype : !mop.dtype
    ```
  }];

  let arguments = (ins MOP_CpuTensorSpecType: $spec);
  let results = (outs MOP_DTypeType:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorCreateOp
//===----------------------------------------------------------------------===//

def CpuTensorCreateOp : MOP_Op<"cpu.tensor.create", [
    TypesMatchWith<"tensorSpec operand must match result tensor type spec",
                   "result", "tensorSpec",
                   "$_self.cast<CpuTensorType>().getTensorSpec()">,
    TypesMatchWith<"buffer operand must be consistent with result tensor type",
                   "result", "buffer",
                   "$_self.cast<CpuTensorType>().getTensorSpec().getExpectedBufferType()">]> {
  let summary = "Creates a CPU tensor from a tensor spec and a buffer.";
  let description = [{
    Creates a CPU tensor from a given tensor spec and a buffer. The construction
    is delayed until the given chain is ready, since the buffer is typically
    written into as a side-effect of an operation.

    Example:
    ```mlir
    %ch : !bef.chain

    %spec : !mop.cpu.tensor_spec<4x16xf32>
    %buf : !mop.cpu.buffer<256>
    %tensor = mop.cpu.tensor.create %ch %spec %buf : !mop.cpu.tensor<4x16xf32>
    %tensor : !mop.cpu.tensor<4x16xf32>

    %spec : !mop.cpu.tensor_spec<?x16xf32>
    %buf : !mop.cpu.buffer
    %tensor = mop.cpu.tensor.create %ch %spec %buf : !mop.cpu.tensor<?x16xf32>
    %tensor : !mop.cpu.tensor<?x16xf32>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    operand buffer will be copied into the tensor if it could still
    be mutated, ie if there are multiple outstanding consumers of the
    buffer's SSA value:
    ```mlir
    %buf1 = mop.cpu.buffer.alloc ...
    %ten1 = mop.cpu.tensor.create %ch ... %buf1 // copy
    ... other users of %buf1 ...

    %buf2 = mop.cpu.buffer.alloc ...
    %ten2 = mop.cpu.tensor.create %ch ... %buf2 // move
    ... no other users of %buf2 ...
    ```

    If the buffer is the result of a `mop.cpu.buffer.realloc` then it
    will be 'borrowing' rather than 'owning'. However the same rules
    apply:
    ```mlir
    %buf3 = mop.cpu.buffer.realloc ...
    %ten3 = mop.cpu.tensor.create %ch ... %buf3 // copy (as 'owning')
    ... other users of %buf3 ...

    %buf4 = mop.cpu.buffer.realloc ...
    %ten4 = mop.cpu.tensor.create %ch ... %buf4 // move (as 'borrowing')
    ... no other users of %buf4 ...
    ```
  }];

  let arguments = (ins ChainType: $chain, MOP_CpuTensorSpecType: $tensorSpec,
                       MOP_CpuBufferType: $buffer);
  let results = (outs MOP_CpuTensorType: $result);
  let assemblyFormat = [{ $chain $tensorSpec $buffer attr-dict
                          `:` qualified(type($result)) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorExtractOp
//===----------------------------------------------------------------------===//

def CpuTensorExtractOp : MOP_Op<"cpu.tensor.extract", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extracts the tensor spec and buffer of a CPU tensor.";
  let description = [{
    Returns the tensor spec and buffer components of a CPU tensor. The
    resulting buffer will have a statically known size if and only if the
    tensor spec is fully static.

    Example:
    ```mlir
    %tensor : !mop.cpu.tensor<8x64xf32>
    %spec, %buf = mop.cpu.tensor.extract %tensor : !mop.cpu.tensor<8x64xf32>
    %spec : !mop.cpu.tensor_spec<8x64>
    %buf : !mop.cpu.buffer<2048>

    %tensor : !mop.cpu.tensor<?x64xf32>
    %spec, %buf = mop.cpu.tensor.extract %tensor : !mop.cpu.tensor<?x64xf32>
    %spec : !mop.cpu.tensor_spec<?x64>
    %buf : !mop.cpu.buffer<>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    returned buffer will be a copy of the tensor buffer if the tensor
    operand could still be observed, ie if if there are multiple outstanding
    consumers of the tensor's SSA value:
    ```mlir
    %ten1 = mop.cpu.tensor.create ...
    %spec1, %buf1 = mop.cpu.tensor.extract %ten1 // copy
    ... other users of %ten1 ...

    %ten2 = mop.cpu.tensor.create ...
    %spec2, %buf2 = mop.cpu.tensor.extract %ten2 // move
    ... no other users of %ten2 ...
    ```

    If the tensor's buffer was the result of a `mop.cpu.buffer.realloc`
    then it will be 'borrowing' rather than 'owning'. However the
    same rules apply:
    ```mlir
    %buf3 = mop.cpu.buffer.realloc ...
    %ten3 = mop.cpu.tensor.create %buf3
    %spec4, %buf4 = mop.cpu.tensor.extract %ten3  // copy (as 'owning')
    ... other users of %ten3 ...

    %buf5 = mop.cpu.buffer.realloc ...  // (*)
    %ten5 = mop.cpu.tensor.create %buf5
    %spec6, %buf6 = mop.cpu.tensor.extract %ten5  // move (as 'borrowing')
    ... no other users of %ten5 ...
    ```
    Note that any mutation of %buf6 will be applied to the cached
    buffer in the context, as established by (*). However, this is
    acceptable due to the rules of `mop.cpu.buffer.realloc`.
  }];

  let arguments = (ins MOP_CpuTensorType: $tensor);
  let results = (outs MOP_CpuTensorSpecType: $tensorSpecResult,
                      MOP_CpuBufferType: $bufferResult);
  let assemblyFormat = "$tensor attr-dict `:` qualified(type($tensor))";
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

def ModelOp : MOP_Op<"model", [HasParent<"ModuleOp">, Symbol,
                               IsolatedFromAbove]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    This op creates an executable model from single-block regions
    representing its 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      () -> (!bef.chain, !mop.context)

    The 'execute' entry point has the implicit signature:
      (!bef.chain, !mop.context, in1, ..., inn) ->
      (!bef.chain, res1, ..., resm)
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

    The input and result types in the signature cannot depend on any
    framework- or mop-specific types, and must instead use only MLIR builtin
    (eg tensor), Modular (eg !M.array) and BEF (eg !bef.unknown) types.
    (Perhaps we need a !bef.anyasyncvalue type?).

    Example:
    ```mlir
    // At the top-level
    mop.model @example_model : tfl
      init {
        %ch = bef.chain.new
        %ctx = mop.context.create 0
        bef.output %ch, %ctx : !bef.chain, !mop.context
      }
      execute (%ch: !bef.chain, %ctx: !mop.context, %arg: tensor<?x4x16xf32>)
              -> (!bef.chain, tensor<?x10xf32>) {
        bef.output %ch, %arg : !bef.chain, tensor<?x4x16xf32>
      }
    ```
  }];

  // CAUTION: Attributes must be ordered alphabetically to correspond to
  // expected indexes in M::Model subclasses and attribute extraction
  // from the BEF PrimitiveDescriptor.
  let arguments = (ins ModelFrameworkAttr:$frameworkName,            // 0
                       TypeAttrOf<FunctionType>:$inputOutputTypes,   // 1
                       SymbolNameAttr:$sym_name);                    // 2
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
                   "StringAttr":$frameworkName,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$outputTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>
  ];

  let assemblyFormat = [{
    $sym_name `:` custom<ClassName>($frameworkName) attr-dict
      `init` $init
      `execute` custom<ExecuteRegion>($inputOutputTypes, $execute)
  }];
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
