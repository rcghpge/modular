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

include "GraphRT/BEFDialect/BEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypes.td"

// Base class for ops in this dialect.
class MGP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MGPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CpuBufferConstantOp
//===----------------------------------------------------------------------===//

def CpuBufferConstantOp : MGP_Op<"cpu.buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a buffer hosted on the CPU whose data is aliased with
    the bytes in the given attribute.
  }];
  let description = [{
    Creates a CPU-hosted buffer whose byte array is aliased with the byte array
    in the given attribute. The attribute must support contiguous iteration
    over `uint8_t` elements. The result buffer type will have a size matching
    the attribute size.

    Example valid attributes include `#m.aligned_bytes`, and the builtin
    `dense_resource' and `array`. `#m.aligned_bytes` can be used for small,
    inline byte arrays with a given (optional) alignment. `dense_resource`
    can be used for large, out-of-line byte arrays with a (optional) alignment
    encoded in the resource blob. Any such alignment will be respected
    by the constant buffer when recovered from the BEF file. Furthermore, the
    alignment will also be respected by any deep copies of the constant
    buffer.

    When using a `dense_resource` with an alignment, it is necessary to
    type the attribute with `mgp.aligned_bytes` so that the alignment can
    captured before the resource corresponding to the label has been
    established. Otherwise, the `dense_resource` attribute can be given any
    shaped type over `uint8_t`.

    The result buffer will 'borrow' the underlying data.

    Example:
    ```mlir
    %buf = mgp.cpu.buffer.constant #m.aligned_buffer<8: "0x00010203">
    %buf : !mgp.cpu.buffer<4>  // at least 8 byte aligned in storage

    %buf = mgp.cpu.buffer.constant dense_resource<label>:!mgp.aligned_bytes<4, 16>
    %buf : !mgp.cpu.buffer<4>  // at least 16 byte aligned in storage
                               // the actual resource must have the same alignment

    %buf = mgp.cpu.buffer.constant array<ui8: 0, 1, 2, 3>
    %buf : !mgp.cpu.buffer<4>  // no specific alignment in storage
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ElementsAttr:$aValue, UI64Attr:$bRawAlign);
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    custom<ElementsAttrWithAlign>($aValue, $bRawAlign) attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
     /// Returns the buffer's alignment.
     Optional<uint64_t> getOptAlign();
  }];

  let builders = [
    OpBuilder<(ins "ElementsAttr":$value)>
  ];
}


//===----------------------------------------------------------------------===//
//
// We have a family of 6 buffer allocation primitives, depending on choice of
//  {static size vs dynamic size vs dynamic tensor spec} x {alloc vs re-alloc}
//
//  CpuBufferAllocStaticOp: Alloc with statically known size
//    mgp.cpu.buffer.alloc.static        %ctx      64
//  CpuBufferAllocDynamicOp: Alloc with dynamic size
//    mgp.cpu.buffer.alloc.dynamic       %ctx      %size
//  CpuBufferAllocSpecOp: Alloc with dynamic tensor spec
//    mgp.cpu.buffer.alloc.spec          %ctxt     %spec
//  CpuBufferReallocStaticOp: Re-alloc with statically known size
//    mgp.cpu.buffer.realloc.static  %ch %ctx[5,1] 64
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic size
//    mgp.cpu.buffer.realloc.dynamic %ch %ctx[5,1] %size
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic tensor spec
//    mgp.cpu.buffer.realloc.spec    %ch %ctx[5,1] %spec
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CpuBufferAllocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocStaticOp : MGP_Op<"cpu.buffer.alloc.static", [
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
    reclaimed once the last reference to the result goes out of scope.

    Example:
    ```mlir
    %ctx : !mgp.context

    %buf = mgp.cpu.buffer.alloc.static %ctx 64 align 8
    %buf : !mgp.cpu.buffer<64>  // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.static %ctx 64
    %buf : !mgp.cpu.buffer<64>  // 64 bytes with no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_ContextType:$theContext,
                       UI64Attr:$aSize,                                         // 0
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign); // 1
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $aSize (`align` $bRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocDynamicOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocDynamicOp : MGP_Op<"cpu.buffer.alloc.dynamic", [
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
    %ctx : !mgp.context
    %size : index

    %buf = mgp.cpu.buffer.alloc.dynamic %ctx %size align 8
    %buf : !mgp.cpu.buffer  // aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.dynamic %ctx %size
    %buf : !mgp.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext,
                       Index:$runtimeSize,
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $runtimeSize (`align` $rawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocSpecOp : MGP_Op<"cpu.buffer.alloc.spec", [
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
    %ctx : !mgp.context
    %spec : !mgp.tensor_spec<?x64xf32>

    %buf = mgp.cpu.buffer.alloc.spec %ctx %spec align 8 : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer  // aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.spec %ctx %spec : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext,
                       MGP_CpuTensorSpecType:$runtimeSpec,
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $runtimeSpec (`align` $rawAlign^)? attr-dict
    `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocStaticOp : MGP_Op<"cpu.buffer.realloc.static", [
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
         %buf1 = mgp.cpu.buffer.realloc.static %ch %ctx[2, 0] 48 align 8
       }
       execute (...) -> ... {
         ...
         %buf2 = mgp.cpu.buffer.realloc.static %ch %ctx[2, 0] 48 align 16
       }
    ```
    the second realloc will always reuse the buffer from the first realloc.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.static %ch %ctx[42, 2] 64 align 8
    %buf : !mgp.cpu.buffer<64>   // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.realloc.static %ch %ctx[42, 2] 64
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
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $inChain $theContext `[` $aSlot `,` $bBuffer `]` $cSize
    (`align` $dRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocDynamicOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocDynamicOp : MGP_Op<"cpu.buffer.realloc.dynamic", [
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
    %ctx : !mgp.context
    %size : index
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.dynamic %ch %ctx[42, 2] %size align 8
    %buf : !mgp.cpu.buffer      // 8 bytes aligned

    %buf = mgp.cpu.buffer.realloc.dynamic %ch %ctx[42, 2] %size
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
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $inChain $theContext `[` $aSlot `,` $bBuffer `]` $runtimeSize
    (`align` $cRawAlign^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocSpecOp : MGP_Op<"cpu.buffer.realloc.spec", [
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
    %ctx : !mgp.context
    %spec : !mgp.cpu.tensor_spec<?x64xf32>
    %ch : !bef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.spec %ch %ctx[42, 2] %spec align 8 : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer      // 8 bytes aligned

    %buf = mgp.cpu.buffer.realloc.spec %ch %ctx[42, 2] : %spec !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer<64>  // no specific alignment
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_CpuTensorSpecType:$runtimeSpec,
                       UI64Attr:$aSlot,                                         // 0
                       UI64Attr:$bBuffer,                                       // 1
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$cRawAlign); // 3
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $inChain $theContext `[` $aSlot `,` $bBuffer `]` $runtimeSpec
    (`align` $cRawAlign^)? attr-dict `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferCopyOp
//===----------------------------------------------------------------------===//

def CpuBufferCopyOp : MGP_Op<"cpu.buffer.copy", [
    AllTypesMatch<["buffer", "result"]>]> {
  let summary = "Copies the contents of a CPU buffer.";
  let description = [{
    This op returns a copy of a CPU buffer. The result will be 'owning'.

    Generally this op should only be needed when testing the use
    of `mgp.cpu.buffer.realloc` using a `!mgp.context` which is created
    within the test itself. By rule 2 of `mgp.cpu.buffer.realloc` the
    result of `mgp.cpu.buffer.realloc` cannot outlive the context itself,
    so a copy is needed to return that buffer to the test harness
    for printing etc. It takes `!mgp.context` as an argument to
    make sure that the context survives until the copy is performed.

    Any alignment constraint on the operand buffer is preserved in
    the result.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf1 : !mgp.cpu.buffer
    %buf2 = mgp.cpu.buffer.copy %ctx %buf1 : !mgp.cpu.buffer
    %buf2 : !mgp.cpu.buffer  // with same alignment as %buf1
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext, MGP_CpuBufferType:$buffer);
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext $buffer attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferSetOp
//===----------------------------------------------------------------------===//

def CpuBufferSetOp : MGP_Op<"cpu.buffer.set", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Set each byte of the given buffer to the given value (default = 0).
  }];
  let description = [{
    Sets each byte of the allocated buffer to the given value (default = 0).

    Example:
    ```mlir
    %buf : !mgp.cpu.buffer<>

    // Set each byte of the buffer to 0.
    %ch = mgp.cpu.buffer.set %buf : !mgp.cpu.buffer<>
    // Set each byte of the buffer to 1.
    %ch = mgp.cpu.buffer.set %buf 1 : !mgp.cpu.buffer<>
    ```
  }];

  let arguments = (ins MGP_CpuBufferType:$buffer,
                       DefaultValuedAttr<UI8Attr, "0">:$value);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $buffer ($value^)? attr-dict `:` qualified(type($buffer))
  }];
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
    %ch0 : !bef.chain

    // Creates a default state slot at index 42 with room for
    // two buffers.
    %ch1 = mgp.state_slot.create %ch0 %ctx[42] 2
    %ch1 : !bef.chain
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
    $inChain $theContext `[` $slot `]` $numBuffers attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecCreateOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecCreateOp : MGP_Op<"cpu.tensor_spec.create", [
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

    %spec = mgp.cpu.tensor_spec.create %dim0 %dim1 : !mgp.cpu.tensor_spec<?x?xf32>
    %spec : !mgp.cpu.tensor_spec<?x?xf32>

    %spec = mgp.cpu.tensor_spec.create : !mgp.cpu.tensor_spec<32x4xf32>
    %spec : !mgp.cpu.tensor_spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by BEF.
  let arguments = (ins Variadic<Index>:$runtimeDims,
                       UI64ArrayElementsAttr:$aRawDims,  // 0
                       UI8Attr:$bRawDType);              // 1
  let results = (outs MGP_CpuTensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims attr-dict
    `:` custom<RawDimsAndDType>($aRawDims, $bRawDType)
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "CpuTensorSpecType":$spec,
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

    %spec = mgp.cpu.tensor_spec.create.dynamic %dtype %dim0, %dim1 : !mgp.cpu.tensor_spec<?x?x?>
    %spec : !mgp.cpu.tensor_spec<?x?x?>

    %spec = mgp.cpu.tensor_spec.create %dtype : !mgp.cpu.tensor_spec<32x4x?>
    %spec : !mgp.cpu.tensor_spec<32x4x?>
    ```
  }];

  // NOTE: The attribute effectively encodes the shape of the result's tensor
  // spec type in a form understandable by BEF.
  let arguments = (ins MGP_DTypeType:$runtimeDType, Variadic<Index>:$runtimeDims,
                       UI64ArrayElementsAttr:$rawDims);
  let results = (outs MGP_CpuTensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDType $runtimeDims attr-dict
                          `:` custom<RawDims>($rawDims) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecDimsDynamicOp
//===----------------------------------------------------------------------===//

def CpuTensorSpecDimsDynamicOp : MGP_Op<"cpu.tensor_spec.dims.dynamic", [
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
    %spec : !mgp.cpu.tensor_spec<?x?xf32>
    %dim0, %dim1 = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %dim = mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %dim : index

    %spec : !mgp.cpu.tensor_spec<32x4xf32>
    mgp.cpu.tensor_spec.dims.dynamic %spec : !mgp.cpu.tensor_spec<32x4xf32> // no-op
    ```
  }];

  let arguments = (ins MGP_CpuTensorSpecType: $spec,
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
    %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %size = mgp.cpu.tensor_spec.size %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %size : index
    ```
  }];

  let arguments = (ins MGP_CpuTensorSpecType: $spec);
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
    %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %elems = mgp.cpu.tensor_spec.num_elems %spec : !mgp.cpu.tensor_spec<32x?xf32>
    %elems : index
    ```
  }];

  let arguments = (ins MGP_CpuTensorSpecType: $spec);
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
    %spec : !mgp.cpu.tensor_spec<32x?xunk>
    %dtype = mgp.cpu.tensor_spec.dtype %spec : !mgp.cpu.tensor_spec<32x?xunk>
    %dtype : !mgp.dtype
    ```
  }];

  let arguments = (ins MGP_CpuTensorSpecType: $spec);
  let results = (outs MGP_DTypeType:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// CpuTensorCreateOp
//===----------------------------------------------------------------------===//

def CpuTensorCreateOp : MGP_Op<"cpu.tensor.create", [
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

    %spec : !mgp.cpu.tensor_spec<4x16xf32>
    %buf : !mgp.cpu.buffer<256>
    %tensor = mgp.cpu.tensor.create %ch %buf %spec : !mgp.cpu.tensor<4x16xf32>
    %tensor : !mgp.cpu.tensor<4x16xf32>

    %spec : !mgp.cpu.tensor_spec<?x16xf32>
    %buf : !mgp.cpu.buffer
    %tensor = mgp.cpu.tensor.create %ch %buf %spec : !mgp.cpu.tensor<?x16xf32>
    %tensor : !mgp.cpu.tensor<?x16xf32>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    operand buffer will be copied into the tensor if it could still
    be mutated, ie if there are multiple outstanding consumers of the
    buffer's SSA value:
    ```mlir
    %buf1 = mgp.cpu.buffer.alloc ...
    %ten1 = mgp.cpu.tensor.create %ch %buf1 ... // copy
    ... other users of %buf1 ...

    %buf2 = mgp.cpu.buffer.alloc ...
    %ten2 = mgp.cpu.tensor.create %ch %buf2 ... // move
    ... no other users of %buf2 ...
    ```

    If the buffer is the result of a `mgp.cpu.buffer.realloc` then it
    will be 'borrowing' rather than 'owning'. However the same rules
    apply:
    ```mlir
    %buf3 = mgp.cpu.buffer.realloc ...
    %ten3 = mgp.cpu.tensor.create %ch ... %buf3 // copy (as 'owning')
    ... other users of %buf3 ...

    %buf4 = mgp.cpu.buffer.realloc ...
    %ten4 = mgp.cpu.tensor.create %ch ... %buf4 // move (as 'borrowing')
    ... no other users of %buf4 ...
    ```
  }];

  let arguments = (ins ChainType: $inChain,
                       MGP_CpuBufferType: $buffer,
                       MGP_CpuTensorSpecType: $tensorSpec);
  let results = (outs MGP_CpuTensorType: $result);
  let assemblyFormat = [{
    $inChain $buffer $tensorSpec attr-dict `:` qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorExtractOp
//===----------------------------------------------------------------------===//

def CpuTensorExtractOp : MGP_Op<"cpu.tensor.extract", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extracts the tensor spec and buffer of a CPU tensor.";
  let description = [{
    Returns the tensor spec and buffer components of a CPU tensor. The
    resulting buffer will have a statically known size if and only if the
    tensor spec is fully static.

    Example:
    ```mlir
    %tensor : !mgp.cpu.tensor<8x64xf32>
    %buf, %spec = mgp.cpu.tensor.extract %tensor : !mgp.cpu.tensor<8x64xf32>
    %buf : !mgp.cpu.buffer<2048>
    %spec : !mgp.cpu.tensor_spec<8x64>

    %tensor : !mgp.cpu.tensor<?x64xf32>
    %buf, %spec = mgp.cpu.tensor.extract %tensor : !mgp.cpu.tensor<?x64xf32>
    %buf : !mgp.cpu.buffer<>
    %spec : !mgp.cpu.tensor_spec<?x64>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    returned buffer will be a copy of the tensor buffer if the tensor
    operand could still be observed, ie if if there are multiple outstanding
    consumers of the tensor's SSA value:
    ```mlir
    %ten1 = mgp.cpu.tensor.create ...
    %buf1, %spec1 = mgp.cpu.tensor.extract %ten1 // copy
    ... other users of %ten1 ...

    %ten2 = mgp.cpu.tensor.create ...
    %buf2, %spec2 = mgp.cpu.tensor.extract %ten2 // move
    ... no other users of %ten2 ...
    ```

    If the tensor's buffer was the result of a `mgp.cpu.buffer.realloc`
    then it will be 'borrowing' rather than 'owning'. However the
    same rules apply:
    ```mlir
    %buf3 = mgp.cpu.buffer.realloc ...
    %ten3 = mgp.cpu.tensor.create %buf3
    %buf4, %spec4 = mgp.cpu.tensor.extract %ten3  // copy (as 'owning')
    ... other users of %ten3 ...

    %buf5 = mgp.cpu.buffer.realloc ...  // (*)
    %ten5 = mgp.cpu.tensor.create %buf5
    %buf6, %spec6 = mgp.cpu.tensor.extract %ten5  // move (as 'borrowing')
    ... no other users of %ten5 ...
    ```
    Note that any mutation of %buf6 will be applied to the cached
    buffer in the context, as established by (*). However, this is
    acceptable due to the rules of `mgp.cpu.buffer.realloc`.
  }];

  let arguments = (ins MGP_CpuTensorType: $tensor);
  let results = (outs MGP_CpuBufferType: $bufferResult,
                      MGP_CpuTensorSpecType: $tensorSpecResult);
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

def ModelOp : MGP_Op<"model", [HasParent<"ModuleOp">, Symbol,
                               IsolatedFromAbove]> {
  let summary = [{
    Represents an executable model with 'init' and 'execute' entry points.
  }];
  let description = [{
    This op creates an executable model from single-block regions
    representing its 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      () -> (!bef.chain, !mgp.context)

    The 'execute' entry point has the implicit signature:
      (!bef.chain, !mgp.context, in1, ..., inn) ->
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
    framework- or mgp-specific types, and must instead use only MLIR builtin
    (eg tensor), Modular (eg !M.array) and BEF (eg !bef.unknown) types.
    (Perhaps we need a !bef.anyasyncvalue type?).

    Example:
    ```mlir
    // At the top-level
    mgp.model @example_model : tfl
      init {
        %ch = bef.chain.new
        %ctx = mgp.context.create 0
        bef.output %ch, %ctx : !bef.chain, !mgp.context
      }
      execute (%ch: !bef.chain, %ctx: !mgp.context, %arg: tensor<?x4x16xf32>)
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

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
