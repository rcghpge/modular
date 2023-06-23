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

include "GraphRT/MEFDialect/MEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "GenericML/MGPDialect/MGPBufferAllocInterface.td"
include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.td"

// Base class for ops in this dialect.
class MGP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MGPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CpuBufferConstantOp
//===----------------------------------------------------------------------===//

def CpuBufferConstantOp : MGP_Op<"cpu.buffer.constant", [
    Pure, ConstantLike, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = [{
    Creates a buffer hosted on the CPU whose data is aliased with
    the bytes in the given attribute.
  }];
  let description = [{
    Creates a CPU-hosted buffer whose byte array is aliased with the byte array
    in the given attribute. The attribute must support contiguous iteration
    over `uint8_t` elements. The result buffer type will have a size matching
    the attribute size.

    Example valid attributes include `#M.aligned_bytes`, and the builtin
    `dense_resource' and `array`. `#M.aligned_bytes` can be used for small,
    inline byte arrays with a given alignment. `dense_resource` can be used
    for large, out-of-line byte arrays with a (optional) alignment encoded in
    the resource blob or type annotation. Any such alignment will be respected
    by the constant buffer when recovered from the MEF file. Furthermore, the
    alignment will also be respected by any deep copies of the constant buffer.

    When using a `dense_resource` with an alignment, it is necessary to
    type the attribute with `!M.aligned_bytes` so that the alignment can
    captured before the resource corresponding to the label has been
    established. Otherwise, the `dense_resource` attribute can be given any
    shaped type over `uint8_t`.

    The result buffer will 'borrow' the underlying data.

    Example:
    ```mlir
    %buf = mgp.cpu.buffer.constant #M.aligned_bytes<"0x00010203", align 8>
    %buf : !mgp.cpu.buffer<4>  // at least 8 byte aligned in storage

    %buf = mgp.cpu.buffer.constant dense_resource<label>:!M.aligned_bytes<4, align 16>
    %buf : !mgp.cpu.buffer<4>  // at least 16 byte aligned in storage
                               // the actual resource must have the same alignment

    %buf = mgp.cpu.buffer.constant #M.dense_array<ui8: 0, 1, 2, 3>
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
  let hasFolder = 1;

  let extraClassDeclaration = [{
     /// Returns the buffer's alignment.
     std::optional<uint64_t> getOptAlign();
  }];

  let builders = [
    OpBuilder<(ins "ElementsAttr":$value)>,
    OpBuilder<(ins "Type":$type, "ElementsAttr":$value)>,
  ];
}

//===----------------------------------------------------------------------===//
// CpuBufferCacheOp
//===----------------------------------------------------------------------===//

def CpuBufferCacheOp : MGP_Op<"cpu.buffer.cache"> {
  let summary = "Stores an !mgp.cpu.buffer in a dedicated state slot.";
  let description = [{
    Stores the given !mgp.cpu.buffer at buffer index 0 in a default state slot
    at the given state slot index.

    If the buffer has no other references it will be moved into the state slot.
    If the buffer is borrowed it must have at least 'execute' lifetime.
    If the buffer is mutable a deep copy will be taken and stored.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf : !mgp.buffer<256>
    %inCh : !mef.chain
    %outCh = mgp.cpu.buffer.cache %inCh, %ctx[5], %buf : !mgp.cpu.buffer<256>
    %outCh : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_CpuBufferType:$buffer,
                       UI64Attr:$slot);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $slot `]` `,` $buffer attr-dict
    `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferGetCachedOp
//===----------------------------------------------------------------------===//

def CpuBufferGetCachedOp : MGP_Op<"cpu.buffer.get_cached", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Retrieves an !mcp.cpu.buffer from a dedicated state slot.";
  let description = [{
    Retrieves the !mgp.cpu.buffer at buffer index 0 in the default state slot
    at the given state slot index. Returns an error if the retrieved CPU
    buffer's size does not agree with the size (if any) given in the
    !mgp.cpu.buffer type annotation.

    Example:
    ```mlir
    %ctx : !mgp.context
    %inCh : !mef.chain
    %buf = mgp.cpu.buffer.get_cached %inCh, %ctx[5] : !mgp.cpu.buffer<256>
    %buf : !mgp.cpu.buffer<256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aSlot,      // 0
                       UI64Attr:$bRawSize);  // 1
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `]` attr-dict
    `:` custom<RawSizeAsBufferType>($bRawSize)
  }];
}

//===----------------------------------------------------------------------===//
//
// We have a family of 6 buffer allocation primitives, depending on choice of
//  {static size vs dynamic size vs dynamic tensor spec} x {alloc vs re-alloc}
//
//  CpuBufferAllocStaticOp: Alloc with statically known size
//    mgp.cpu.buffer.alloc.static    %ch,            64
//  CpuBufferAllocDynamicOp: Alloc with dynamic size
//    mgp.cpu.buffer.alloc.dynamic   %ch,            %size
//  CpuBufferAllocSpecOp: Alloc with dynamic tensor spec
//    mgp.cpu.buffer.alloc.spec      %ch,            %spec
//  CpuBufferReallocStaticOp: Re-alloc with statically known size
//    mgp.cpu.buffer.realloc.static  %ch, %ctx[5,1], 64
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic size
//    mgp.cpu.buffer.realloc.dynamic %ch, %ctx[5,1], %size
//  CpuBufferReallocDynamicOp: Re-alloc with dynamic tensor spec
//    mgp.cpu.buffer.realloc.spec    %ch, %ctx[5,1], %spec
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CpuBufferAllocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocStaticOp : MGP_Op<"cpu.buffer.alloc.static", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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

    %buf = mgp.cpu.buffer.alloc.static %ch, 64 align 8
    %buf : !mgp.cpu.buffer<64>  // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.static %ch, 64
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
}

//===----------------------------------------------------------------------===//
// CpuBufferAllocDynamicOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocDynamicOp : MGP_Op<"cpu.buffer.alloc.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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
// CpuBufferAllocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferAllocSpecOp : MGP_Op<"cpu.buffer.alloc.spec", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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
    %ch : !mef.chain
    %spec : !mgp.cpu.tensor_spec<?x64xf32>

    %buf = mgp.cpu.buffer.alloc.spec %ch, %spec, align 8 : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer  // aligned to 8 bytes

    %buf = mgp.cpu.buffer.alloc.spec %ch, %spec : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer  // no specific alignment
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                   MGP_CpuTensorSpecType:$runtimeSpec,
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$rawAlign);
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $runtimeSpec (`,` `align` $rawAlign^)? attr-dict
    `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferReallocStaticOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocStaticOp : MGP_Op<"cpu.buffer.realloc.static", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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
         %buf1 = mgp.cpu.buffer.realloc.static %ch, %ctx[2, 0], 48, align 8
       }
       execute (...) -> ... {
         ...
         %buf2 = mgp.cpu.buffer.realloc.static %ch, %ctx[2, 0], 48, align 16
       }
    ```
    the second realloc will always reuse the buffer from the first realloc.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch : !mef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.static %ch, %ctx[42, 2], 64, align 8
    %buf : !mgp.cpu.buffer<64>   // 64 bytes aligned to 8 bytes

    %buf = mgp.cpu.buffer.realloc.static %ch, %ctx[42, 2], 64
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

def CpuBufferReallocDynamicOp : MGP_Op<"cpu.buffer.realloc.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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
    %ch : !mef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.dynamic %ch, %ctx[42, 2], %size, align 8
    %buf : !mgp.cpu.buffer      // 8 bytes aligned

    %buf = mgp.cpu.buffer.realloc.dynamic %ch, %ctx[42, 2], %size
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
// CpuBufferReallocSpecOp
//===----------------------------------------------------------------------===//

def CpuBufferReallocSpecOp : MGP_Op<"cpu.buffer.realloc.spec", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGPBufferAllocInterface, ["getOutBuffer"]>]> {
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
    %ch : !mef.chain

    // Reallocate the buffer in slot index 42 and buffer index 2.
    %buf = mgp.cpu.buffer.realloc.spec %ch, %ctx[42, 2], %spec, align 8 : !mgp.cpu.tensor_spec<?x64xf32>
    %buf : !mgp.cpu.buffer      // 8 bytes aligned

    %buf = mgp.cpu.buffer.realloc.spec %ch, %ctx[42, 2] : %spec !mgp.cpu.tensor_spec<?x64xf32>
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
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `,` $bBuffer `]` `,` $runtimeSpec
    (`,` `align` $cRawAlign^)? attr-dict `:` qualified(type($runtimeSpec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferSetWithIndex
//===----------------------------------------------------------------------===//

def CpuBufferSetWithIndexOp : MGP_Op<"cpu.buffer.set_with_index"> {
  let summary = "Set a buffer with the values from index inputs.";
  let description = [{
    The runtime index values will be sign extended or truncated based on the
    given buffer's size. Therefore only buffer sizes of 4 and 8 times the number
    of index inputs are supported by the primitive.

    Example:
    ```mlir
    %idx0 : index
    %buf0 : !mgp.cpu.buffer
    %ch0 = mgp.cpu.buffer.set_with_index %buf0, %idx0 : !mgp.cpu.buffer<4>

    %idx1 : index
    %idx2 : index
    %buf1 : !mgp.cpu.buffer
    %ch1 = mgp.cpu.buffer.set_with_index %buf1, %idx0, %idx1, %idx2 : !mgp.cpu.buffer<24>
    ```
  }];

  let arguments = (ins MGP_CpuBufferType:$buffer, Variadic<Index>:$values);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $buffer (`,` $values^)? attr-dict `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferCopyOp
//===----------------------------------------------------------------------===//

def CpuBufferCopyOp : MGP_Op<"cpu.buffer.copy", [
    AllTypesMatch<["buffer", "result"]>]> {
  let summary = "Returns a deep copy of a CPU buffer.";
  let description = [{
    This op returns a deep copy of a CPU buffer. The result will be 'owning'.
    Any alignment constraint on the operand buffer is preserved in the result.

    Generally this op should only be needed when testing uses of buffers
    which are managed by a `!mgp.context` which is created within the test
    itself.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf1 : !mgp.cpu.buffer
    %buf2 = mgp.cpu.buffer.copy %ctx, %buf1 : !mgp.cpu.buffer
    %buf2 : !mgp.cpu.buffer  // with same alignment as %buf1
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext, MGP_CpuBufferType:$buffer);
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $theContext `,` $buffer attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CpuBufferConcatOp
//===----------------------------------------------------------------------===//

def CpuBufferConcatOp : MGP_Op<"cpu.buffer.concat"> {
  let summary = "Concatenates a sequence of input buffers into an output "
                "buffer.";
  let description = [{
    Concatenates a sequence of input buffers into an output buffer. If only a
    single operand is given then mimics an ordinary buffer copy. The the output
    buffer and all input buffers have statically known sizes then they are
    checked for consistency at compile time. A runtime error will be returned
    if the actual runtime sizes of input and output buffers are not consistent.

    Example:
    ```mlir
    %in_ch : !mef.chain
    %out_buf : !mgp.cpu.buffer<256>
    %in0_buf : !mgp.cpu.buffer<128>
    %in1_buf : !mgp.cpu.buffer
    %in2_buf : !mgp.cpu.buffer
    %out_ch = mgp.cpu.buffer.concat %in_ch, %out_buf, %in0_buf, %in1_buf, %in2_buf
              : (!mgp.cpu.buffer<128>, !mgp.cpu.buffer, !mgp.cpu.buffer)
              -> !mgp.cpu.buffer<256>
    %out_ch : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_CpuBufferType:$outBuffer,
                       Variadic<MGP_CpuBufferType>:$inBuffers);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $outBuffer `,` $inBuffers attr-dict
    `:` functional-type($inBuffers, $outBuffer)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferSectionStaticBorrowOp
//===----------------------------------------------------------------------===//

def CpuBufferSectionStaticBorrowOp : MGP_Op<"cpu.buffer.section.static.borrow", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    GRTExtendsLifetimeInterface]> {
  let summary = "Borrows a section with statically known offset and size from "
                "input buffer.";
  let description = [{
    Returns a borrowed buffer who's underlying storage points into a section of
    the input buffer. The offset and size of the section is given by
    attributes, and the buffer input and result types must have statically
    known sizes. Thus this op is really only useful in conjunction with static
    memory planning. A validation error is triggered if the requested section
    is out of range of the input buffer.

    Note that no copying is done, and the result buffer merely borrows from
    the (offset into) the input buffer.

    The result buffer will inherit the mutablity and, if possible, alignment
    of the input buffer.

    The runtime will guarantee the input buffer will outlive the result buffer.
    However, currently the runtime does NOT check for overlapping concurrent
    borrows, or overlapping reads/writes to the same contents, even in debug
    builds.

    Example:
    ```mlir
    %ch:  !mef.chain
    %in_buf : !mgp.cpu.buffer<256>
    %out_buf =
        mgp.cpu.buffer.section.static.borrow %ch, %in_buf, offset 128, size 64
        : !mgp.cpu.buffer<256>
    %out_buf : !mgp.cpu.buffer<64>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_CpuBufferType:$inBuffer,
                       UI64Attr:$aOffset,
                       UI64Attr:$bSize);
  let results = (outs MGP_CpuBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $inBuffer `,` `offset` $aOffset `,` `size` $bSize attr-dict
    `:` qualified(type($inBuffer))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuBufferSplitBorrowOp
//===----------------------------------------------------------------------===//

def CpuBufferSplitBorrowOp : MGP_Op<"cpu.buffer.split.borrow", [
    GRTExtendsLifetimeInterface]> {
  let summary = "Splits an input buffer into multiple result buffers which "
                "borrow sections from the input.";
  let description = [{
    Splits an input buffer into multiple result buffers which borrow sections
    from the input. An `mgp.cpu.buffer.concat` of the result buffers would
    recover the contents of the input buffer. The sizes of each result buffer
    can be supplied either implicitly by the result buffer type's size, or
    explicitly as an index operand if the result buffer type is unsized.

    A runtime error will be returned if the actual runtime sizes of input and
    result buffers and size operands are not consistent.

    Note that no copying is done, and the result buffers merely borrow from the
    (offsets into) the input buffer.

    The result buffers will inherit the mutablity and, if possible, alignment
    of the input buffer.

    The runtime will guarantee the input buffer will outlive the result
    buffers. However, currently the runtime does NOT check for overlapping
    reads/writes to the same contents, even in debug builds.

    Example:
    ```mlir
    %ch:  !mef.chain
    %in_buf : !mgp.cpu.buffer<256>
    %size1 : index
    %size2 : index
    %out0_buf, %out1_buf, %out2_buf =
        mgp.cpu.buffer.split.borrow %ch, %in_buf, %size1, %size2
        : !mgp.cpu.buffer<256> -> (!mgp.cpu.buffer<128>, !mgp.cpu.buffer, !mgp.cpu.buffer)
    %out0_buf : !mgp.cpu.buffer<128>
    %out1_buf : !mgp.cpu.buffer
    %out2_buf : !mgp.cpu.buffer
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_CpuBufferType:$inBuffer,
                       Variadic<Index>:$runtimeSizes,
                       UI64ArrayElementsAttr:$outBufferRawSizes);
  let results = (outs Variadic<MGP_CpuBufferType>:$outBuffers);
  let assemblyFormat = [{
    $inChain `,` $inBuffer (`,` $runtimeSizes^)? attr-dict
    `:` qualified(type($inBuffer))
        custom<BufferTypesAndRawSizes>(type($outBuffers), $outBufferRawSizes)
  }];
  let hasVerifier = 1;
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
    %buf : !mgp.cpu.buffer

    // Set each byte of the buffer to 0.
    %ch = mgp.cpu.buffer.set %buf : !mgp.cpu.buffer
    // Set each byte of the buffer to 1.
    %ch = mgp.cpu.buffer.set %buf, 1 : !mgp.cpu.buffer
    ```
  }];

  let arguments = (ins MGP_CpuBufferType:$buffer,
                   DefaultValuedAttr<UI8Attr, "0">:$value);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $buffer (`,` $value^)? attr-dict `:` qualified(type($buffer))
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

    %spec = mgp.cpu.tensor_spec.create %dim0, %dim1 : !mgp.cpu.tensor_spec<?x?xf32>
    %spec : !mgp.cpu.tensor_spec<?x?xf32>

    %spec = mgp.cpu.tensor_spec.create : !mgp.cpu.tensor_spec<32x4xf32>
    %spec : !mgp.cpu.tensor_spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by MEF.
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

    %spec = mgp.cpu.tensor_spec.create.dynamic %dtype, %dim0, %dim1 : !mgp.cpu.tensor_spec<?x?x?>
    %spec : !mgp.cpu.tensor_spec<?x?x?>

    %spec = mgp.cpu.tensor_spec.create %dtype : !mgp.cpu.tensor_spec<32x4x?>
    %spec : !mgp.cpu.tensor_spec<32x4x?>
    ```
  }];

  // NOTE: The attribute effectively encodes the shape of the result's tensor
  // spec type in a form understandable by MEF.
  let arguments = (ins MGP_DTypeType:$runtimeDType,
                   Variadic<Index>:$runtimeDims,
                   UI64ArrayElementsAttr:$rawDims);
  let results = (outs MGP_CpuTensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDType `,` $runtimeDims attr-dict
                          `:` custom<RawDims>($rawDims) }];
  let hasVerifier = 1;
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

// TODO: Look into representing dynamic buffers more explicitly.
// E.g. !mgp.cpu.buffer<?> instead of just !mgp.cpu.buffer.
def CpuTensorCreateOp : MGP_Op<"cpu.tensor.create", [
    TypesMatchWith<"tensorSpec operand must match result tensor type spec",
                   "result", "tensorSpec",
                   "$_self.cast<CpuTensorType>().getTensorSpec()">]> {
  let summary = "Creates a CPU tensor from a tensor spec and a buffer.";
  let description = [{
    Creates a CPU tensor from a given tensor spec and a buffer. The construction
    is delayed until the given chain is ready, since the buffer is typically
    written into as a side-effect of an operation.

    If the result tensor type is fully static then the buffer is expected to
    have a known and matching size. However if the actual buffer type cannot be
    correctly inferred from the spec types, the user must annotate this op with
    the correct buffer type.

    Example:
    ```mlir
    %ch : !mef.chain

    %spec : !mgp.cpu.tensor_spec<4x16xf32>
    %buf : !mgp.cpu.buffer<256>
    %tensor = mgp.cpu.tensor.create %ch, %buf, %spec : !mgp.cpu.tensor<4x16xf32>
    %tensor : !mgp.cpu.tensor<4x16xf32>

    %spec : !mgp.cpu.tensor_spec<?x16xf32>
    %buf : !mgp.cpu.buffer
    %tensor = mgp.cpu.tensor.create %ch, %buf, %spec : !mgp.cpu.tensor<?x16xf32>
    %tensor : !mgp.cpu.tensor<?x16xf32>

    %spec : !mgp.cpu.tensor_spec<4x16xf32>
    %buf : !mgp.cpu.buffer
    %tensor = mgp.cpu.tensor.create %ch, %buf, %spec !mgp.cpu.buffer : !mgp.cpu.tensor<4x16xf32>
    %tensor : !mgp.cpu.tensor<4x16xf32>

    %spec : !mgp.cpu.tensor_spec<4x?xf32>
    %buf : !mgp.cpu.buffer<16>
    %tensor = mgp.cpu.tensor.create %ch, %buf, %spec !mgp.cpu.buffer<16> : !mgp.cpu.tensor<4x?xf32>
    %tensor : !mgp.cpu.tensor<4x?xf32>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    operand buffer will be copied into the tensor if it could still
    be mutated, ie if there are multiple outstanding consumers of the
    buffer's SSA value:
    ```mlir
    %buf1 = mgp.cpu.buffer.alloc ...
    %ten1 = mgp.cpu.tensor.create %ch, %buf1, ... // copy
    ... other users of %buf1 ...

    %buf2 = mgp.cpu.buffer.alloc ...
    %ten2 = mgp.cpu.tensor.create %ch, %buf2, ... // move
    ... no other users of %buf2 ...
    ```

    If the buffer is the result of a `mgp.cpu.buffer.realloc` then it
    will be 'borrowing' rather than 'owning'. However the same rules
    apply:
    ```mlir
    %buf3 = mgp.cpu.buffer.realloc ...
    %ten3 = mgp.cpu.tensor.create %ch, ... %buf3 // copy (as 'owning')
    ... other users of %buf3 ...

    %buf4 = mgp.cpu.buffer.realloc ...
    %ten4 = mgp.cpu.tensor.create %ch, ... %buf4 // move (as 'borrowing')
    ... no other users of %buf4 ...
    ```
  }];

  let arguments = (ins ChainType: $inChain,
                   MGP_CpuBufferType: $buffer,
                   MGP_CpuTensorSpecType: $tensorSpec);
  let results = (outs MGP_CpuTensorType: $result);
  let assemblyFormat = [{
    $inChain `,` $buffer `,` $tensorSpec attr-dict
    custom<MayBeDynamicBuffer>(type($buffer), type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CpuTensorExtractOp
//===----------------------------------------------------------------------===//

def CpuTensorExtractOp : MGP_Op<"cpu.tensor.extract", [
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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
// DebugCpuTensorPrintOp
//===----------------------------------------------------------------------===//

def DebugCpuTensorPrintOp : MGP_Op<"debug.cpu.tensor.print"> {
  let summary = "Prints a debug representation of argument tensor. For "
                "debugging and testing only.";
  let description = [{
    Prints a debug representation of argument tensor. For lit unit tests.

    Example:
    ```mlir
    %tensor : !mgp.cpu.tensor<8x64xf32>
    %ch0 : !mef.chain
    %ch1 = mgp.debug.cpu.tensor.print %ch0, %tensor : !mgp.cpu.tensor<8x64xf32>
    %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain, MGP_CpuTensorType: $tensor);
  let results = (outs ChainType: $outChain);
  let assemblyFormat = "$inChain `,` $tensor attr-dict `:` qualified(type($tensor))";
}

//===----------------------------------------------------------------------===//
// DebugCpuBufferPrintOp
//===----------------------------------------------------------------------===//

def DebugCpuBufferPrintOp : MGP_Op<"debug.cpu.buffer.print"> {
  let summary = "Prints a debug representation of argument buffer. For "
                "debugging and testing only.";
  let description = [{
    Prints a debug representation of argument buffer. For lit unit tests.

    Example:
    ```mlir
    %buf : !mgp.cpu.buffer<8>
    %ch0 : !mef.chain
    %ch1 = mgp.debug.cpu.buffer.print %ch0, %buf : !mgp.cpu.buffer<8>
    %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins ChainType:$inChain, MGP_CpuBufferType: $buffer);
  let results = (outs ChainType: $outChain);
  let assemblyFormat = "$inChain `,` $buffer attr-dict `:` qualified(type($buffer))";
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
// CASCreateInstanceOp
//===----------------------------------------------------------------------===//

def CASCreateInstanceOp : MGP_Op<"cas.create_instance"> {
  let summary = "Creates a CAS instance at given cache directory.";
  let description = [{
    Creates a CAS instance at the given cache directory, and stores it at
    the given state slot index. The same index should be used by
    mgp.cas.lookup and mgp.cas.store.

    Example:
    ```mlir
    %ctx : !mgp.context
    %inCh : !mef.chain
    %outCh = mgp.cas.create_instance %inCh, %ctx[0] at "/path/to/cache"
    %outCh : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aSlot,     // 0
                       StrAttr:$bCacheDir); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `]` `at` $bCacheDir attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CASLookupOp
//===----------------------------------------------------------------------===//

def CASLookupOp : MGP_Op<"cas.lookup", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Lookup an !mgp.cpu.buffer from the CAS instance at the given "
                "slot under the given key.";
  let description = [{
    Lookup an !mgp.cpu.buffer from the CAS instance at the given slot under
    the given key.  Returns an error if the retrieved CPU buffer's size does
    not agree with the size (if any) given in the !mgp.cpu.buffer type
    annotation.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch : !mef.chain
    %buf = mgp.cas.lookup %ch, %ctx[0], "SOMECASID" { dRawAlign=64 : ui64 } : !mgp.cpu.buffer<256>
    %buf : !mgp.cpu.buffer<256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aSlot,                                         // 0
                       StrAttr:$bKey,                                           // 1
                       UI64Attr:$cRawSize,                                      // 2
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$dRawAlign); // 3
  let results = (outs MGP_CpuBufferType:$result);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `]` `,` custom<Base64>($bKey) attr-dict
    `:` custom<RawSizeAsBufferType>($cRawSize)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CASStoreOp
//===----------------------------------------------------------------------===//

def CASStoreOp : MGP_Op<"cas.store"> {
  let summary = "Insert a copy of an !mgp.cpu.buffer in the CAS instance at the "
                "given slot under the given key.";
  let description = [{
    Insert a copy of an !mgp.cpu.buffer in the CAS instance at the given slot
    under the given key.

    This is provided for writing unit tests. The given buffer will always be
    copied into the CAS instance.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf : !mgp.cpu.buffer<256>
    %inCh : !mef.chain
    %outCh = mgp.cas.store %inCh, %ctx[0], "SOMECASID", %buf : !mgp.cpu.buffer<256>
    %outCh : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_CpuBufferType:$buffer,
                       UI64Attr:$aSlot,   // 0
                       StrAttr:$bKey);    // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aSlot `]` `,` custom<Base64>($bKey)
    `,` $buffer attr-dict `:` qualified(type($buffer))
  }];
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

def ToGRTBoolOp : MGP_Op<"to_grt_bool"> {
  let summary = "Convert a single byte Buffer into a builtin i1.";
  let description = [{
    Convert a single byte Buffer into a builtin i1.

    Example:
    ```mlir
    %0 = mgp.to_grt_bool(%ch0, %buf) : (!mef.chain, !mgp.cpu.buffer<1>) -> i1
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_ScalarBuffer:$arg);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : MGP_Op<"assert"> {
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

    The input and result types in the signature cannot depend on any
    framework- or mgp-specific types, and must instead use only MLIR builtin
    (eg tensor), Modular (eg !M.array) and MEF (eg !mef.unknown) types.
    (Perhaps we need a !mef.anyasyncvalue type?).

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

#endif  // GENERICML_MGPDIALECT_MGPOPS_TD
