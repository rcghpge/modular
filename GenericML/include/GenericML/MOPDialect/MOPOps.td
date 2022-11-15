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

// TODO(#4728): The syntax for slot indexes could be greatly improved, eg "#42".

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MOP_Op<"buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Creates a buffer whose data is aliased with the bytes in the "
                "given attribute.";
  let description = [{
    Creates a buffer whose byte array is aliased with the byte array
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
    %buf = mop.buffer.constant #m.aligned_buffer<8: "0x00010203">
    %buf : !mop.buffer<4>  // at least 8 byte aligned in storage

    %buf = mop.buffer.constant dense_resource<label>:!m.array<4xui8>
    %buf : !mop.buffer<4>  // respecting alignment encoded in blob at label

    %buf = mop.buffer.constant array<ui8: 0, 1, 2, 3>
    %buf : !mop.buffer<4>  // no specific alignment in storage
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the align constraint for the buffer, if any.
    Optional<uint64_t> getAlign();
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//

def BufferAllocOp : MOP_Op<"buffer.alloc", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates an uninitialized buffer of given runtime or static "
                "size and optional alignment.";
  let description = [{
    Allocates an uninitialized buffer whose byte array is of a given size. The
    size can be supplied either as a runtime argument or as an attribute. Any
    alignment constraint must be supplied as an attribute.

    Note that any align constraint is not propagated into the result type.

    The result buffer will be 'owning', and the underlying memory will be
    reclaimed once the last reference to the result goes out of scope. In
    debug builds the buffer will also be marked 'mutable'.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index

    %buf = mop.buffer.alloc %ctxt %size {align=8:ui64}
    %buf : !mop.buffer<>  // at least 8 byte aligned in storage

    %buf = mop.buffer.alloc %ctxt {size=64:ui64}
    %buf : !mop.buffer<64>  // no specific alignment in storage

    %buf = mop.buffer.alloc %ctxt {size=64:ui64, align=16:ui64}
    %buf : !mop.buffer<64>  // at least 16 byte aligned in storage
    ```
  }];

  // TODO(#4301): Would be nicer to use OptionalAttr for $size and $align but
  // MLIRToBEF cannot handle that.
  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MOP_ContextType:$theContext, Optional<Index>:$runtimeSize,
                       DefaultValuedAttr<UI64Attr, "UINT64_MAX">:$align,  // 0
                       DefaultValuedAttr<UI64Attr, "UINT64_MAX">:$size);  // 1
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$theContext ($runtimeSize^)? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferReallocOp
//===----------------------------------------------------------------------===//

def BufferReallocOp : MOP_Op<"buffer.realloc", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates, reallocates or reuses a buffer of given runtime or "
                "static size and optional alignment.";
  let description = [{
    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer.

    If the size is known statically then the runtime size operand does not need
    to be supplied as an operand. This can be used to pre-allocate a buffer
    of statically known size.

    This op accepts a chain argument so that access to the slot state can
    be guaranteed to come after construction and initialization of the
    state slot at the slot index.

    Though the cache entry will be an 'owning' reference to the buffer,
    the result of this op is always a 'borrowing' reference to the same
    underlying memory. (In debug builds both refs will also be marked
    'mutable'). The compiler must respect the following rules to avoid
    use-after-free errors and mutation-via-aliases at runtime:
     1. For all control flow paths, at most one mop.buffer.realloc can
        execute for the same slot and buffer indexes.
        This rule is easy to follow since no two operations in the graph should
        share the same slot, and buffer indexes should correspond to op result
        indexes.
     2. The lifetime of the context must exceed the lifetime of the returned
        buffer. This rule is easy to follow since the context is held by the
        Modular runtime M::ExecutableModel, which must outlive both all calls
        to execute and all uses of aliasing results of execute.

    Note that though the buffer will be cached, the contents of the buffer need
    not be preserved between calls to `mop.buffer.realloc` in successive
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
         %buf1 = mop.buffer.realloc %ch %ctxt { size=48, align=8, slot=2, buffer=0 }
       }
       execute (...) -> ... {
         ...
         %buf2 = mop.buffer.realloc %ch %ctxt { size=48, align=16, slot=2, buffer=0 }
       }
    ```
    the second realloc will always reuse the buffer from the first realloc.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index
    %ch : !bef.chain

    %buf = mop.buffer.realloc %ch %ctxt %size { align=8:ui64, slot=42, buffer=2 }
    %buf : !mop.buffer      // at least 8 byte aligned in storage

    %buf = mop.buffer.realloc %ch %ctxt { size=64:ui64, slot=42, buffer=2 }
    %buf : !mop.buffer<64>  // no specific alignment in storage
    ```
  }];

  // TODO(#4301): Would be nicer to use OptionalAttr for $size and $align but
  // MLIRToBEF cannot handle that.
  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$ch,
                       MOP_ContextType:$theContext,
                       Optional<Index>:$runtimeSize,
                       DefaultValuedAttr<UI64Attr, "UINT64_MAX">:$align, // 0
                       UI64Attr:$buffer,                                 // 1
                       DefaultValuedAttr<UI64Attr, "UINT64_MAX">:$size,  // 2
                       UI64Attr:$slot);                                  // 3
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$ch $theContext ($runtimeSize^)? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferCopyOp
//===----------------------------------------------------------------------===//

def BufferCopyOp : MOP_Op<"buffer.copy", [
    AllTypesMatch<["buffer", "result"]>]> {
  let summary = "Copies the contents of a buffer.";
  let description = [{
    This op returns a copy of a buffer. The result will be 'owning'.
    In debug builds the result will also be 'mutable'.

    Generally this op should only be needed when testing the use
    of `mop.buffer.realloc` using a `!mop.context` which is created
    within the test itself. By rule 2 of `mop.buffer.realloc` the
    result of `mop.buffer.realloc` cannot outlive the context itself,
    so a copy is needed to return that buffer to the test harness
    for printing etc. It takes `!mop.context` as an argument to
    make sure that the context survives until the copy is performed.

    Any alignment constraint on the operand buffer is preserved in
    the result.

    Example:
    ```mlir
    %ctxt : !mop.context
    %buf1 : !mop.buffer
    %buf2 = mop.buffer.copy %ctxt %buf1 : !mop.buffer
    %buf2 : !mop.buffer  // with same alignment as %buf1
    ```
  }];

  let arguments = (ins MOP_ContextType:$theContext, MOP_BufferType:$buffer);
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = [{
    $theContext $buffer attr-dict `:` qualified(type($result))
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
    %ctxt = mop.context.create 4
    %ctxt : !mop.context
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
  let summary = "Creates a default state slot within the context at the "
                "given slot index.";
  let description = [{
    Creates a default state slot at the given slot index with room for the
    given number of cached buffers (which generally represent operator
    outputs). The op returns a chain indicating when it has completed execution.

    Example:
    ```mlir
    %ctxt : !mop.context

    %ch = mop.state_slot.create %ctxt 1 { slot = 0 }
    %ch : !bef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names. Thankfully this matches our desired argument order.
  let arguments = (ins MOP_ContextType:$theContext,
                       UI64Attr:$numBuffers,  // 0
                       UI64Attr: $slot);      // 1
  let results = (outs ChainType:$result);
  let assemblyFormat = "$theContext $numBuffers attr-dict";
}

//===----------------------------------------------------------------------===//
// SpecCreateOp
//===----------------------------------------------------------------------===//

def SpecCreateOp : MOP_Op<"spec.create", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Creates a tensor spec from a mix of runtime and static "
                "dimensions and a statically known dtype.";
  let description = [{
    Creates a tensor spec of statically known rank with dimensions taken
    from either runtime operands or statically known attributes. The element
    type must be known statically. Operands must be given for each unknown
    dimension in order. It is valid for all dimensions to be known statically,
    in which case no operands are required.

    Tensor specs are represented as M::TensorSpecs in the Modular runtime.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mop.spec.create %dim0 %dim1 : !mop.spec<?x?xf32>
    %spec : !mop.spec<?x?xf32>

    %spec = mop.spec.create : !mop.spec<32x4xf32>
    %spec : !mop.spec<32x4xf32>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names. We include the 'fst' and 'snd' prefixes to force this.
  // NOTE: The two attributes effectively encode the result's tensor spec type
  // in a form understandable by BEF.
  let arguments = (ins Variadic<Index>:$runtimeDims,
                       SI64ArrayElementsAttr:$fstRawShape,  // 0
                       UI8Attr:$sndRawDType);               // 1
  let results = (outs MOP_TensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDims attr-dict
                          `:` custom<RawShapeAndDType>($fstRawShape, $sndRawDType) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecCreateDynamicOp
//===----------------------------------------------------------------------===//

def SpecCreateDynamicOp : MOP_Op<"spec.create.dyn", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Creates a tensor spec from a runtime DType and a mix of "
                "runtime and static dimensions.";
  let description = [{
    Creates a tensor spec of statically known rank with element type taken
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

    %spec = mop.spec.create.dyn %dtype %dim0, %dim1 : !mop.spec<?x?x?>
    %spec : !mop.spec<?x?x?>

    %spec = mop.spec.create %dtype : !mop.spec<32x4x?>
    %spec : !mop.spec<32x4x?>
    ```
  }];

  // NOTE: The attribute effectively encodes the shape of the result's tensor
  // spec type in a form understandable by BEF.
  let arguments = (ins MOP_DTypeType:$runtimeDType, Variadic<Index>:$runtimeDims,
                       SI64ArrayElementsAttr:$rawShape);
  let results = (outs MOP_TensorSpecType:$result);
  let assemblyFormat = [{ $runtimeDType $runtimeDims attr-dict
                          `:` custom<RawShape>($rawShape) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecDimsDynamicOp
//===----------------------------------------------------------------------===//

def SpecDimsDynamicOp : MOP_Op<"spec.dims.dyn", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Returns the runtime values of the dimensions of tensor spec "
                "which are not statically known.";
  let description = [{
    Returns the actual runtime value of each dimension in a tensor spec where
    that dimension is not known statically. It is valid for the spec to be fully
    static, in which case no results are returned (and the op is a no-op).

    Example:
    ```mlir
    %spec : !mop.spec<?x?xf32>
    %dim0, %dim1 = mop.spec.dims.dyn %spec : !mop.spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mop.spec<32x?xf32>
    %dim = mop.spec.dims.dyn %spec : !mop.spec<32x?xf32>
    %dim : index

    %spec : !mop.spec<32x4xf32>
    mop.spec.dims.dyn %spec : !mop.spec<32x4xf32> // no-op
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $spec, SI64ArrayElementsAttr:$rawShape);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    $spec attr-dict `:` custom<TensorSpecTypeWithRawShape>(type($spec), $rawShape)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecSizeOp
//===----------------------------------------------------------------------===//

def SpecSizeOp : MOP_Op<"spec.size"> {
  let summary = "Returns the byte size needed for a buffer for a tensor with "
                "the given tensor spec.";
  let description = [{
    Returns the number of bytes needed to represent the buffer backing a tensor
    of the given tensor spec. Generally the spec's type will contain at least
    one unknown dimension and/or an unknown element type (otherwise the size
    is already known).

    Example:
    ```mlir
    %spec : !mop.spec<32x?xf32>
    %size = mop.spec.size %spec : !mop.spec<32x?xf32>
    %size : index
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// SpecNumElemsOp
//===----------------------------------------------------------------------===//

def SpecNumElemsOp : MOP_Op<"spec.num_elems"> {
  let summary = "Returns the number of elements in a tensor with the given "
                "tensor spec.";
  let description = [{
    Returns the number of elements in a tensor of the given tensor spec.
    Generally the spec's type will contain at least one unknown dimension
    (otherwise the number of elements is already known).

    Example:
    ```mlir
    %spec : !mop.spec<32x?xf32>
    %elems = mop.spec.num_elems %spec : !mop.spec<32x?xf32>
    %elems : index
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// SpecDTypeOp
//===----------------------------------------------------------------------===//

def SpecDTypeOp : MOP_Op<"spec.dtype"> {
  let summary = "Returns the DType of the given tensor spec.";
  let description = [{
    Returns the runtime DType of the given tensor spec. Generally the spec's
    type will have an unknown element type (otherwise the result is already
    known).

    Example:
    ```mlir
    %spec : !mop.spec<32x?xunk>
    %dtype = mop.spec.dtype %spec : !mop.spec<32x?xunk>
    %dtype : !mop.dtype
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $spec);
  let results = (outs MOP_DTypeType:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

def TensorCreateOp : MOP_Op<"tensor.create", [
    TypesMatchWith<"tensorSpec operand must match result tensor type spec",
                   "result", "tensorSpec",
                   "$_self.cast<TensorType>().getTensorSpec()">,
    TypesMatchWith<"buffer operand must be consistent with result tensor type",
                   "result", "buffer",
                   "$_self.cast<TensorType>().getTensorSpec().getExpectedBufferType()">]> {
  let summary = "Creates a tensor from a tensor spec and a buffer.";
  let description = [{
    Creates a tensor from a given tensor spec and a buffer. The construction
    is delayed until the given chain is ready, since the buffer is typically
    written into as a side-effect of an operation.

    Example:
    ```mlir
    %ch : !bef.chain

    %spec : !mop.spec<4x16xf32>
    %buf : !mop.buffer<256>
    %tensor = mop.tensor.create %ch %spec %buf : !mop.tensor<4x16xf32>
    %tensor : !mop.tensor<4x16xf32>

    %spec : !mop.spec<?x16xf32>
    %buf : !mop.buffer
    %tensor = mop.tensor.create %ch %spec %buf : !mop.tensor<?x16xf32>
    %tensor : !mop.tensor<?x16xf32>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    operand buffer will be copied into the tensor if it could still
    be mutated, ie if there are multiple outstanding consumers of the
    buffer's SSA value:
    ```mlir
    %buf1 = mop.buffer.alloc ...
    %ten1 = mop.tensor.create %ch ... %buf1 // copy
    ... other users of %buf1 ...

    %buf2 = mop.buffer.alloc ...
    %ten2 = mop.tensor.create %ch ... %buf2 // move
    ... no other users of %buf2 ...
    ```

    If the buffer is the result of a `mop.buffer.realloc` then it
    will be 'borrowing' rather than 'owning'. However the same rules
    apply:
    ```mlir
    %buf3 = mop.buffer.realloc ...
    %ten3 = mop.tensor.create %ch ... %buf3 // copy (as 'owning')
    ... other users of %buf3 ...

    %buf4 = mop.buffer.realloc ...
    %ten4 = mop.tensor.create %ch ... %buf4 // move (as 'borrowing')
    ... no other users of %buf4 ...
    ```
  }];

  let arguments = (ins ChainType: $chain, MOP_TensorSpecType: $tensorSpec,
                       MOP_BufferType: $buffer);
  let results = (outs MOP_TensorType: $result);
  let assemblyFormat = [{ $chain $tensorSpec $buffer attr-dict
                          `:` qualified(type($result)) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorExtractOp
//===----------------------------------------------------------------------===//

def TensorExtractOp : MOP_Op<"tensor.extract", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extracts the tensor spec and buffer of a tensor.";
  let description = [{
    Returns the tensor spec and buffer components of a tensor. The resulting
    buffer will have a statically known size if and only if the tensor spec is
    fully static.

    Example:
    ```mlir
    %tensor : !mop.tensor<8x64xf32>
    %spec, %buf = mop.tensor.extract %tensor : !mop.tensor<8x64xf32>
    %spec : !mop.spec<8x64>
    %buf : !mop.buffer<2048>

    %tensor : !mop.tensor<?x64xf32>
    %spec, %buf = mop.tensor.extract %tensor : !mop.tensor<?x64xf32>
    %spec : !mop.spec<?x64>
    %buf : !mop.buffer<>
    ```

    Tensors are immutable, but buffers are mutable. This means the
    returned buffer will be a copy of the tensor buffer if the tensor
    operand could still be observed, ie if if there are multiple outstanding
    consumers of the tensor's SSA value:
    ```mlir
    %ten1 = mop.tensor.create ...
    %spec1, %buf1 = mop.tensor.extract %ten1 // copy
    ... other users of %ten1 ...

    %ten2 = mop.tensor.create ...
    %spec2, %buf2 = mop.tensor.extract %ten2 // move
    ... no other users of %ten2 ...
    ```

    If the tensor's buffer was the result of a `mop.buffer.realloc`
    then it will be 'borrowing' rather than 'owning'. However the
    same rules apply:
    ```mlir
    %buf3 = mop.buffer.realloc ...
    %ten3 = mop.tensor.create %buf3
    %spec4, %buf4 = mop.tensor.extract %ten3  // copy (as 'owning')
    ... other users of %ten3 ...

    %buf5 = mop.buffer.realloc ...  // (*)
    %ten5 = mop.tensor.create %buf5
    %spec6, %buf6 = mop.tensor.extract %ten5  // move (as 'borrowing')
    ... no other users of %ten5 ...
    ```
    Note that any mutation of %buf6 will be applied to the cached
    buffer in the context, as established by (*). However, this is
    acceptable due to the rules of `mop.buffer.realloc`.
  }];

  let arguments = (ins MOP_TensorType: $tensor);
  let results = (outs MOP_TensorSpecType: $tensorSpecResult, MOP_BufferType: $bufferResult);
  let assemblyFormat = "$tensor attr-dict `:` qualified(type($tensor))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelFrameworkAttr : ConfinedAttr<StrAttr, [
    AttrConstraint<
      CPred<"BEFContents::isValidFrameworkName($_self.cast<StringAttr>().getValue())">,
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
        %ctxt = mop.context.create 0
        bef.output %ch, %ctxt : !bef.chain, !mop.context
      }
      execute (%ch: !bef.chain, %ctxt: !mop.context, %arg: tensor<?x4x16xf32>)
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
