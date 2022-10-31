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
include "GenericML/MOPDialect/MOPAttrs.td"
include "GenericML/MOPDialect/MOPTypes.td"

// Base class for ops in this dialect.
class MOP_Op<string mnemonic, list<Trait> traits = []> :
    Op<MOPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MOP_Op<"buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constructs a buffer whose data is aliased with the "
                "`uint8_t` array in the given attribute.";
  let description = [{
    Constructs a buffer whose byte array is aliased with the `uint8_t` array
    in the given attribute. The attribute must support contiguous iteration
    over `uint8_t`. The attribute must agree with the buffer type's size and/or
    alignment if given.

    Example valid attributes include `BufferAttr`, `DenseUI8ResourceElementsAttr`
    and `DenseI8Array`. However note that `DenseI8Array` does not convey any
    align constraint.

    Note that any align constraint of the argument is not propagated into the
    result type.

    Example:
    ```mlir
    %buf = mop.buffer.constant #mop.buffer<8: "0x00010203">
    %buf : !mop.buffer<size=4>  // at least 8 byte aligned in storage

    %buf = mop.buffer.constant dense_resource<label>:!mop.buffer<size=4, align=8>
    %buf : !mop.buffer<size=4>  // at least 8 byte aligned in storage

    %buf = mop.buffer.constant array<ui8: 0, 1, 2, 3>
    %buf : !mop.buffer<size=4>  // no specific alignment in storage
    ```
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the align constraint for the buffer, if any.
    Optional<int64_t> getAlign();
  }];
}

//===----------------------------------------------------------------------===//
// BufferCreateOp
//===----------------------------------------------------------------------===//

def BufferCreateOp : MOP_Op<"buffer.create", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates a buffer of given runtime or static size and "
                "optional alignment.";
  let description = [{
    Allocates a buffer whose byte array is of a given size. The size can be
    supplied either as a runtime argument or as an attribute. Any align
    constraint must be supplied as an attribute.

    Note that any align constraint is not propagated into the result type.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index

    %buf = mop.buffer.create %ctxt %size {align=8:si64}
    %buf : !mop.buffer<>  // at least 8 byte aligned in storage

    %buf = mop.buffer.create %ctxt {size=64:si64}
    %buf : !mop.buffer<size=64>  // no specific alignment in storage

    %buf = mop.buffer.create %ctxt {size=64:si64, align=16:si64}
    %buf : !mop.buffer<size=64>  // at least 16 byte aligned in storage
    ```
  }];

  // TODO(#4301): Would be nicer to use OptionalAttr for $size and $align but
  // MLIRToBEF cannot handle that.
  let arguments = (ins MOP_ContextType:$theContext, Optional<Index>:$runtimeSize,
                       DefaultValuedAttr<SI64Attr, "ShapedType::kDynamicSize">:$size,
                       DefaultValuedAttr<SI64Attr, "ShapedType::kDynamicSize">:$align);
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$theContext ($runtimeSize^)? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferRecreateOp
//===----------------------------------------------------------------------===//

def BufferRecreateOp : MOP_Op<"buffer.recreate", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates or reuses a buffer of given runtime or static size "
                "and optional alignment, and caches it within in a primitive "
                "context.";
  let description = [{
    The operation has access to a buffer cached at the given output index in
    the primitive context at the given slot in the global context. If a buffer
    of matching size already exists then it is returned directly. Otherwise a
    new buffer is allocated and the cache is updated.

    Note that any align constraint on the buffer type is ignored if the
    cached buffer is reused. In particular, in:
    ```mlir
       %buf1 = mop.buffer.recreate %ctxt %size { align=8, slot=2, output=0 }
       %buf2 = mop.buffer.recreate %ctxt %size { align=16, slot=2, output=0 }
    ```
    the second recreate will never re-allocate the cached buffer from the first.

    If the size is known statically then the runtime size operand does not need
    to be supplied as an operand. In this case the 'recreate' is simply
    pre-allocating a buffer of statically known size.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index

    %buf = mop.buffer.recreate %ctxt %size { align=8:si64, slot=42, output=2 }
    %buf : !mop.buffer<>  // at least 8 byte aligned in storage

    %buf = mop.buffer.recreate %ctxt { size=64:si64, slot=42, output=2 }
    %buf : !mop.buffer<size=64>  // no specific alignment in storage
    ```
  }];

  // TODO(#4301): Would be nicer to use OptionalAttr for $size and $align but
  // MLIRToBEF cannot handle that.
  let arguments = (ins MOP_ContextType:$theContext, Optional<Index>:$runtimeSize,
                       DefaultValuedAttr<SI64Attr, "ShapedType::kDynamicSize">:$size,
                       DefaultValuedAttr<SI64Attr, "ShapedType::kDynamicSize">:$align,
                       UI64Attr:$slot, UI64Attr:$output);
  let results = (outs MOP_BufferType:$result);
  let assemblyFormat = "$theContext ($runtimeSize^)? attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MOP_Op<"context.create"> {
  let summary = "Constructs a context with room for primitive operation "
                "contexts.";
  let description = [{
    Constructs a context with the given number of slots for primitive operation
    contexts. Generally a computational operator, such as 'lib.foo', will be
    backed by two primitives, such as 'libp.foo.setup' and 'libp.foo'. The
    former can be called by the model's 'init' to allocate and initialize
    a primitive operation context for each call to 'lib.foo' in the model.
    The latter can retrieve that primitive operation context during the model's
    'execute'.

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
// PrimContextCreateOp
//===----------------------------------------------------------------------===//

def PrimContextCreateOp : MOP_Op<"primcontext.create"> {
  let summary = "Constructs a primitive context into the global mop context.";
  let description = [{
    Constructs a default primitive context with the given number of outputs
    into the given slot of the global mop context. A buffer corresponding to
    each output value can be cached into the primitive context. The op returns
    a chain indicating when it have completed execution.

    Example:
    ```mlir
    %ctxt : !mop.context

    %ch = mop.primcontext.create %ctxt 0 1
    %ch : !bef.chain
    ```
  }];

  let arguments = (ins MOP_ContextType:$theContext, UI64Attr:$numOutputs,
                       UI64Attr: $slot);
  let results = (outs ChainType:$result);
  let assemblyFormat = "$theContext $slot $numOutputs attr-dict";
}

//===----------------------------------------------------------------------===//
// SpecWrapOp
//===----------------------------------------------------------------------===//

def SpecWrapOp : MOP_Op<"spec.wrap", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constructs a tensor spec from a mix of runtime and static "
                "dimensions.";
  let description = [{
    Constructs a tensor spec of statically known rank with dimensions taken
    from either runtime operands or statically known dimensions, and using the
    given statically known dtype. Operands must be given for each unknown (-1)
    dimension in order. It is valid for all dimensions to be known, in which
    case no operands are required.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mop.spec.wrap %dim0 %dim1 [-1, -1] f32
    %spec : !mop.spec<?x?xf32>

    %spec = mop.spec.wrap %dim [32, -1] f32
    %spec : !mop.spec<32x?xf32>

    %spec = mop.spec.wrap [32, 4] f32
    %spec : !mop.spec<32x4xf32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$runtimeDims,
                       SI64ArrayElementsAttr:$dims, TypeAttr:$elementType);
  let results = (outs MOP_TensorSpecType:$result);
  let assemblyFormat = [{
    $runtimeDims `[` custom<DenseIntArray>($dims, "64") `]`
    custom<DTypeAttr>($elementType) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecUnwrapOp
//===----------------------------------------------------------------------===//

def SpecUnwrapOp : MOP_Op<"spec.unwrap", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs the statically unknown dimensions in a tensor "
                "spec.";
  let description = [{
    Returns the runtime value of each dimension in a tensor spec where the
    dimensions is not known statically. It is valid for the spec to be fully
    static, in which case no results are returned (and the op is useless!).

    Example:
    ```mlir
    %spec : !mop.spec<?x?xf32>
    %dim0, %dim1 = mop.spec.unwrap %spec [-1, -1] : !mop.spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mop.spec<32x?xf32>
    %dim = mop.spec.unwrap %spec [32, -1] : !mop.spec<32x?xf32>
    %dim : index

    %spec : !mop.spec<32x4xf32>
    mop.spec.unwrap %spec [32, 4] : !mop.spec<32x4xf32> // no-op
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $spec, SI64ArrayElementsAttr:$dims);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = [{
    $spec `[` custom<DenseIntArray>($dims, "64") `]` attr-dict
    `:` qualified(type($spec))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecSizeOp
//===----------------------------------------------------------------------===//

def SpecSizeOp : MOP_Op<"spec.size"> {
  let summary = "Calculates the byte size of a buffer for a tensor with given "
                "spec";
  let description = [{
    Returns the number of bytes needed to represent the buffer backing a tensor
    of the given spec. Generally the spec's type will contain at least one
    dynamic dimension.

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
// TensorWrapOp
//===----------------------------------------------------------------------===//

def TensorWrapOp : MOP_Op<"tensor.wrap", [
    TypesMatchWith<"spec operand must match result tensor type spec",
                   "result", "tensorSpec",
                   "$_self.cast<TensorType>().getTensorSpec()">,
    TypesMatchWith<"buffer operand must be consistent with result tensor type",
                   "result", "buffer",
                   "$_self.cast<TensorType>().getExpectedBufferType()">]> {
  let summary = "Constructs a tensor from a tensor spec and a buffer.";
  let description = [{
    Constructs a tensor from a given tensor spec and buffer.

    Example:
    ```mlir
    %s : !mop.spec<4x16xf32>
    %buf : !mop.buffer<size=256>
    %tensor = mop.tensor.wrap %spec %buf : !mop.tensor<4x16xf32>
    %tensor : !mop.tensor<4x16xf32>

    %spec : !mop.spec<?x16xf32>
    %buf : !mop.buffer<>
    %tensor = mop.tensor.wrap %spec %buf : !mop.tensor<?x16xf32>
    %tensor : !mop.tensor<?x16xf32>
    ```
  }];

  let arguments = (ins MOP_TensorSpecType: $tensorSpec, MOP_BufferType: $buffer);
  let results = (outs MOP_TensorType: $result);
  let assemblyFormat = "$tensorSpec $buffer attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorUnwrapOp
//===----------------------------------------------------------------------===//

def TensorUnwrapOp : MOP_Op<"tensor.unwrap", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs a tensor into its tensor spec and buffer.";
  let description = [{
    Returns the tensor spec and buffer components of a tensor. The resulting
    buffer will have a statically known size if and only if the tensor spec is
    fully static.

    Example:
    ```mlir
    %tensor : !mop.tensor<8x64xf32>
    %spec, %buf = mop.tensor.unwrap %tensor : !mop.tensor<8x64xf32>
    %spec : !mop.spec<8x64>
    %buf : !mop.buffer<size=2048>

    %tensor : !mop.tensor<?x64xf32>
    %spec, %buf = mop.tensor.unwrap %tensor : !mop.tensor<?x64xf32>
    %spec : !mop.spec<?x64>
    %buf : !mop.buffer<>
    ```
  }];

  let arguments = (ins MOP_TensorType: $tensor);
  let results = (outs MOP_TensorSpecType: $tensorSpecResult, MOP_BufferType: $bufferResult);
  let assemblyFormat = "$tensor attr-dict `:` qualified(type($tensor))";
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
