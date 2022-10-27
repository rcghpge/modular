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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "GenericML/MOPDialect/MOPDialect.td"
include "GenericML/MOPDialect/MOPAttrs.td"
include "GenericML/MOPDialect/MOPTypes.td"

// Base class for ops in this dialect.
class MOPOp<string mnemonic, list<Trait> traits = []> :
    Op<MOPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MOPOp<"buffer.constant",
                             [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constructs a buffer whose data is aliased with the "
                "`uint8_t` array in the given attribute";
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
  let results = (outs BufferType:$result);
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

// TODO: Would prefer to write a type constraint like
//   TypesMatchWith<"result type must match buffer type without align",
//                  "bufferType", "result",
//                  "$_self.forgetAlign()">
// instead of using DeclareOpInterfaceMethods<InferTypeOpInterface> but
// ods generates invalid code.
def BufferCreateOp : MOPOp<"buffer.create",
                        [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates a buffer of given runtime or static size and "
                "optional alignment";
  let description = [{
    Allocates a buffer whose byte array is of a given size. If the size is
    known statically within the buffer type it does not need to be supplied
    as an argument. Otherwise it must be provided explicitly. Any align
    constraint in the buffer type is respected in the allocation.

    Note that any align constraint is not propagated into the result type.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index

    %buf = mop.buffer.create %ctxt %size : !mop.buffer<align=8>
    %buf : !mop.buffer<>  // at least 8 byte aligned in storage

    %buf = mop.buffer.create %ctxt : !mop.buffer<size=8>
    %buf : !mop.buffer<size=8>  // no specific alignment in storage
    ```
  }];

  // NOTE: We include an explicit $bufferType attributes so that the alignment
  // can be conveyed to the op even though we don't want it in the result
  // type.
  let arguments = (ins ContextType:$context, Optional<Index>:$runtimeSize,
                       TypeAttrOf<BufferType>:$bufferType);
  let results = (outs BufferType:$result);
  let assemblyFormat = "$context ($runtimeSize^)?  attr-dict `:` $bufferType";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the align constraint for the buffer, if any.
    Optional<int64_t> getAlign();
  }];
}

//===----------------------------------------------------------------------===//
// BufferRecreateOp
//===----------------------------------------------------------------------===//

def BufferRecreateOp : MOPOp<"buffer.recreate", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Allocates or reuses a buffer of runtime size and optional "
                "alignment, and caches it within in a primitive context";
  let description = [{
    The operation has access to a buffer cached at the given output index in
    the primitive context at the given slot in the global context. If a buffer
    of matching size already exists then it is returned directly. Otherwise a
    new buffer is allocated and the cache is updated.

    Note that any align constraint on the buffer type is ignored if the
    cached buffer is reused. In particular, in:
    ```mlir
       %buf1 = mop.buffer.recreate %ctxt %size { slot=2, output=0 } : !mop..buffer<align=8>
       %buf2 = mop.buffer.recreate %ctxt %size { slot=2, output=0 } : !mop..buffer<align=16>
    ```
    the second recreate will never re-allocate the cached buffer.

    If the bufffer type has a size constraint then the runtime size operand
    does not need to be supplied as an argument. In this case the 'recreate'
    is simply pre-allocating a buffer of statically known size.

    Example:
    ```mlir
    %ctxt : !mop.context
    %size : index

    %buf = mop.buffer.recreate %ctxt %size { slot=42, output=2 } : !mop.buffer<align=8>
    %buf : !mop.buffer<>  // at least 8 byte aligned in storage

    %buf = mop.buffer.recreate %ctxt { slot=42, output=2 } : !mop.buffer<size=64>
    %buf : !mop.buffer<size=64>  // no specific alignment in storage
    ```
  }];

  // NOTE: We include an explicit $bufferType attributes so that the alignment
  // can be conveyed to the op even though we don't want it in the result
  // type.
  let arguments = (ins ContextType:$context, Optional<Index>:$runtimeSize,
                       I64Attr:$slot, I64Attr:$output, TypeAttrOf<BufferType>:$bufferType);
  let results = (outs BufferType:$result);
  let assemblyFormat = "$context ($runtimeSize^)? attr-dict `:` $bufferType";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the align constraint for the buffer, if any.
    Optional<int64_t> getAlign();
  }];
}

//===----------------------------------------------------------------------===//
// ContextCreateOp
//===----------------------------------------------------------------------===//

def ContextCreateOp : MOPOp<"context.create"> {
  let summary = "Constructs a context with room for primitive operation "
                "contexts";
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
  let results = (outs ContextType:$result);
  let assemblyFormat = "$numSlots attr-dict";
}

//===----------------------------------------------------------------------===//
// SpecWrapOp
//
// TODO: The statically known dimensions and element type need to be passed as
// attributes so we can encode the operation in BEF.
//===----------------------------------------------------------------------===//

def SpecWrapOp : MOPOp<"spec.wrap"> {
  let summary = "Constructs a tensor spec from a mix of runtime and static "
                "dimensions";
  let description = [{
    Constructs a spec of statically known rank with dimensions taken from
    either runtime operands or the given spec type. Operands must be given
    for each unknown '?' shape dimension in order. It is valid for the spec
    to be fully static, in which case no operands are required.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %spec = mop.spec.wrap %dim0 %dim1 : !mop.spec<?x?xf32>
    %spec : !mop.spec<?x?xf32>

    %spec = mop.spec.wrap %dim : !mop.spec<32x?xf32>
    %spec : !mop.spec<32x?xf32>

    %spec = mop.spec.wrap : !mop.spec<32x4xf32>
    %spec : !mop.spec<32x4xf32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$runtimeDims);
  let results = (outs TensorSpecType:$result);
  let assemblyFormat = "$runtimeDims attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecUnwrapOp
//
// TODO: May need to return every dimension instead of just the dynamic dims?
//===----------------------------------------------------------------------===//

def SpecUnwrapOp : MOPOp<"spec.unwrap",
                          [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs a spec into its dimensions where they are unknown "
                "in the given spec type";
  let description = [{
    Returns the runtime value of each dimension in a spec who's spec type
    has the unknown '?' dimension. It is valid for the spec to be fully static,
    in which case no results are returned.

    Example:
    ```mlir
    %spec : !mop.spec<?x?xf32>
    %dim0, %dim1 = mop.spec.unwrap %spec : !mop.spec<?x?xf32>
    %dim0, %dim1 : index, index

    %spec : !mop.spec<32x?xf32>
    %dim = mop.spec.unwrap %spec : !mop.spec<32x?xf32>
    %dim : index

    %spec : !mop.spec<32x4xf32>
    mop.spec.unwrap %spec : !mop.spec<32x4xf32>  // no-op
    ```
  }];

  let arguments = (ins TensorSpecType: $spec);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpecSizeOp
//===----------------------------------------------------------------------===//

def SpecSizeOp : MOPOp<"spec.size"> {
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

  let arguments = (ins TensorSpecType: $spec);
  let results = (outs Index:$result);
  let assemblyFormat = "$spec attr-dict `:` qualified(type($spec))";
}

//===----------------------------------------------------------------------===//
// TensorWrapOp
//===----------------------------------------------------------------------===//

def TensorWrapOp : MOPOp<"tensor.wrap", [
    TypesMatchWith<"spec operand must match result tensor type spec",
                   "result", "tensorSpec",
                   "$_self.cast<TensorType>().getTensorSpec()">,
    TypesMatchWith<"buffer operand must be consistent with result tensor type",
                   "result", "buffer",
                   "$_self.cast<TensorType>().getExpectedBufferType()",
                   "BufferType::agreeOnSize">]> {
  let summary = "Constructs a tensor from a tensor spec and a buffer";
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

  let arguments = (ins TensorSpecType: $tensorSpec, BufferType: $buffer);
  let results = (outs TensorType: $result);
  let assemblyFormat = "$tensorSpec $buffer attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TensorUnwrapOp
//===----------------------------------------------------------------------===//

def TensorUnwrapOp : MOPOp<"tensor.unwrap", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs a tensor into its tensor spec and buffer";
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

  let arguments = (ins TensorType: $tensor);
  let results = (outs TensorSpecType: $tensorSpecResult, BufferType: $bufferResult);
  let assemblyFormat = "$tensor attr-dict `:` qualified(type($tensor))";
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
