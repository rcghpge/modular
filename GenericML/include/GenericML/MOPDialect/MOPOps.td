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
// ShapePackOp
//===----------------------------------------------------------------------===//

def ShapePackOp : MOPOp<"shape.pack"> {
  let summary = "Constructs a tensor shape from a mix of runtime and static "
                "dimensions";
  let description = [{
    Constructs a shape of statically known rank with dimensions taken from
    either runtime operands or the given shape type. Operands must be given
    for each unknown '?' shape dimension in order. It is valid for the shape
    to be fully static, in which case no operands are required.

    Example:
    ```mlir
    %dim0 : index
    %dim1 : index

    %shape = mop.shape.pack %dim0 %dim1 : !mop.shape<?x?xf32>
    %shape : !mop.shape<?x?xf32>

    %shape = mop.shape.pack %dim : !mop.shape<32x?xf32>
    %shape : !mop.shape<32x?xf32>

    %shape = mop.shape.pack : !mop.shape<32x4xf32>
    %shape : !mop.shape<32x4xf32>
    ```
  }];

  let arguments = (ins Variadic<Index>:$runtimeDims);
  let results = (outs TensorShapeType:$result);
  let assemblyFormat = "$runtimeDims attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ShapeUnpackOp
//===----------------------------------------------------------------------===//

def ShapeUnpackOp : MOPOp<"shape.unpack",
                          [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs a shape into its dimensions where they are unknown "
                "in the shape's type";
  let description = [{
    Returns the runtime value of each dimension in a shape who's shape type
    has the unknown '?' dimension. It is valid for the shape
    to be fully static, in which case no results are returned.

    Example:
    ```mlir
    %shape : !mop.shape<?x?xf32>
    %dim0, %dim1 = mop.shape.unpack %shape : !mop.shape<?x?xf32>
    %dim0, %dim1 : index, index

    %shape : !mop.shape<32x?xf32>
    %dim = mop.shape.unpack %shape : !mop.shape<32x?xf32>
    %dim : index

    %shape : !mop.shape<32x4xf32>
    mop.shape.unpack %shape : !mop.shape<32x4xf32>  // no-op
    ```
  }];

  let arguments = (ins TensorShapeType: $shape);
  let results = (outs Variadic<Index>:$results);
  let assemblyFormat = "$shape attr-dict `:` qualified(type($shape))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ShapeSizeOp
//===----------------------------------------------------------------------===//

def ShapeSizeOp : MOPOp<"shape.size"> {
  let summary = "Calculates the byte size of a tensor with given shape and "
                "dtype";
  let description = [{
    Returns the number of bytes needed to represent the buffer backing a tensor
    of the given shape and dtype. Generally the shape's type will contain at
    least one dynamic dimension.

    Example:
    ```mlir
    %shape : !mop.shape<32x?xf32>
    %size = mop.shape.size %shape !mop.dtype<f32> : !mop.shape<32x?xf32>
    %size : index
    ```
  }];

  let arguments = (ins TensorShapeType: $shape, TypeAttrOf<DTypeType>: $dtype);
  let results = (outs Index:$result);
  let assemblyFormat = "$shape $dtype attr-dict `:` qualified(type($shape))";
}

//===----------------------------------------------------------------------===//
// TensorPackOp
//===----------------------------------------------------------------------===//

def TensorPackOp : MOPOp<"tensor.pack", [
    TypesMatchWith<"shape operand must match result tensor type shape",
                   "result", "tensorShape",
                   "$_self.cast<TensorType>().getTensorShape()">,
    TypesMatchWith<"buffer operand must be consistent with result tensor type",
                   "result", "buffer",
                   "$_self.cast<TensorType>().getExpectedBufferType()",
                   "BufferType::agreeOnSize">]> {
  let summary = "Constructs a tensor from a shape and buffer";
  let description = [{
    Constructs a tensor from a given shape and buffer.

    Example:
    ```mlir
    %shape : !mop.shape<4x16>
    %buf : !mop.buffer<size=256>
    %tensor = mop.tensor.pack %shape %buf : !mop.tensor<4x16xf32>
    %tensor : !mop.tensor<4x16xf32>

    %shape : !mop.shape<?x16>
    %buf : !mop.buffer<>
    %tensor = mop.tensor.pack %shape %buf : !mop.tensor<?x16xf32>
    %tensor : !mop.tensor<?x16xf32>
    ```
  }];

  let arguments = (ins TensorShapeType: $tensorShape, BufferType: $buffer);
  let results = (outs TensorType: $result);
  let assemblyFormat = "$tensorShape $buffer attr-dict `:` qualified(type($result))";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the result tensor's dtype.
    DTypeType getDType();
  }];
}

//===----------------------------------------------------------------------===//
// TensorUnpackOp
//===----------------------------------------------------------------------===//

def TensorUnpackOp : MOPOp<"tensor.unpack", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Deconstructs a tensor into its shape and buffer";
  let description = [{
    Returns the shape and buffer components of a tensor. The resulting buffer
    will have a statically known size if and only if the tensor shape is
    fully static.

    Example:
    ```mlir
    %tensor : !mop.tensor<8x64xf32>
    %shape, %buf = mop.tensor.unpack %tensor : !mop.tensor<8x64xf32>
    %shape : !mop.shape<8x64>
    %buf : !mop.buffer<size=2048>

    %tensor : !mop.tensor<?x64xf32>
    %shape, %buf = mop.tensor.unpack %tensor : !mop.tensor<?x64xf32>
    %shape : !mop.shape<?x64>
    %buf : !mop.buffer<>
    ```
  }];

  let arguments = (ins TensorType: $tensor);
  let results = (outs TensorShapeType: $tensorShapeResult, BufferType: $bufferResult);
  let assemblyFormat = "$tensor attr-dict `:` qualified(type($tensor))";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

def ModelOp : MOPOp<"model", [HasParent<"ModuleOp">, Symbol,
                              IsolatedFromAbove]> {
  let summary = "Represents an executable model with 'init' and 'execute' "
                "entry points.";
  let description = [{
    This op constructs an executable model from single-block regions
    representing it's 'init' and 'execute' entry points.

    The 'init' entry point has the implicit signature:
      () -> (!bef.chain, !mop.context)

    The 'execute' entry point has the implicit signature:
      (!bef.chain, !mop.context, in1, ..., inn) ->
      (!bef.chain, res1, ..., resm)
    where:
      (in1, ..., inn) -> (res1, ..., resm)
    is given as the op's explicit signature.

    The runtime implicitly calls 'init' when the model executable is
    first loaded, and re-uses the resulting !mop.context for each
    subsequent call to 'execute'.

    Models are represented by MOP::Model in the Modular runtime.

    Example:
    ```mlir
    // In the top-level
    mop.model @example_model
      init {
        ...
      }
      execute (%ch: !bef.chain, %ctxt: !mop.context, %arg: !mop.tensor<4x?xf32>)
              -> (!bef.chain, !mop.tensor<4x?xf32>) {
        ...
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, DictionaryAttr:$extraAttrs,
                       TypeAttrOf<FunctionType>:$inputResultTypes);
  let regions = (region SizedRegion<1>:$init, SizedRegion<1>:$execute);

  let extraClassDeclaration = [{
    Block *getInitBlock() { return &getInit().front(); }
    Block *getExecuteBlock() { return &getExecute().front(); }
  }];

  let builders = [
    /// Creates a model with empty 'init' and 'execute' bodies and
    /// with `inputTypes` and `resultTypes` for the op's explicit signature.
    /// `inputArgLocs` specifies the location information for the
    /// explicit arguments to the 'execute' entry point, and thus should
    /// have the same length as `inputTypes`.
    OpBuilder<(ins "StringAttr":$sym_name,
                   "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$resultTypes,
                   "ArrayRef<Location>":$inputArgLocs,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$extraAttrs)>,
  ];

  // We can almost get away with an assemblyFormat of:
  //  "$sym_name attr-dict `init` $init `execute` $inputResultTypes $execute";
  // but we want to bind the execute region/block arguments right
  // after the 'execute' keyword and not within its region/block.
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
}

#endif  // GENERICML_MOPDIALECT_MOPOPS_TD
