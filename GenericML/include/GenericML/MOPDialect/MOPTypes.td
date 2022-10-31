//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MOP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOPDIALECT_MOPTYPES_TD
#define GENERICML_MOPDIALECT_MOPTYPES_TD

include "GenericML/MOPDialect/MOPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MOP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MOPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// MOP_BufferType
//===----------------------------------------------------------------------===//

def MOP_BufferType : MOP_Type<"Buffer", "buffer",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of uninterpreted byte arrays, optionally with statically "
                "known size and alignment";
  let description = [{
    Buffers are byte arrays which may have a statically known size and
    alignment 'constraints'. Nothing is implied about the internal structure
    of the bytes, such as rank, dimensions, underlying element type, padding,
    or element endianness. The compiler is responsible for ensuring the
    producers and consumers of buffers agree on their interpretation of the
    buffer for the intended target.

    Generally buffers are the lowest-common denominator across kernel
    implementations. Buffers may be wrapped by various tensor types, or
    used directly.

    This type implements `ShapedTypeInference` as a rank-1 tensor over
    `uint8_t` with optimally known first dimension. This in turn supports the
    `ElementsAttrInterface` on buffer attributes.

    Buffers are represented by `M::LLCL::RCBufferRefs` in the Modular runtime.

    Note that align constraints in buffer types are generally only needed
    when constructing a buffer from a dense_resource. Otherwise the align
    constraint is not propagated within types to simplify type checking.

    Examples:
    ```mlir
    !mop.buffer<size = 16>
    !mop.buffer<size = 16, align = 64>
    !mop.buffer
    ```
  }];

  let parameters = (ins OptionalParameter<"Optional<int64_t>">:$size,
                        OptionalParameter<"Optional<int64_t>">:$align);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unconstrained.
     static const int64_t unknownSize;

     /// Returns this buffer type without any align constraint.
     BufferType forgetAlign();
  }];
}

//===----------------------------------------------------------------------===//
// MOP_ContextType
//===----------------------------------------------------------------------===//

def MOP_ContextType : MOP_Type<"Context", "context"> {
  let summary = "Type of execution contexts holding a number of primitive "
                "operation contexts";
  let description = [{
    Contexts are objects created by the 'init' phase of a model, then reused
    for each call to the model's 'execute'. Contexts may contain any number
    of primitive operation contexts. A primitive operation context holds
    attributes, scratch space, pre-computed data and, if required, cached
    output buffers needed by a primitive. Each primitive may declare its
    own context structure, and each call to that primitive in the graph may
    have its own instance of that structure.

    Contexts are represented by `M::MOP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mop.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MOP_TensorSpecType
//===----------------------------------------------------------------------===//

def MOP_TensorSpecType : MOP_Type<"TensorSpec", "spec", [
    DeclareTypeInterfaceMethods<SubElementTypeInterface>,
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. It must be
    between one and five dimensions. Each dimension can be a strictly positive
    integer value or the distinguished '?' unknown dimensions. The element type
    must be representable as an M::DType, which can express a smaller set of element
    types than the MLIR Integer and Float types. However for convenience we
    allow the dtype to be supplied as an MLIR type.

    This type does not support:
     - Implied scalars (eg with a rank zero shape).
     - Dynamic ranks.
     - Symbolic dimensions.
     - Unknown or symbolic dtypes.

    TensorSpecs are represented by M::TensorSpec in the Modular runtime.
    The runtime spec will have the same rank and matching dimensions for
    the known dimensions in this type, and will match the dtype exactly.

    For convenience this type can be constructed using the same body syntax
    as the built-in tensor type, however the dtype is parsed using the
    M::DType string parser.
    ```mlir
    !mop.spec<4x16xf32>
    !mop.spec<?x?xui8>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t", "dimensions">:$shape, "Type":$elementType);
  let assemblyFormat = "`<` custom<ShapeAndDType>($shape, $elementType) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the maximum supported rank for all tensor specs.
    static constexpr size_t getMaximumRank() { return 5; }

    /// Returns the rank of this tensor spec.
    size_t getRank() const;

    /// Returns the number of dynamic dimensions in this spec.
    int64_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or empty if the
    /// dimension is dynamic.
    Optional<int64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// empty if the spec is not fully static.
    Optional<int64_t> getNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or empty if the spec is not fully static.
    Optional<int64_t> getSizeInBytes() const;
  }];
}

//===----------------------------------------------------------------------===//
// MOP_TensorType
//===----------------------------------------------------------------------===//

// TODO: See also zap.ndbuffer.
def MOP_TensorType : MOP_Type<"Tensor", "tensor",
                         [DeclareTypeInterfaceMethods<SubElementTypeInterface>,
                          DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensors who's underlying buffer is in "
                "dense row-major order according to tensor spec.";
  let description = [{
    This type is for dense CPU-friendly tensors of statically known rank, partially
    or fully statically known shape, and statically known element type. It is
    parameterized over the tensor spec. Values of this type consist of a runtime
    spec and an underlying buffer. The elements in the buffer are in dense
    row-major order according to the runtime shape, with endianness according to
    the target architecture.

    Tensors are represented by M::Tensor in the modular runtime.

    Unlike the MLIR builtin tensor type this type:
     - Only supports a small set of element types.
     - Only supports shapes of known rank.
     - Does not support quantization attributes on the element type.
     - Does not support affine maps for describing layout.
     - Is explicitly backed by the M::Tensor type.

    For convenience we 'inline' the tensor shape into the tensor type syntax:
    ```mlir
    !mop.tensor<4x16xf32>
    !mop.tensor<?x?xui8>
    ```
  }];

  let parameters = (ins MOP_TensorSpecType:$tensorSpec);
  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) `>`";

  let extraClassDeclaration = [{
    /// Returns a BufferType which captures the required size for this tensor
    /// (with no align constraint). The result will only have a size constraint
    /// if the tensor's shape is fully static.
    BufferType getExpectedBufferType() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or empty if the spec is not fully static.
    Optional<int64_t> getSizeInBytes() const { return getTensorSpec().getSizeInBytes(); }
  }];
}

#endif  // GENERICML_MOPDIALECT_MOPTYPES_TD
