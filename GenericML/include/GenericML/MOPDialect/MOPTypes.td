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

// TODO(#4727): The buffer and tensor types are not intended to be universal,
// so we should give them a 'cpu' or perhaps 'host' suffix. Similarly for the
// ops.

//===----------------------------------------------------------------------===//
// MOP_BufferType
//===----------------------------------------------------------------------===//

def MOP_BufferType : MOP_Type<"Buffer", "buffer",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of mutable, uninterpreted byte arrays, optionally with "
                "a statically known size.";
  let description = [{
    Buffers are mutable byte arrays which may have a statically known size.
    Nothing is implied about the internal structure of the bytes, such as rank,
    dimensions, underlying element type, padding, or element endianness. The
    compiler is responsible for ensuring the producers and consumers of buffers
    agree on their interpretation of the buffer for the intended target, and
    correctly sequence their side-effects.

    Generally buffers are the lowest-common denominator representation of
    tensor data across kernel implementations. Buffers may be wrapped by various
    tensor types, including !mop.tensor, or used directly.

    This type implements `ShapedTypeInference` as a rank-1 tensor over
    `uint8_t` with optimally known first dimension. This in turn supports the
    `ElementsAttrInterface` on buffer attributes.

    Buffers are represented by `M::LLCL::BufferRef` in the Modular runtime.

    Note that buffers may respect alignment constraints. However those are
    specified either when the buffer is created or are encoded in the
    attribute backing a constant buffer.

    Though not exposed in MLIR, buffers may be 'owning' or 'borrowing' over
    their underlying memory. Owning buffers automatically reclaim their
    underlying memory when no further references to the buffer are possible.
    Borrowing buffers assume the lifetime of the buffer is contained within
    the lifetime of the memory being borrowed. This difference is relevant
    to understanding the performance of many of the mop ops.

    Similarly, though not exposed in MLIR or release builds, buffers may
    also be 'mutable' or 'immutable'. The runtime will assert fail if
    an immutable buffer is modified. This is for debugging only.

    Since buffers are mutable, any op reading from a buffer must also accept
    a chain, and any op writing to a buffer must also return a chain. In
    this way the compiler can correctly sequence side-effects over buffers.

    The size of a buffer type has two representations:
     - 'Native' form: an `Optional<uint64_t>`, with none denoting unknown.
     - 'Raw attribute' form: An UI64Attr, with `UINT64_MAX` denoting
       unknown.

    Examples:
    ```mlir
    !mop.buffer<16>
    !mop.buffer
    ```
  }];

  let parameters = (ins OptionalParameter<"Optional<uint64_t>">:$size);
  let assemblyFormat = "(`<` $size^ `>`)?";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unconstrained.
     /// That ref will point to this static. We are forced to use int64_t
     /// for compatability with the MLIR builtins.
     static const int64_t unknownSize;

     /// Returns the buffer's size in 'raw attribute' form, which is
     /// a UI64Attr with value UINT64_MAX if the size is unknown.
     IntegerAttr getRawSize() const;
  }];

  let builders = [
    /// Constructs a buffer type from its size in 'raw attribute' form.
    /// As a convenience we allow the null attribute to denote unknown
    /// size.
    TypeBuilder<(ins "IntegerAttr":$rawSizeOrNull)>,

    /// Constructs a buffer type from its size in 'raw attribute' form,
    /// using the underlying uint64_t.
    TypeBuilder<(ins "uint64_t":$rawSize)>,
  ];
}

//===----------------------------------------------------------------------===//
// MOP_ContextType
//===----------------------------------------------------------------------===//

def MOP_ContextType : MOP_Type<"Context", "context"> {
  let summary = "Type of execution contexts holding a number of state slots.";
  let description = [{
    Contexts may be created by the 'init' entry point of a model, then reused
    by each call to the model's 'execute' entry point.

    Contexts may contain any number of 'state slots'. A state slot holds
    arbitrary data needed by operations or entire framework runtimes, such as
    attributes, scratch space, pre-computed data or cached buffers. Each op may
    declare its own state slot structure, and each call to that op in the graph
    (an 'operation') may have its own instance of that structure.

    In general a computational operator, such as 'lib.foo', will be represented
    by two primitives, such as 'libp.foo.setup' and 'libp.foo'. The former can
    be invoked by the model's 'init' body to allocate and initialize a state
    slot. The latter can access that state slot when the operation is executing
    in the model's 'execute' body.

    Contexts are represented by `M::MOP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mop.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MOP_UnknownType
//===----------------------------------------------------------------------===//

def MOP_UnknownType : MOP_Type<"Unknown", "unknown"> {
  let summary = "The distinguished unknown type.";
  let description = [{
    This type is used by TensorSpecType::getElementType to denote
    that the tensor spec has an unknown element type.
  }];
}

//===----------------------------------------------------------------------===//
// MOP_DTypeType
//===----------------------------------------------------------------------===//

def MOP_DTypeType : MOP_Type<"DType", "dtype"> {
  let summary = "Runtime representation of the element type of a CPU tensor.";
  let description = [{
    This type represents a compact runtime encoding of an element type. It is
    implemented by M::DType in the Modular runtime.

    Examples:
    ```mlir
    !mop.dtype   // A uint8_t encoding of an element type at runtime.
    ```
  }];

  let extraClassDeclaration = [{
    /// Returns the DType representing the given MLIR IntegerType, FloatType
    /// or UnknownType. The UnknownType is represented as none. Returns an error
    /// if the MLIR type has no representation as a DType.
    static ErrorOr<Optional<DType>> fromMlirType(Type mlirType);

    /// Returns the MLIR IntegerType, FloatType or UnknownType representing the
    /// given DType. The none value is represented as the UnknownType MLIR type.
    /// Returns an error if the DType has no representation as an MLIR type.
    static ErrorOr<Type> toMLIRType(MLIRContext *ctxt, Optional<DType> dtype);

    /// Returns non-empty error string if the given DType cannot be represented as
    /// an MLIR IntegerType, FloatType or UnknownType. Otherwise returns the
    /// empty string.
    static std::string hasNoMLIRType(Optional<DType> dtype);
  }];
}

//===----------------------------------------------------------------------===//
// MOP_TensorSpecType
//===----------------------------------------------------------------------===//

def MOP_TensorSpecType : MOP_Type<"TensorSpec", "spec", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and element type of a tensor. The rank
    must be between one and five. Each dimension must be a strictly positive
    integer value, or the distinguished -1 dimension denoting unknown. The
    element type must be a DType, or none denoting unknown.

    This type does not support:
     - Dynamic ranks.
     - Rank zero shapes, sometimes used to denote scalars.
     - Symbolic dimensions or element type.

    Tensor specs are represented by M::TensorSpec in the Modular runtime.
    The runtime value will agree with the rank, known shape dimensions,
    and known element type of the tensor spec type.

    For convenience this type may be denoted using the same body syntax
    as the built-in tensor type. However, the element type is parsed using
    the M::DType string parser, or may be 'unk' to denote unknown.

    DTypes can represent most variations of MLIR IntegerType and FloatTypes.
    However:
     - Signless and non power-of-two MLIR IntegerTypes have no DType
       representation.
     - The complex and a few float DTypes have no MLIR Type representation.
     - The 'invalid' (zero) DType is never permitted.
    If a tensor spec is given a DType we also check it is representable as
    an MLIR type.
    TODO(#4718): Are these discrepancies something we should redress?

    This type implements ShapedTypeInterface, using the distinguished
    UnknownType to represent an unknown element type. This means we
    must use 'dtype' to refer to the element type in DType form, but
    'elementType' to refer to the element type in MLIR type form.

    The components of tensor spec types have three representation forms:
     - 'Native' form: an array of int64_t for the shape (-1 denotes unknown
       dimension) and an Optional<DType> for the element type (none denotes
       unknown). This is the most natural representation from the MLIR/C++
       side. However, care is taken to ensure only DTypes which are
       also representable as MLIR types are considered valid.
     - 'Shaped type' form: an array of int64_t for the shape (-1 denotes
       unknown dimension) and an MLIR type for the element type (UnknownType
       denotes unknown). This supports the ShapedTypeInterface, but not
       all IntegerTypes and FloatTypes have a DType representation.
     - 'Raw attribute' form: an IntArrayElementsAttr of 64-bit signed
       integers for the shape (-1 denotes unknown dimension), and a
       IntegerAttr of a 8-bit unsigned integer for the element type
       (using the DType::value encoding, but with the 'invalid' (zero) DType
       denoting unknown). This supports ops which must, in effect, encode
       tensor spec types into their argument attributes using only the built-in
       BEF attributes.

    Examples:
    ```mlir
    !mop.spec<4x16xf32>
    !mop.spec<?x?xunk>
    ```
  }];

  let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "Optional<DType>":$dtype);
  let assemblyFormat = "`<` custom<ShapeAndDType>($shape, $dtype) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the maximum supported rank for all tensor specs.
    static constexpr size_t getMaximumRank() { return 5; }

    /// Returns the rank of this tensor spec.
    size_t getRank() const;

    /// Returns the number of unknown, aka 'dynamic' dimensions in
    /// this tensor spec.
    size_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or empty if the
    /// dimension is dynamic.
    Optional<uint64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// empty if the tensor spec is not fully static.
    Optional<uint64_t> getNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or empty if the tensor spec is not fully static.
    Optional<uint64_t> getSizeInBytes() const;

    /// Returns the shape of this tensor spec in 'raw' attribute form.
    /// Each element of the result will be an I64Int, with -1 denoting
    /// an unknown dimension.
    IntArrayElementsAttr getRawShape() const;

    /// Returns the element type of this tensor spec in 'raw' attribute
    /// form. The result will be a UI8Int using the DType::value encoding,
    /// with 0 denoting the unknown element type.
    IntegerAttr getRawDType() const;

    /// Returns a BufferType which captures the required size for this tensor.
    /// The result will only have a size constraint if the tensor's spec is
    /// fully static.
    BufferType getExpectedBufferType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'shaped type' form.
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                        "Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw attribute' form.
    /// As a convenience, the null dtype attribute is taken to denote the
    /// unknown element type.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawShape,
                                        "IntegerAttr":$rawDTypeOrNull)>,

    /// Construct a tensor spec from its components in 'raw attribute' form,
    /// using underlying integer values.
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape, "uint8_t":$rawDType)>,
  ];
}

//===----------------------------------------------------------------------===//
// MOP_TensorType
//===----------------------------------------------------------------------===//

// TODO: See also zap.ndbuffer.
def MOP_TensorType : MOP_Type<"Tensor", "tensor",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensors who's underlying buffer is in "
                "dense row-major order according to a tensor spec.";
  let description = [{
    This type is for dense CPU-friendly, immutable tensors of statically known
    rank, partially or fully statically known shape, and possibly statically
    known element type. It is parameterized over a tensor spec type. Values of
    this type consist of a runtime tensor spec and an underlying buffer. The
    elements in the buffer are in dense row-major order according to the
    runtime tensor spec, with endianness according to the target architecture.

    Tensors are represented by M::Tensor in the modular runtime.

    Unlike the MLIR builtin tensor type this type:
     - Only supports a DType-representable element types.
     - Only supports shapes of known rank.
     - Does not support quantization attributes on the element type.
     - Does not support affine maps for describing layout.
     - Is explicitly backed by the M::Tensor type.
     - Allows the element type to be unknown at compile time.

    Tensors are immutable. When a tensor is created from a buffer, the buffer
    must be copied if it could still be written. When a buffer is extracted
    from a tensor, the buffer must be copied if the tensor could still be read.

    For convenience we 'inline' the tensor shape into the tensor type syntax:
    ```mlir
    !mop.tensor<4x16xf32>
    !mop.tensor<?x?xui8>
    ```
  }];

  let parameters = (ins MOP_TensorSpecType:$tensorSpec);
  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) `>`";
}

#endif  // GENERICML_MOPDIALECT_MOPTYPES_TD
