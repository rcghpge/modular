//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines types in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPES_TD
#define GENERICML_MGPDIALECT_MGPTYPES_TD

include "GenericML/MGPDialect/MGPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MGP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MGPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// CpuBufferType
//===----------------------------------------------------------------------===//

def MGP_CpuBufferType : MGP_Type<"CpuBuffer", "cpu.buffer",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of CPU-hosted, mutable, uninterpreted byte arrays, with "
                "an optional statically known size.";
  let description = [{
    `!mgp.cpu.buffer`s are mutable CPU-hosted byte arrays which may have a
    statically known size. Nothing is implied about the internal structure
    of the bytes, such as rank, dimensions, underlying element type, padding,
    or element endianness. The compiler is responsible for ensuring the
    producers and consumers of buffers agree on their interpretation of the
    buffer for the intended target, and correctly sequence their side-effects.

    Generally `!mgp.cpu.buffer`s are the lowest-common denominator representation
    of tensor data across CPU kernel implementations. Buffers may be wrapped by
    various tensor types, including `!mgp.cpu.tensor`, or used directly.

    This type implements `ShapedTypeInterface` as a rank-1 tensor over
    `uint8_t` with optimally known first dimension. This in turn supports the
    `ElementsAttrInterface` on buffer attributes.

    Buffers are represented by `M::LLCL::BufferRef` in the Modular runtime.

    Note that buffers may respect alignment constraints. However those are
    specified either when the buffer is created, or are encoded in the
    attribute backing a constant buffer, and are not captured within the
    buffer type.

    Though not exposed in MLIR, buffers may be 'owning' or 'borrowing' over
    their underlying memory. Owning buffers automatically reclaim their
    underlying memory when no further references to the buffer are possible.
    Borrowing buffers assume the lifetime of the buffer is contained within
    the lifetime of the memory being borrowed. This difference is relevant
    to understanding the performance of many of the mgp ops.

    Similarly, though not exposed in MLIR or release builds, buffers may
    also be 'mutable' or 'immutable'. The runtime will assert fail if
    an immutable buffer is modified. This is for debugging only.

    Since buffers are generally considered mutable, any op reading from a
    buffer must also accept a chain, and any op writing to a buffer must
    also return a chain. In this way the compiler can correctly sequence
    side-effects over buffers.

    The size of a buffer type has two representations:
     - 'std::optional form': an `std::optional<uint64_t>`, with none denoting unknown.
     - 'Raw form': An UI64Attr, with ~0 denoting unknown.

    Examples:
    ```mlir
    !mgp.cpu.buffer<16>   // A 16-byte buffer.
    !mgp.cpu.buffer       // A buffer whose size is only known at runtime
    ```
  }];

  let parameters = (ins OptionalParameter<"std::optional<uint64_t>">:$optSize);
  let assemblyFormat = "(`<` $optSize^ `>`)?";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unknown.
     /// That ref will point to this static.
     static const int64_t unknownSize;

     /// Returns the buffer's size in 'raw form', which is
     /// a UI64Attr with value ~0 if the size is unknown.
     IntegerAttr getRawSize() const;
  }];

  let builders = [
    /// Constructs a buffer type from its size in 'raw form'.
    /// As a convenience we allow the null attribute to denote unknown
    /// size.
    TypeBuilder<(ins "IntegerAttr":$rawSizeOrNull)>,

    /// Constructs a buffer type from its size in 'raw form'.
    TypeBuilder<(ins "uint64_t":$rawSize)>,
  ];
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def MGP_ContextType : MGP_Type<"Context", "context"> {
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

    Contexts are represented by `M::MGP::Context` in the Modular runtime.

    Example:
    ```mlir
    !mgp.context
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DTypeType
//===----------------------------------------------------------------------===//

def MGP_DTypeType : MGP_Type<"DType", "dtype"> {
  let summary = "Runtime representation of the element type of a CPU tensor.";
  let description = [{
    This type represents a compact runtime encoding of an element type. It is
    implemented by M::DType in the Modular runtime, and is only intended to
    represent common int and float types common across modern CPUs.

    Examples:
    ```mlir
    !mgp.dtype   // A uint8_t encoding of an element type at runtime.
    ```
  }];

  let extraClassDeclaration = [{
    /// Returns the DType representing the given MLIR IntegerType, FloatType
    /// or NoneType. The NoneType is represented as unknown. Returns an error
    /// if the MLIR type has no representation as a DType.
    static ErrorOr<std::optional<DType>> fromMlirType(Type mlirType);

    /// Returns the MLIR IntegerType, FloatType or NoneType representing the
    /// given optDType. The unknown value is represented as the NoneType MLIR
    /// type. Returns an error if the DType has no representation as an MLIR
    /// type.
    static ErrorOr<Type> toMLIRType(MLIRContext *context, std::optional<DType> optDType);

    /// Returns error if the given optDType cannot be represented as an
    /// MLIR IntegerType, FloatType or NoneType. Otherwise returns success.
    static ErrorOrSuccess hasMLIRType(std::optional<DType> optDType);

    /// Returns the 'raw form' of optDType. This will be 0 if DType is unknown,
    /// otherwise the DType's value.
    static uint8_t asRawDType(std::optional<DType> optDType);

    /// Return the 'optional form' of rawDType. This will be none if the
    /// rawDType is 0, otherwise the DType with rawDType as value.
    static std::optional<DType> fromRawDType(uint8_t rawDType);
  }];
}

//===----------------------------------------------------------------------===//
// CpuTensorSpecType
//===----------------------------------------------------------------------===//

def MGP_CpuTensorSpecType : MGP_Type<"CpuTensorSpec", "cpu.tensor_spec", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of tensor specs of statically known rank.";
  let description = [{
    This type represents the shape and dtype of a tensor hosted on
    the CPU. The rank must be between one and five. Each dimension must be
    a strictly positive integer value, or the distinguished '?' dimension
    denoting unknown. The dtype must be a DType, or none denoting unknown.

    This type does not support:
     - Dynamic ranks.
     - Symbolic dimensions or dtypes.

    Tensor specs are represented by M::TensorSpec in the Modular runtime.
    The runtime value will agree with the rank, known shape dimensions,
    and known dtype of the tensor spec type.

    For convenience this type may be denoted using the same body syntax
    as the built-in tensor type. However, the dtype is parsed using
    the M::DType string parser, or may be 'unk' to denote unknown.

    Rank-0 Tensors that represent scalars are allowed; they are stored
    identically to rank 1 tensors of length 1.

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
    NoneType to represent an unknown dtype. This means:
     - We use 'dims' to refer to the dimensions in
       ArrayRef<std::optional<uint64_t>> form, but 'shape' to refer to
       them in ArrayRef<int64_t> form.
     - We use 'dtype' to refer to the dtype in DType form, but
       'elementType' to refer to it in MLIR type form.

    The components of tensor spec types have three representation forms:
     - 'std::optional form': an array of std::optional<uint64_t> for the dimensions
       (none denotes unknown dimension) and an std::optional<DType> for the dtype
       (none denotes unknown). This is the most natural representation
       from the MLIR/C++ side. However, care is taken to ensure only DTypes
       which are also representable as MLIR types are considered valid.
     - 'Shaped type form': an array of int64_t for the shape
       (ShapedType::kDynamicSize denotes unknown dimension) and an MLIR
       type for the element type (NoneType denotes unknown). This supports
       the ShapedTypeInterface, but not all IntegerTypes and FloatTypes have
       a DType representation.
     - 'Raw form': an IntArrayElementsAttr of uint64_ts for the dimensions
       (~0 denotes unknown dimension), and a IntegerAttr of a 8-bit unsigned
       integer for the dtype (using the DType::value encoding, but
       with the 'invalid' (zero) DType denoting unknown). This supports ops
       which must, in effect, encode tensor spec types into their argument
       attributes using only the built-in MEF attributes.

    Examples:
    ```mlir
    !mgp.cpu.tensor_spec<4x16xf32>
    !mgp.cpu.tensor_spec<?x?xunk>
    ```
  }];

  // NOTE: To support the ShapedType::getShape() method we must maintain
  // the shape in the ArrayRef<int64_t> form as well as our preferred
  // ArrayRef<std::optional<uint64_t>> form.
  let parameters = (ins ArrayRefParameter<"std::optional<uint64_t>">:$dims,
                        "std::optional<DType>":$optDType,
                        ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<DimsAndDType>($dims, $optDType, $shape) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns the rank of this tensor spec.
    size_t getRank() const;

    /// Returns the number of unknown, aka 'dynamic' dimensions in
    /// this tensor spec.
    size_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or nullopt if the
    /// dimension is dynamic.
    std::optional<uint64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// nullopt if the tensor spec is not fully static.
    std::optional<uint64_t> getNumElements() const;

    /// Returns the number of elements for the known dimensions of this spec.
    /// The actual number of elements will be this value multiplied by the
    /// runtime dimensions of all dynamic dimensions.
    uint64_t getPreScaledNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or nullopt if the tensor spec is not fully static.
    std::optional<uint64_t> getSizeInBytes() const;

    /// Returns the byte size needed to represent getNumPreScaledElements(),
    /// or nullopt if the tensor spec does not have a statically known dtype.
    std::optional<uint64_t> getPreScaledSizeInBytes() const;

    /// Returns the dimensions of this tensor spec in 'raw form'.
    /// Each element of the result will be an UI64Int, with ~0 denoting
    /// an unknown dimension.
    IntArrayElementsAttr getRawDims() const;

    /// Returns the dtype of this tensor spec in 'raw form'.
    /// The result will be a UI8Int using the DType::value encoding,
    /// with 0 denoting the unknown dtype.
    IntegerAttr getRawDType() const;

    /// Returns a CpuBufferType which captures the required size for this
    /// tensor. The result will only have a size constraint if the tensor's
    /// spec is fully static.
    CpuBufferType getExpectedBufferType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'optional form'.
    TypeBuilder<(ins "ArrayRef<std::optional<uint64_t>>":$dims,
                     "std::optional<DType>":$optDType)>,

    /// Constructs a tensor spec from its components in 'shaped type form'.
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                        "Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw form',
    /// using underlying integer values.
    TypeBuilder<(ins "ArrayRef<uint64_t>":$rawDims, "uint8_t":$rawDType)>,

    /// Construct a tensor spec from its components in 'raw form'.
    /// As a convenience, the null dtype attribute is taken to denote the
    /// unknown element type.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawDims,
                                        "IntegerAttr":$rawDTypeOrNull)>,
  ];
}

//===----------------------------------------------------------------------===//
// CpuTensorType
//===----------------------------------------------------------------------===//

// TODO: See also zap.ndbuffer.
def MGP_CpuTensorType : MGP_Type<"CpuTensor", "cpu.tensor", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of CPU-hosted tensors whose underlying buffer is in "
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
    !mgp.cpu.tensor<4x16xf32>
    !mgp.cpu.tensor<?x?xui8>
    ```
  }];

  let parameters = (ins MGP_CpuTensorSpecType:$tensorSpec);
  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) `>`";
}

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
