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
  let summary = "Type of tensor specs.";
  let description = [{
    This type represents the shape and dtype of a concrete tensor. The
    tensor could be an !mgp.cpu.tensor, or some other tensor representation.

    TODO: Remove 'cpu' prefix.

    Both static and dynamic (aka unknown) aspects are supported:
     - The rank may be static (implied by the number of dimensions, ie the
       number of `x` delimiters) or dynamic (denoted by `*`). Only ranks up to
       kMaxRank are supported (currently 8). A rank of zero denotes 'scalars',
       though in practice this has the same runtime representation as rank 1,
       dimension 1.
     - For static ranks, each dimension may be static (an unsigned integer
       literal) or dynamic (denoted by `?`). Symbolic dimensions are not
       expressible in MGP.
     - The dtype may be static (using the DType::getFromString syntax) or
       dynamic (denoted by`unk`).

    Tensor specs are represented by M::TensorSpec in the Modular runtime.
    The runtime tensor spec for a concrete tensor should refine its compile-time
    tensor spec.

    DTypes can represent most variations of MLIR IntegerType and FloatTypes.
    However:
     - Signless and non power-of-two MLIR IntegerTypes have no DType
       representation.
     - The complex and a few float DTypes have no MLIR Type representation.
     - The 'invalid' (zero) DType is never permitted.
    If a tensor spec is given a DType we also require it to be representable as
    an MLIR type.
    TODO(#4718): Are these discrepancies something we should redress?

    This type implements ShapedTypeInterface. We'll use 'dims'/'dtype' when
    referring to our internal representation, and 'shape'/'element type' when
    referring to the ShapedTypeInterface representation.

    Here are all the different representations in flight:
     - For dimensions:
        - 'optional' form: std::optional<uint64_t>, where nullopt denotes
          a dynamic dimension. God's own representation.
        - 'signed' form: int64_t, where ShapedType::kDynamicSize denotes
          a dynamic dimension. As used by ShapedTypeInterface.
        - 'raw' form: uint64_t, where ~0 denotes a dynamic dimension. Used
          when dimensions need to be encoded into MEF ops via attributes.
     - For shapes:
        - 'optional' form: std::optional<ArrayRef<std::optional<uint64_t>>>,
          where nullopt denotes a dynamic rank.
        - 'shaped type' form: ArrayRef<int64_t>, as used by ShapedTypeInterface.
        - 'raw' form: IntArrayElementsAttr of UI64Ints.
     - For dtypes:
        - 'optional' form: std::optional<DType>, where nullopt denotes a
          dynamic dtype.
        - 'shaped type' form: An MLIR type, where NoneType denotes a dynamic
          dtype.
        - 'raw' form: uint8_t, where 0 (ie invalid) denotes a dynamic dtype.

    Examples:
    ```mlir
    !mgp.cpu.tensor_spec<4x16xf32>
    !mgp.cpu.tensor_spec<?x3xunk>
    !mgp.cpu.tensor_spec<*xi32>
    ```
  }];

  // NOTE: To support the ShapedTypeInterface::getShape() method we maintain
  // 'optional' (dimsOrEmpty) and 'shaped type' (shapeOrEmpty) forms.
  // See the builders for the smart constructors.
  let parameters = (ins "bool":$hasRank,
                        ArrayRefParameter<"std::optional<uint64_t>">:$dimsOrEmpty,
                        "std::optional<DType>":$optDType,
                        ArrayRefParameter<"int64_t">:$shapeOrEmpty);
  let assemblyFormat = "`<` custom<DimsAndDType>($hasRank, $dimsOrEmpty, $optDType, $shapeOrEmpty) `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// For the ShapedTypeInterface we implement:
    ///   bool hasRank();
    ///   ShapedType cloneWith(std::optional<ArrayRef<int64_t>> shape,
    ///                        Type elementType);
    ///   Type getElementType();
    ///   ArrayRef<int64_t> getShape(); // requires hasRank().
    /// CAUTION: Most of the following methods require hasRank() and will
    ///          assert fail otherwise. Take care.

    /// Returns the dimensions of this tensor spec, or nullopt if dynamic
    /// ranked.
    /// Dynamic ranked safe.
    std::optional<ArrayRef<std::optional<uint64_t>>> getOptDims() const;

    /// Returns the dimensions of this tensor spec.
    /// Requires hasRank().
    ArrayRef<std::optional<uint64_t>> getDims() const;

    /// Returns the rank of this tensor spec.
    /// Requires hasRank().
    size_t getRank() const;

    /// Returns the number of unknown, aka 'dynamic' dimensions in
    /// this tensor spec.
    /// Requires hasRank().
    size_t getNumDynamicDims() const;

    /// Return the dimension at index i in this spec, or nullopt if the
    /// dimension is dynamic.
    /// Requires hasRank().
    std::optional<uint64_t> operator[](size_t i) const;

    /// Returns the number of elements in a tensor of this spec, or
    /// nullopt if the tensor shape is not fully static (ie dynamic rank,
    /// or at least one dynamic dimension).
    /// Dynamic ranked safe.
    std::optional<uint64_t> getNumElements() const;

    /// Returns the number of elements for the known dimensions of this spec.
    /// The actual number of elements will be this value multiplied by the
    /// runtime dimensions of all dynamic dimensions.
    /// Requires hasRank().
    uint64_t getPreScaledNumElements() const;

    /// Returns the size of the buffer needed to represent a tensor
    /// of this spec, or nullopt if the tensor spec is not fully static
    /// (ie dynamic rank, at least one dynamic dimension, or dynamic dtype).
    /// Dynamic ranked safe.
    std::optional<uint64_t> getSizeInBytes() const;

    /// Returns the byte size needed to represent getNumPreScaledElements(),
    /// or nullopt if the tensor spec does not have a statically known dtype.
    /// Requires hasRank().
    std::optional<uint64_t> getPreScaledSizeInBytes() const;

    /// Returns the dimensions of this tensor spec in 'raw form'.
    /// Each element of the result will be an UI64Int, with ~0 denoting
    /// an unknown dimension.
    /// Requires hasRank().
    IntArrayElementsAttr getRawDims() const;

    /// Returns the dtype of this tensor spec in 'raw form'.
    /// The result will be a UI8Int using the DType::value encoding,
    /// with 0 denoting the unknown dtype.
    IntegerAttr getRawDType() const;

    /// Returns a CpuBufferType which captures the required size for this
    /// tensor. The result will only have a size constraint if the tensor's
    /// spec is fully static (rank, dims and dtype).
    /// Dynamic ranked safe.
    CpuBufferType getExpectedBufferType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'optional' form.
    TypeBuilder<(ins "std::optional<ArrayRef<std::optional<uint64_t>>>":$optDims,
                     "std::optional<DType>":$optDType)>,

    /// Constructs a tensor spec from its components in 'shaped type' form.
    /// Only supports statically known ranks.
    TypeBuilderWithInferredContext<(ins "ArrayRef<int64_t>":$shape,
                                        "Type":$elementType)>,

    /// Constructs a dynamically ranked tensor spec with the given element type.
    TypeBuilderWithInferredContext<(ins "Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw form',
    /// using underlying integer values. Only supports statically known ranks.
    TypeBuilder<(ins "ArrayRef<uint64_t>":$rawDims, "uint8_t":$rawDType)>,

    /// Construct a tensor spec from its components in 'raw form', using
    /// actual attributes. As a convenience, the null dtype attribute is taken
    /// to denote the unknown element type. Only supports statically known
    /// ranks.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawDims,
                                        "IntegerAttr":$rawDTypeOrNull)>,
  ];
}

//===----------------------------------------------------------------------===//
// CpuTensorType
//===----------------------------------------------------------------------===//

def MGP_CpuTensorType : MGP_Type<"CpuTensor", "cpu.tensor", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "Type of CPU-hosted tensors whose underlying buffer is in "
                "dense row-major order according to a tensor spec.";
  let description = [{
    This type is for dense CPU-friendly, immutable tensors. It is parameterized
    over a tensor spec type. Values of this type consist of a runtime tensor
    spec and an underlying buffer. The elements in the buffer are in dense
    row-major order according to the runtime tensor spec, with endianness
    according to the target architecture.

    Tensors are represented by M::Tensor in the modular runtime.

    Unlike the MLIR builtin tensor type this type:
     - Only supports a DType-representable element types.
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
    !mgp.cpu.tensor<*xi32>
    ```
  }];

  let parameters = (ins MGP_CpuTensorSpecType:$tensorSpec);
  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) `>`";
}

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
