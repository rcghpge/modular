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
include "GenericML/MGPDialect/MGPTypeInterfaces.td"
include "GraphRT/MEFDialect/MEFTypeInterfaces.td"
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

// TODO(#32165): DEPRECATED
def MGP_CpuBufferType : MGP_Type<"CpuBuffer", "cpu.buffer",
                         [DeclareTypeInterfaceMethods<ShapedTypeInterface>,
                          DeclareTypeInterfaceMethods<MGP_BufferTypeInterface>
                         ]> {
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

    Buffers are represented by `M::AsyncRT::BufferRef` in the Modular runtime.

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
     mlir::IntegerAttr getRawSize() const;
  }];

  let builders = [
    /// Constructs a buffer type from its size in 'raw form'.
    /// As a convenience we allow the null attribute to denote unknown
    /// size.
    TypeBuilder<(ins "mlir::IntegerAttr":$rawSizeOrNull)>,

    /// Constructs a buffer type from its size in 'raw form'.
    TypeBuilder<(ins "uint64_t":$rawSize)>,

    /// Constructs a buffer type which matches the require size of the given
    /// tensor spec type. The buffer will have unknown size if the tensor spec
    /// has dynamic rank, dimensions or dtype.
    TypeBuilderWithInferredContext<(ins "TensorSpecType":$tensorSpecType)>,

    /// Constructs a buffer type which matches the required size and any other
    /// properties of the given tensor type. The buffer will have unknown size
    /// if the tensor has dynamic rank, dimensions or dtype.
    TypeBuilderWithInferredContext<(ins "CpuTensorType":$tensorType)>,
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
// CpuTensorType
//===----------------------------------------------------------------------===//

// TODO(#32165): DEPRECATED
def MGP_CpuTensorType : MGP_Type<"CpuTensor", "cpu.tensor", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
    DeclareTypeInterfaceMethods<MGP_TensorTypeInterface>]> {
  let summary = "Type of CPU-hosted tensors whose underlying buffer is in "
                "dense row-major order according to a tensor spec.";
  let description = [{
    This type is for dense CPU-friendly, immutable tensors. It is parameterized
    over a tensor spec type. Values of this type consist of a runtime tensor
    spec and an underlying buffer. The elements in the buffer are in dense
    row-major order according to the runtime tensor spec, with endianness
    according to the target architecture.

    Tensors are represented by Tensor in the modular runtime.

    Unlike the MLIR builtin tensor type this type:
     - Only supports a DType-representable element types.
     - Does not support quantization attributes on the element type.
     - Does not support affine maps for describing layout.
     - Is explicitly backed by the Tensor type.
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

  let parameters = (ins "MGP::TensorSpecType":$tensorSpec);
  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) `>`";
}

//===----------------------------------------------------------------------===//
// ListType
//===----------------------------------------------------------------------===//

def MGP_List : MGP_Type<"List", "list",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of immutable list container.";
  let description = [{
    This type represents an immutable list of elements.
  }];

  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Opaque
//===----------------------------------------------------------------------===//

def MGP_OpaqueType : MGP_Type<"Opaque", "opaque",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque MGP type";
  let description = [{
    This is an opaque type that can represent anything.
    This is backed by an AnyAsyncValue type.

    Example:
    ```mlir
    !mgp.opaque
    !mgp.opaque<{foo = "bar"}>
    ```
  }];

  let parameters = (ins "StringAttr": $symbol,
                        OptionalParameter<"std::optional<DictionaryAttr>">:$metadata);
  let assemblyFormat = "`<` $symbol (`,` $metadata^)? `>`";
}

include "GenericML/MGPDialect/MGPDeviceTypes.td"

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
