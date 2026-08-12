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

include "GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.td"
include "MLRT/GraphRT/MEFDialect/MEFTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Support/MDialect/MAttrs.td"

// Base class for types in this dialect.
class MGP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MGPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ContextType
//===----------------------------------------------------------------------===//

def MGP_ContextType : MGP_Type<"Context", "context",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
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

//===----------------------------------------------------------------------===//
// DeviceContextType
//===----------------------------------------------------------------------===//

def MGP_DeviceContextType : MGP_Type<"DeviceContext", "device_context",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "MGP representation of the driver DeviceContext";
  let description = [{
    This is essentially opaque to MGP but it is still useful to represent
    directly as it is simpler to introspect.

    Examples:
    ```mlir
    // A device context for CUDA gpu 3.
    !mgp.device_context<<"gpu", 3>>
    ```
  }];

  let parameters = (ins M_DeviceRefAttr:$device);
  let assemblyFormat = "`<` $device `>`";
}

//===----------------------------------------------------------------------===//
// DeviceGraphType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphType : MGP_Type<"DeviceGraph", "device_graph",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Type of a constructed device graph.";
  let description = [{
    Represents a constructed device-side graph (e.g. a CUDA graph) that can be
    replayed via `mgp.device_graph.execute`.

    The type carries the graph's signature as a `FunctionType`. Inputs are
    buffers the caller must copy into the graph's frozen-input addresses at
    replay time; results are buffers the graph writes through at replay time.
    Both inputs and results are passed as operands to
    `mgp.device_graph.execute` -- the distinction at the type level exists so
    that the synthesis pass and verifiers can reason about read-only vs.
    written operands without inspecting the build body.

    The device the graph targets is encoded by the device parameter on the
    individual `!mgp.buffer` types within the signature.

    Example:
    ```mlir
    !mgp.device_graph<
        (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) ->
        (!mgp.buffer<<"gpu", 0>>)>
    ```
  }];

  let parameters = (ins "FunctionType":$signature);
  let assemblyFormat = "`<` $signature `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::FunctionType":$signature)>,
    TypeBuilder<(ins "mlir::TypeRange":$inputs, "mlir::TypeRange":$results)>,
    TypeBuilder<(ins "mlir::ValueRange":$inputs, "mlir::ValueRange":$results)>,
  ];
}

//===----------------------------------------------------------------------===//
// DeviceGraphBuilderType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphBuilderType : MGP_Type<"DeviceGraphBuilder",
  "device_graph.builder",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Builder that records nodes into a device graph under construction.";
  let description = [{
    An opaque handle to a device-graph builder. Created in the build body of
    `mgp.device_graph.create` (from the body's `!mgp.device_context`) and
    threaded as the launch argument of `mogg.call.build_graph`, in place of the
    `!mogg.context` that `mogg.call.execute` would take. Each kernel's
    `build_graph` Mojo method records a node against the builder rather than
    dispatching immediately; once every node is recorded the builder is
    instantiated into an `!mgp.device_graph`.

    Example:
    ```mlir
    !mgp.device_graph.builder
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphNodeType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphNodeType : MGP_Type<"DeviceGraphNode", "device_graph.node",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque handle to a node recorded into a device graph.";
  let description = [{
    An opaque handle to a single node recorded into a device graph via a
    kernel's `build_graph` method. Produced by `mogg.call.build_graph` and
    propagated out of the recording stub. The handle carries no structure yet;
    it exists so that future cross-stub dependency tracking can name prior
    nodes (a `depends_on` operand list) without inspecting the build body.

    Example:
    ```mlir
    !mgp.device_graph.node
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DeviceGraphCacheType
//===----------------------------------------------------------------------===//

def MGP_DeviceGraphCacheType : MGP_Type<"DeviceGraphCache",
  "device_graph.cache",
  [DeclareTypeInterfaceMethods<MEF_AnyTypeInterface>]> {
  let summary = "Opaque handle to a model's device-graph cache.";
  let description = [{
    An opaque handle to the cache of instantiated device graphs belonging to a
    model. Produced by `mgp.device_graph.cache` from the model's `!mgp.context`
    and consumed by `mgp.device_graph.create`, which reuses a cached graph
    instead of rebuilding one when the key derived from its inputs hits.

    There is one cache per model: it lives in the state context, so it is
    created on first use and outlives any single execution.

    Example:
    ```mlir
    !mgp.device_graph.cache
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TensorSpecType
//===----------------------------------------------------------------------===//

def MGP_TensorSpecType : MGP_Type<"TensorSpec", "tensor_spec", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Type of tensor specs.";
  let description = [{
    `!mgp.tensor_spec` represents the shape and dtype of a tensor.

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

    DTypes can represent most variations of MLIR IntegerType and FloatTypes.
    However:
     - Signless and non power-of-two MLIR IntegerTypes have no DType
       representation.
     - The complex and a few float DTypes have no MLIR Type representation.
     - The 'invalid' (zero) DType is never permitted.
    If a tensor spec is given a DType we also require it to be representable as
    an MLIR type.

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

    `!mgp.tensor_specs` are represented by `MLRT::AsyncValueRef<M::TensorSpec>`
    in the host C++ runtime. All dimensions in the runtime value should 'refine'
    the corresponding compile time spec, ie runtime dims must equal statically
    known compile-time dims.

    Examples:
    ```mlir
    // Fully statically known spec.
    !mgp.tensor_spec<4x16xf32>
    // Spec with one dynamic dimension and dynamic dtype.
    !mgp.tensor_spec<?x3xunk>
    // Dynamically ranked spec.
    !mgp.tensor_spec<*xi32>
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
    std::optional<llvm::ArrayRef<std::optional<uint64_t>>> getOptDims() const;

    /// Returns the dimensions of this tensor spec.
    /// Requires hasRank().
    llvm::ArrayRef<std::optional<uint64_t>> getDims() const;

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
    mlir::IntegerAttr getRawDType() const;
  }];

  let builders = [
    /// Constructs a tensor spec from its components in 'optional' form.
    TypeBuilder<(ins "std::optional<llvm::ArrayRef<std::optional<uint64_t>>>":$optDims,
                     "std::optional<DType>":$optDType)>,

    /// Constructs a tensor spec from its components in 'shaped type' form.
    /// Only supports statically known ranks.
    TypeBuilderWithInferredContext<(ins "llvm::ArrayRef<int64_t>":$shape,
                                        "mlir::Type":$elementType)>,

    /// Constructs a dynamically ranked tensor spec with the given element type.
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType)>,

    /// Construct a tensor spec from its components in 'raw form',
    /// using underlying integer values. Only supports statically known ranks.
    TypeBuilder<(ins "llvm::ArrayRef<uint64_t>":$rawDims, "uint8_t":$rawDType)>,

    /// Construct a tensor spec from its components in 'raw form', using
    /// actual attributes. As a convenience, the null dtype attribute is taken
    /// to denote the unknown element type. Only supports statically known
    /// ranks.
    TypeBuilderWithInferredContext<(ins "IntArrayElementsAttr":$rawDims,
                                        "mlir::IntegerAttr":$rawDTypeOrNull)>,
  ];
}

//===----------------------------------------------------------------------===//
//
// Device-generic types in the MGP dialect. All of these types accept an
// #M.device_ref attribute indicating the specific type and 'virtual' device
// id the type should apply to.
//
//===----------------------------------------------------------------------===//

class MGP_DeviceType<string name, string mnemonic, list<Trait> traits = []> :
    MGP_Type<
      name,
      mnemonic,
      [DeclareTypeInterfaceMethods<MGP_HasDeviceTypeInterface>] # traits
    >;

def MGP_OptDeviceRefParam :
  DefaultValuedParameter<"M::DeviceRefAttr",
                         "M::DeviceRefAttr::getDefault($_ctxt)">;

//===----------------------------------------------------------------------===//
// DeviceBufferType
//===----------------------------------------------------------------------===//

def MGP_DeviceBufferType : MGP_DeviceType<"DeviceBuffer", "buffer", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Byte array buffers.";
  let description = [{
    `!mgp.buffer`s are mutable byte arrays which reside on the device given by
    the `#M.device_ref` attribute.

    Nothing is implied about the internal structure of the bytes, such as rank,
    dimensions, underlying element type, padding, or element endianness. The
    compiler is responsible for ensuring the producers and consumers of buffers
    agree on their interpretation of the buffer for the intended device.

    Generally `!mgp.buffer`s are the lowest-common denominator representation
    of tensor data across kernel implementations. Buffers may be wrapped by
    various tensor types, including `!mgp.tensor`, or used directly.

    This type implements `ShapedTypeInterface` as a rank-1 tensor over
    `uint8_t`. This in turn supports the `ElementsAttrInterface` on buffer
    attributes.

    Note that buffers may respect alignment constraints. However those are
    specified either when the buffer is created, or are encoded in the
    attribute backing a constant buffer, and are not captured within the
    buffer type.

    Since `!mgp.buffers` are generally considered mutable, any op reading from
    a buffer must also accept an `!mgp.chain`, and any op writing to a buffer
    must also return an `!mgp.chain`. In this way the compiler can correctly
    sequence writes and reads.

    `!mgp.buffer`s are represented by (subclasses of)
    `MLRT::AsyncValueRef<BufferRef>` in the host C++ runtime. Some devices,
    such as CUDA, may need to include additional data to control device-side
    synchronization, see `!mgp.chain` for more details.

    Examples:
    ```mlir
    // A buffer on the CUDA GPU with virtual device id 3.
    !mgp.buffer<<"gpu", 3>>
    // A buffer on the CPU with virtual device id 0.
    !mgp.buffer<<"cpu", 0>>
    // Abbreviation for the above.
    !mgp.buffer
    ```
  }];

  let parameters = (ins MGP_OptDeviceRefParam:$device);
  let assemblyFormat = "(`<` $device^ `>`)?";
  let genVerifyDecl = 1;

  let builders = [
    /// Constructs a buffer type from the given device.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device)>,

    /// Constructs a buffer type which matches the device and any other properties
    /// of the given tensor type.
    TypeBuilderWithInferredContext<(ins "M::MGP::DeviceTensorType":$tensorType)>,
  ];
}

//===----------------------------------------------------------------------===//
// DeviceTensorType
//===----------------------------------------------------------------------===//

def MGP_DeviceTensorType : MGP_DeviceType<"DeviceTensor", "tensor", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Type of tensors respecting a tensor spec.";
  let description = [{
    `!mgp.tensor`s are immutable tensors consisting of a buffer and a tensor
    spec, where the buffer resides on the device given by the `#M.device_ref`
    attribute. The elements in the buffer are generally in dense row-major
    order according to the runtime tensor spec, with endianness according to
    the device's architecture.

    Tensors are immutable. When a tensor is created from a buffer, the buffer
    must be copied if it could still be written. When a buffer is extracted
    from a tensor, the buffer must be copied if the tensor could still be read.

    `!mgp.tensor`s are represented by (subclasses of) `Tensor` in the
    host-side C++ runtime. Some devices, such as CUDA, may need to include
    additional data with the tensor, such as synchronization streams. The
    tensor's underlying byte-array buffer may also be passed around as a
    raw `void*` pointer.

    `!mgp.tensors`s are represented by (subclasses of)
    `MLRT::AsyncValueRef<Tensor>` in the host C++ runtime. Some devices,
    such as CUDA, may need to include additional data to control device-side
    synchronization, see `!mgp.chain` for more details.

    Examples:
    ```mlir
    // Statically-shaped tensor on the CUDA GPU with virtual device id 3.
    !mgp.tensor<4x16xf32, <"gpu", 3>>
    // Dynamically-shaped tensor on the CPU with virtual device id 0.
    !mgp.tensor<?x?xf32, <"cpu", 0>>
    // Abbreviation for the above.
    !mgp.tensor<?x?xf32>
    ```
  }];

  let parameters = (ins
    MGP_OptDeviceRefParam:$device,
    "M::MGP::TensorSpecType":$tensorSpec
  );

  let assemblyFormat = "`<` custom<TensorSpec>($tensorSpec) ( `,` $device^ )? `>`";

  let builders = [
    /// Constructs a tensor type from the given device and tensor spec.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "M::MGP::TensorSpecType":$tensorSpecType)>,
  ];

  let extraClassDeclaration = [{
     /// Return the tensor spec for tensor. If isShapeOnly is true, the tensor
     /// spec will have an unknown dtype.
     M::MGP::TensorSpecType getTensorSpec(bool isShapeOnly);
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPTYPES_TD
