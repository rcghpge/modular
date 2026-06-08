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

include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPDialect.td"
include "GraphCompiler/GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.td"
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
    DeclareTypeInterfaceMethods<MGP_TensorTypeInterface>,
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

  let parameters = (ins MGP_OptDeviceRefParam:$device,
                        "M::MGP::TensorSpecType":$tensorSpec);
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
