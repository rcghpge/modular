//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Device-generic types in the MGP dialect. All of these types accept an
// #M.device_ref attribute indicating the specific type and 'virtual' device
// id the type should apply to.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDEVICETYPES_TD
#define GENERICML_MGPDIALECT_MGPDEVICETYPES_TD

include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypeInterfaces.td"
include "GraphRT/MEFDialect/MEFTypeInterfaces.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

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
// DeviceChainType
//===----------------------------------------------------------------------===//

def MGP_DeviceChainType : MGP_DeviceType<"DeviceChain", "chain", [
    DeclareTypeInterfaceMethods<MEF_ChainTypeInterface>,
]> {
  let summary = "Controls sequencing of side-effecting computations.";
  let description = [{
    `!mgp.chain`s allow asynchronous computations on the device with the
    `#M.device_ref` attribute to be sequenced. For example, a kernel should not
    read a buffer until it has finished being written to by some other kernel.
    Generally, all side-effecting operations both accept and return an
    `!mgp.chain`.

    All model values are represented in the host C++ runtime system by
    `AsyncRT::AsyncValueRef<T>`s. These become 'ready' when the asynchronous
    computation producing the `T` has completed in some sense (possibly
    with an error). Instances of `!mgp.chain` use `T`s which are (a subclass
    of) the empty struct `AsyncRT::Chain`.

    CPU computations only have two phases:
     - The computation may 'begin execution' when all inputs, including any
       input chain, are ready.
     - When the computation 'finishes execution' (possibly with an error) the
       result chain is marked as ready.
    Thus, on CPU devices, a 'ready' `!mgp.chain` indicates the computation has
    finished.

    However, computations on non-CPU devices with a CPU host, such as a CUDA
    GPU, may have three phases:
     - The computation may be 'launched' by the host as soon as all inputs are
       'known', even though there may still be computation on the device writing
       or allocating those inputs. For example, CUDA can launch a kernel
       depending on `CUdeviceptr`s who's asynchronous allocation may not have
       finished, or which are to be written to by some other launched kernel.
     - The computation actually 'begins execution' on the device when all
       computation associated with the inputs has finished. Typically, the
       device itself has its own synchronization primitives to express this
       without further host synchronization. For example, CUDA has `CUstream`s
       and `CUevent`s.
     - When the computation 'finishes execution', we wish to somehow indicate
       in the result chain that downstream launched kernels may now begin
       their execution.

    Thus, on some non-CPU devices, we actually want the `!mgp.chain` to be
    'ready' as soon as the associated computation has been launched (so that
    downstream computations may also be launched), but have some device-specific
    way to convey when the underlying computation has 'finished'. This 'ready'
    vs 'finished' distinction is why some devices may use a subtype of
    AsyncRT::Chain for their runtime representation.

    The same 'ready' vs 'finished' distinction also applies to other core
    MGP types, such as `!mgp.buffer and `!mgp.tensor`.

    Devices must provide primitives for converting device-specific chains to
    and from a host-side chain so as to support host-device and device-device
    synchronization.

    An `!mgp.chain` on the host device has the same runtime representation
    as a `!mef.chain`, ie `AsyncRT::AsyncValueRef<AsyncRT::Chain>`. No-op conversions
    may be required in some places.

    Examples:
    ```mlir
    // A chain for computations on the CUDA GPU with virtual device id 3.
    !mgp.chain<<"cuda", 3>>
    // A chain for computations on the CPU with virtual device 0, which by
    // convention is the host device. Thus this type is equivalent to
    // `!mef.chain`.
    !mgp.chain<<"cpu", 0>>
    // Abbreviation for above.
    !mgp.chain
    ```
  }];

  let parameters = (ins MGP_OptDeviceRefParam:$device);
  let assemblyFormat = "( `<` $device^ `>` )?";

  let builders = [
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device)>,
  ];
}

//===----------------------------------------------------------------------===//
// DeviceBufferType
//===----------------------------------------------------------------------===//

def MGP_DeviceBufferType : MGP_DeviceType<"DeviceBuffer", "buffer", [
    DeclareTypeInterfaceMethods<ShapedTypeInterface>,
]> {
  let summary = "Byte array buffers.";
  let description = [{
    `!mgp.buffer`s are possibly-mutable byte arrays which may have a statically
    known size, and which reside on the device given by the `#M.device_ref`
    attribute.

    Nothing is implied about the internal structure of the bytes, such as rank,
    dimensions, underlying element type, padding, or element endianness. The
    compiler is responsible for ensuring the producers and consumers of buffers
    agree on their interpretation of the buffer for the intended device.

    Generally `!mgp.buffer`s are the lowest-common denominator representation
    of tensor data across kernel implementations. Buffers may be wrapped by
    various tensor types, including `!mgp.tensor`, or used directly.

    This type implements `ShapedTypeInterface` as a rank-1 tensor over
    `uint8_t` with optimally known first dimension. This in turn supports the
    `ElementsAttrInterface` on buffer attributes.

    Note that buffers may respect alignment constraints. However those are
    specified either when the buffer is created, or are encoded in the
    attribute backing a constant buffer, and are not captured within the
    buffer type.

    Since `!mgp.buffers` are generally considered mutable, any op reading from
    a buffer must also accept an `!mgp.chain`, and any op writing to a buffer
    must also return an `!mgp.chain`. In this way the compiler can correctly
    sequence writes and reads.

    The size of a buffer type has two representations:
     - 'std::optional form': a `std::optional<uint64_t>`, with none denoting unknown.
     - 'Raw form': an `mlir::IntegerAttr` or `uint64_t`, with ~0 denoting unknown.

    `!mgp.buffer`s are represented by (subclasses of)
    `AsyncRT::AsyncValueRef<BufferRef>` in the host C++ runtime. Some devices,
    such as CUDA, may need to include additional data to control device-side
    synchronization, see `!mgp.chain` for more details.

    Examples:
    ```mlir
    // A 16-byte buffer on the CUDA GPU with virtual device id 3.
    !mgp.buffer<16, <"cuda", 3>>
    // A 16-byte buffer on the CPU with virtual device id 0.
    !mgp.buffer<16, <"cpu", 0>>
    // Abbreviation for the above.
    !mgp.buffer<16>
    // A dynamically-sized buffer on the CUDA GPU with virtual device id 3.
    !mgp.buffer<, <"cuda", 3>>
    // A dynamically-sized buffer on the CPU with virtual device id 0.
    !mgp.buffer<, <"cpu", 0>>
    // Abbreviation for the above.
    !mgp.buffer<>
    ```
  }];

  let parameters = (ins MGP_OptDeviceRefParam:$device,
                        OptionalParameter<"std::optional<uint64_t>">:$optSize);
  let assemblyFormat = "`<` ( $optSize^ )? ( `,` $device^ )? `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
     /// getShape may need to return an ArrayRef representing a
     /// dynamic rank-1 shape if the size is unknown.
     /// That ref will point to this static.
     static const int64_t unknownSize;

     /// Returns the buffer's size in 'raw form', which is a UI64Attr with
     /// value ~0 if the size is unknown.
     mlir::IntegerAttr getRawSize() const;
  }];

  let builders = [
    /// Constructs a buffer type from its size in 'optional' form.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "std::optional<uint64_t>":$optSize)>,

    /// Constructs a buffer type from its size in 'raw form'.
    /// As a convenience we allow the null attribute to denote unknown size.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "mlir::IntegerAttr":$rawSizeOrNull)>,

    /// Constructs a buffer type from its size in 'raw form'.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "uint64_t":$rawSize)>,

    /// Constructs a buffer type which matches the require size of the given
    /// tensor spec type. The buffer will have unknown size if the tensor spec
    /// has dynamic rank, dimensions or dtype.
    TypeBuilderWithInferredContext<(ins "M::DeviceRefAttr":$device,
                                        "M::MGP::TensorSpecType":$tensorSpecType)>,

    /// Constructs a buffer type which matches the required size, device, and
    /// any other properties of the given tensor type. The buffer will have
    /// unknown size if the tensor has dynamic rank, dimensions or dtype.
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
    `AsyncRT::AsyncValueRef<Tensor>` in the host C++ runtime. Some devices,
    such as CUDA, may need to include additional data to control device-side
    synchronization, see `!mgp.chain` for more details.

    Examples:
    ```mlir
    // Statically-shaped tensor on the CUDA GPU with virtual device id 3.
    !mgp.tensor<4x16xf32, <"cuda", 3>>
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
     /// spec will have an unkown dtype.
     M::MGP::TensorSpecType getTensorSpec(bool isShapeOnly);
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPDEVICETYPES_TD
