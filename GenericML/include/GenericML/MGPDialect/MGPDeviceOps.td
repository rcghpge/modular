//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Device-generic operations in the MGP dialect. Most of these ops accept
// (at least one) #M.device_ref attribute indicating the specific type and
// 'virtual' device id the ops should apply to. Some ops must also reference
// the 'host' device in addition to some 'other' device. These ops implement
// the `MGP_HasDeviceOpInterface`.
//
// Note that the `DevicePrimitivesPass` rewrites all ops implementing
// `MGP_HasDeviceOpInterface` such that:
//  - the op is renamed to include its device's label name as a suffix.
//  - all #M.device_ref attributes are replaced by just the device's id.
// In this way we have the convenience of first-class device attributes on
// ops for use during lowering and MGP-level transformations, but have the
// efficiency of direct dispatch to per-device label primitives without the
// overhead of dealing with device lables at runtime.
//
// Note that MGP has an `!mgp.chain` type which carries its device. For host
// devices this is has the same runtime representation as for `!mef.chain`.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPDEVICEOPS_TD
#define GENERICML_MGPDIALECT_MGPDEVICEOPS_TD

include "GenericML/MGPDialect/MGPOpInterfaces.td"
include "GenericML/MGPDialect/MGPTensorSpec.td"
include "GenericML/MGPDialect/MGPDialect.td"
include "GenericML/MGPDialect/MGPTypes.td"
include "GenericML/MGPDialect/MGPAttrs.td"
include "GenericML/MGPDialect/MGPOps.td"
include "GraphRT/GRTDialect/GRTChainInterface.td"
include "GraphRT/GRTDialect/GRTExtendsLifetimeInterface.td"
include "GraphRT/GRTDialect/GRTMayAliasInterface.td"
include "GraphRT/MEFDialect/MEFTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
//
// Custom parsers:
//
//   // Parses an optional keyword and #M.device_ref into $device, and
//   // assigns $chain... the type !mgp.chain<$device>.
//   custom<OptDeviceRef>(keyword, $device, type($chain)...)
//
//   // Parses optionally type-annotated operands into $gatingOperands, using
//   // the default type !mgp.chain<$device>.
//   custom<ChainCreateParenOperandList>(
//     ref($device), $gatingOperands, type($gatingOperands)
//   )
//
//   // Parses an !mgp.buffer type and extracts its raw size and device
//   // attributes, and assigns $chain... the type !mgp.chain<$device>.
//   custom<BufferTypeToRawSizeDevice>(type($buffer)?, $rawSize,
//                                     $device, type($chain)...)
//
//   // Parses a statically sized !mgp.buffer type and extracts its size and
//   // device attributes, and assigns $chain... the type !mgp.chain<$device>.
//   custom<StaticBufferTypeToSizeDevice>(type($buffer), $size,
//                                        $device, type($chain)...)
//
//   // Parses a statically sized !mgp.buffer type and extracts its size
//   // and device attributes, and assigns $inBuffer the type
//   // !mgp.buffer<, $device>
//   custom<StaticBufferTypeToSizeDeviceBuffer>(type($outBuffer), $aSize,
//                                              $bDevice, type($inBuffer))
//
//   // Parses a dynamically sized !mgp.buffer type, extracts its device
//   // attribute, and assigns $chain... the type !mgp.chain<$device>.
//   custom<DynamicBufferTypeToDevice>(type($buffer), $device, type($chain)...)
//
//   // Parses an !mgp.buffer type, extracts its device, and assigns $chain...
//   // the type !mgp.chain<$device>.
//   custom<BufferTypeToDevice>($buffer, $device, type($chain)...)
//
//   // Parses an !mgp.tensor type and extracts its device and buffer type.
//   custom<TensorTypeToDeviceBuffer>(type($tensor), $device, type($buffer))
//
//   // Parses an !mgp.tensor type and extracts its device and tensor spec or
//   // shape.
//   custom<TensorTypeToDeviceSpecOrShape>($isShapeOnly, type($tensor),
//                                         $device, type($outSpec))
//
//   // Parses an !mgp.tensor type and extracts its device, buffer type, and
//   // tensor spec type, and assigns $chain... the type !mgp.chain<$device>.
//   custom<TensorTypeToDeviceBufferSpec>(
//     type($tensor)?, $device, type($buffer), type($tensorSpec),
//     type($chain)...
//   )
//
//   // Parses a function type consisting of one input buffer and N output
//   // buffer types, all sharing the same device assigned to $device, assigns
//   // $outBufferRawSizes the raw sizes from the $outBuffer types, and assigns
//   // $chain the type !mgp.chain<$device>.
//   custom<BufferTypesToBufferRawSizesDeviceOutBuffers>(
//     type($inBuffer), $outBufferRawSizes,
//     $device, type($outBuffers), type($chain)...
//   )
//
//   // Parses a function type consisting of one input buffer and one output
//   // buffer type, assigns their devices to $srcDevice and $dstDevice,
//   // and assigns $inChain the type !mgp.chain<$srcDevice>.
//   custom<SrcDstBufferTypesToDevices>(
//     type($srcBuffer), type($dstBuffer),
//     $srcDevice, $dstDevice, type($inChain)
//   )
//
//   // Parses a function type consisting of N input buffers and one output
//   // buffer type, assigns their (consistent!) device to $device, and
//   // assigns $chain the type !mgp.chain<$device>.
//   custom<BufferTypesToDevice>(type($inBuffers), type($outBuffer),
//                               $device, type($inChain))
//
//   // Parses a attribute supporting the ElementsAttr interface, and assigns
//   // its alignment to rawAlign.
//   custom<ElementsAttrWithAlign>(value, rawAlign)
//
//===----------------------------------------------------------------------===//

class MGP_DeviceOp<string mnemonic, list<Trait> traits = []> :
    MGP_Op<
      mnemonic,
      [
        DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,
        DeclareOpInterfaceMethods<MGP_HasDeviceOpInterface>,
      ] # traits
    >;

def MGP_OptDeviceRefAttr :
  DefaultValuedAttr<M_DeviceRefAttr, "kCPULabel, 0">;



//===----------------------------------------------------------------------===//
// DeviceContextProfileStartOp
//===----------------------------------------------------------------------===//

def DeviceContextProfileStartOp : MGP_DeviceOp<"device.context.profile.start"> {
  let summary = "Shim to tag the start of a device context profiling run";
  let description = [{This is temporary op. to enable onnx profiling - Should use max-driver for this in future}];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aDeviceRuntimeSlot,   // 0
                       MGP_OptDeviceRefAttr:$bDevice,  // 1
                       StrAttr:$cTag,
                       StrAttr:$dFilePath); // 2
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` $theContext `[` $aDeviceRuntimeSlot `]` ( `device` $bDevice^ )?  attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    uint64_t DeviceContextProfileStartOp::getSlot() { return getADeviceRuntimeSlot(); }

    M::DeviceRefAttr DeviceContextProfileStartOp::getDevice() { return getBDevice(); }
  }];
}

def DeviceContextProfileEndOp : MGP_DeviceOp<"device.context.profile.end"> {
  let summary = "Shim to tag the start of a device context profiling run";
  let description = [{This is temporary op. to enable onnx profiling - Should use max-driver for this in future}];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aDeviceRuntimeSlot,   // 0
                       MGP_OptDeviceRefAttr:$bDevice,  // 1
                       StrAttr:$cTag,
                       StrAttr:$dFilePath); // 2
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` $theContext `[` $aDeviceRuntimeSlot `]` ( `device` $bDevice^ )?  attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{
    uint64_t DeviceContextProfileEndOp::getSlot() { return getADeviceRuntimeSlot(); }

    M::DeviceRefAttr DeviceContextProfileEndOp::getDevice() { return getBDevice(); }
  }];
}


//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

def ModuleLoadOp : MGP_DeviceOp<"module.load"> {
  let summary = "Loads functions and constants for device from compiled module.";
  let description = [{
    Loads functions and constants from a compiled module encoded into a byte
    array (or, for unit testing only, a multi-line string). The module is
    assigned the next available module index (starting at 0) for the device,
    and each function and constant is assigned the next available function or
    constant index within that module (starting at 0). The functions and
    constants will be associated with the runtime established in the given
    `!mgp.context` slot. A functions may be referenced by
    (module index, function index) in `mgp.generic.execute`, and constant
    by (module index, constant index) in `mgp.buffer.contant.load`.

    There is no universal representation for device modules in the host C++
    runtime, each device type may use its own representation.

    Note that this operation accepts and returns `!mef.chain`s since it is
    part of device setup on the host.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      // Load a CUDA PTX or cubin module with given exported kernel and
      // constant names. The CUDA module will be associated with the CUDA
      // runtime established in slot 5, which should be for the CUDA GPU with
      // virtual device id 3.
      %ch1 = mgp.module.load %ch0, %ctx[5][1],
               dense_resource<label> : !M.aligned_bytes<1000, align 16>,
               ["kernel1", "kernel2"], ["const1", "const2"]
               device <"cuda", 3>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,                                   // 0
                       AnyAttrOf<[M_MultiLineStringAttr, ElementsAttr]>:$bImage, // 1
                       M_MultiLineStringAttr:$cFunctionSymbols,                  // 2
                       M_MultiLineStringAttr:$dConstantSymbols,                  // 3
                       MGP_OptDeviceRefAttr:$eDevice,                            // 4
                       UI64Attr:$fModuleIndex,                                   // 5
                       DefaultValuedAttr<BoolAttr, "false">:$gDebuginfo          // 6
                       );
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `[` $fModuleIndex `]` `,` $bImage `,`
    $cFunctionSymbols `,` $dConstantSymbols
    ( `device` $eDevice^ )? ( `debuginfo` $gDebuginfo^ )? attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ModuleLoadOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr ModuleLoadOp::getDevice() { return getEDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// ModuleLoadPlaceholder
//===----------------------------------------------------------------------===//

def ModuleLoadPlaceholderOp : MGP_Op<"module.load.placeholder"> {
  let summary = "Placeholder for the module load";
  let description = [{
    A placeholder op to provide the chain for the module ops / runtimes.
    This op is added just before lowering kernel ABI to provide a chain
    to link the compiled execute to the module load op in init.

    When we create the load module op we can remove it.
  }];
  let results = (outs ChainType: $outChain);
}

//===----------------------------------------------------------------------===//
// BufferConstantExternalOp
//===----------------------------------------------------------------------===//

def BufferConstantExternalOp : MGP_DeviceOp<"buffer.constant.external", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Initializes a buffer from an external constant.";
  let description = [{
    Returns an `!mgp.buffer` on the given device that is initialized at
    runtime.
    The buffer is backed by a pointer retrieved by its name from a weights
    registry, which is simply a mapping from names to pointers.
    At runtime, the retrieved pointer must respect the alignment set on the op.
    Furthermore, like mgp.buffer.constant, the lifetime of the returned buffer
    is untracked and assumed to outlive graph execution.

    Example:
    ```mlir
    %weights : !mgp.opaque<"weights_registry">
    %ch : !mef.chain
    // Return a buffer corresponding to the named weight "foo" in the
    // weights registry.
    %buf = mgp.buffer.constant.external
      %ch, %ctx[0], %weights["foo", 16384, align 16] :
      !mgp.opaque<"weights_registry"> -> !mgp.buffer<16384>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,                      // 0
                       MGP_OpaqueType:$weightsRegistry,                  // 1
                       UI64Attr:$aRuntimeSlot,                           // 2
                       StrAttr:$bName,                                   // 3
                       UI64Attr:$cSize,                                  // 4
                       UI64Attr:$dAlign,                                 // 5
                       MGP_OptDeviceRefAttr:$eDevice);                   // 6
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot  `]` `,` $weightsRegistry
    `[` $bName `,`$cSize `,` `align` $dAlign `]`
    ( `device` $eDevice^ )? attr-dict `:` qualified(type($weightsRegistry))
    `->` qualified(type($outBuffer))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let extraClassDefinition = [{
    uint64_t BufferConstantExternalOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferConstantExternalOp::getDevice() { return getEDevice(); }

    bool BufferConstantExternalOp::isTrivial() { return false; }
  }];
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MGP_DeviceOp<"buffer.constant", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Allocates and initializes a buffer from a host attribute.";
  let description = [{
    Returns an `!mgp.buffer` on the given device initialized with the byte
    array in the given attribute (which will be loaded from the MEFFile holding
    the compiled model). This may require a host-to-device copy, which may
    be asynchronous.

    Example:
    ```mlir
      %ctx : !mgp.context
      // Returns a buffer on the CUDA GPU with virtual device id 3, whose
      // runtime should have been established in slot 5, which has been
      // initialized with the bytes in the dense resource with 'label'.
      %buf = mgp.buffer.constant %ch, %ctx[5],
               dense_resource<label>:!M.aligned_bytes<256, align 16>
               device <"cuda", 3>
      %buf : !mgp.buffer<256, <"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,                           // 0
                       ElementsAttr:$bValue,                             // 1
                       UI64Attr:$cRawAlign,                              // 2
                       MGP_OptDeviceRefAttr:$dDevice,                    // 3
                       DefaultValuedAttr<BoolAttr, "true">:$eIsTrivial); // 4
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot  `]` `,`
    custom<ElementsAttrWithAlign>($bValue, $cRawAlign)
    ( `device` $dDevice^ )? ( `trivial` $eIsTrivial^ )? attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$inChain, "mlir::Value":$theContext,
                   "uint64_t":$aRuntimeSlot, "mlir::ElementsAttr":$bValue,
                   "M::DeviceRefAttr":$dDevice, "bool":$eIsTrivial)>
  ];

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let extraClassDefinition = [{
    uint64_t BufferConstantOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferConstantOp::getDevice() { return getDDevice(); }

    bool BufferConstantOp::isTrivial() { return getEIsTrivial(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//

def BufferAllocOp : MGP_DeviceOp<"buffer.alloc", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
    DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
]> {
  let summary = "Allocates a buffer with dynamically provided byte size on "
                "given device.";
  let description = [{
    One of a family of 6 buffer allocation primitives.

    Allocates (possibly asynchronously) an uninitialized buffer from a device's
    default memory pool with given byte size, and optional statically known
    alignment. The allocation uses the device-appropriate runtime established
    in the given `!mgp.context` slot. Not all devices support the notion of
    alignment.

    The buffer will be freed when the last reference to it is destroyed.

    Example:
    ```mlir
      %ch : !mgp.chain<<"cuda", 3>>
      %ctx : !mgp.context
      %size : index
      // Allocates %size bytes using the CUDA runtime from slot 5, which
      // should correspond to the CUDA GPU with virtual device id 3.
      %buf = mgp.buffer.alloc %ch, %ctx[5], %size : !mgp.buffer<, <"cuda", 3>>
      %buf : !mgp.buffer<, <"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       Index:$runtimeSize,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aRuntimeSlot,                                 // 0
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign, // 1
                       MGP_OptDeviceRefAttr:$cDevice);                         // 2
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aRuntimeSlot `]` `,` $runtimeSize `,` $deviceContext
    ( `,` `align` $bRawAlign^ )? attr-dict
    `:` custom<BufferTypeToDevice>(
          type($outBuffer), $cDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferAllocOp::getSlot() { return getARuntimeSlot(); }

    uint64_t BufferAllocOp::getRawAlign() { return getBRawAlign(); }

    M::DeviceRefAttr BufferAllocOp::getDevice() { return getCDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

def BufferCacheOp : MGP_DeviceOp<"buffer.cache"> {
  let summary = "Moves an `!mgp.buffer` into a dedicated state slot.";
  let description = [{
    Stores the given `!mgp.buffer` into the state slot at the given index.
    It may be later retrieved by `mgp.buffer.get_cached`. Depending on the
    buffer's device, any computation associated with the buffer may need to be
    'finished' before the store is executed.

    If the buffer is immutable, or this op holds the only reference to the
    buffer, then it will be moved into the cache. Otherwise a deep copy of
    the buffer will be taken.

    Note this op accepts and returns `!mef.chain`s since the buffer cache is
    held on the host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf : !mgp.buffer<, <"cuda", 3>>
    %ch0 : !mef.chain
    // Cache the given buffer in slot index 42, using the CUDA runtime
    // established in slot 5, which should correspond to the CUDA GPU with
    // virtual id 3.
    %ch1 = mgp.buffer.cache %ch0, %ctx[5, 42], %buf
             : !mgp.buffer<256, <"cuda", 3>>
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       UI64Attr:$aRuntimeSlot,         // 0
                       UI64Attr:$bBufferSlot,          // 1
                       MGP_OptDeviceRefAttr:$cDevice); // 2
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `,` $bBufferSlot `]` `,`
    $buffer attr-dict
    `:` custom<BufferTypeToDevice>(type($buffer), $cDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferCacheOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferCacheOp::getDevice() { return getCDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

def BufferGetCachedOp : MGP_DeviceOp<"buffer.get_cached", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Retrieves an `!mgp.buffer` from a dedicated state slot.";
  let description = [{
    Returns (a borrow of) the `!mgp.buffer` in the state slot with the given
    index. The buffer should have already been cached by `mgp.buffer.cache`.
    Returns an error if the cached buffer's size (if any) or device do not
    agree with the `!mgp.buffer` type annotation.

    Note this op is gated by a `!mef.chain` since the cache is held on the
    host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch0 : !mef.chain
    // Returns the buffer cached in slot index 42, using the CUDA runtime
    // established in slot 5, which should correspond to the CUDA GPU with
    // virtual id 3.
    %buf = mgp.buffer.get_cached %ch0, %ctx[5, 42] : !mgp.buffer<256, <"cuda", 3>>
    %buf : !mgp.buffer<256, <"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,         // 0
                       UI64Attr:$bBufferSlot,          // 1
                       UI64Attr:$cRawSize,             // 2
                       MGP_OptDeviceRefAttr:$dDevice); // 3
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `,` $bBufferSlot `]` attr-dict
    `:` custom<BufferTypeToRawSizeDevice>(
          type($outBuffer), $cRawSize, $dDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferGetCachedOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferGetCachedOp::getDevice() { return getDDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferFreeOp
//===----------------------------------------------------------------------===//

def BufferFreeOp : MGP_Op<"buffer.free"> {
  let summary = "No-op used to indicate when a buffer is freed.";
  let description = [{
    Indicate when a buffer may be freed.

    This operation does nothing and has no primitive backing it. It only exists
    to make buffer lifetimes explicit. Buffers are refcounted in the runtime and
    freed when necessary.

    Example:
    ```mlir
      %ctx : !mgp.context
      %ch1 : !mgp.chain
      %buf0 : !mgp.buffer<4>
      %ch2 = mgp.buffer.free %ch1, %buf0 : !mgp.buffer<4>
    ```
  }];

  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
    `:` custom<BufferTypeToChain>(type($buffer), type($inChain))
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$buffer);
  let results = (outs ChainType:$outChain);
}

//===----------------------------------------------------------------------===//
// BufferHostToDeviceOp
//===----------------------------------------------------------------------===//

def BufferHostToDeviceOp : MGP_DeviceOp<"buffer.host_to_device", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Copies a host buffer into a non-host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of an `!mgp.buffer` on the
    host device to an `!mgp.buffer` on a non-host device. The two buffers must
    agree on their runtime size.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed on the device.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %host_buf : !mgp.buffer<>
      %other_buf : !mgp.buffer<, <"cuda", 3>>
      %ch1 = mgp.buffer.host_to_device %ch0, %ctx[4, 5], %host_buf, %other_buf
               : !mgp.buffer<> -> !mgp.buffer<, <"cuda", 3>>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$hostBuffer,
                       MGP_DeviceBufferType:$otherBuffer,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aHostRuntimeSlot,          // 0
                       UI64Attr:$bOtherRuntimeSlot,         // 1
                       MGP_OptDeviceRefAttr:$cHostDevice,   // 2
                       MGP_OptDeviceRefAttr:$dOtherDevice); // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aHostRuntimeSlot `,` $bOtherRuntimeSlot `]`
    `,` $hostBuffer `,` $otherBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($hostBuffer), type($otherBuffer),
          $cHostDevice, $dOtherDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferHostToDeviceOp::getSlot() { return getBOtherRuntimeSlot(); }

    M::DeviceRefAttr BufferHostToDeviceOp::getDevice() { return getDOtherDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferDeviceToHostOp
//===----------------------------------------------------------------------===//

def BufferDeviceToHostOp : MGP_DeviceOp<"buffer.device_to_host", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Copies a non-host buffer into a host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of an `!mgp.buffer` on a
    non-host device to an `!mgp.buffer` on the host device. The two buffers
    must agree on their runtime size.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed. The device must be synchronized with
    `mgp.sync` before the host buffer can be read.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %other_buf : !mgp.buffer<, <"cuda", 3>>
      %host_buf : !mgp.buffer<>
      %ch1 = mgp.buffer.device_to_host %ch0, %ctx[5, 4], %other_buf, %host_buf
               : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$otherBuffer,
                       MGP_DeviceBufferType:$hostBuffer,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aOtherRuntimeSlot,        // 0
                       UI64Attr:$bHostRuntimeSlot,         // 1
                       MGP_OptDeviceRefAttr:$cOtherDevice, // 2
                       MGP_OptDeviceRefAttr:$dHostDevice); // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aOtherRuntimeSlot `,` $bHostRuntimeSlot `]`
    `,` $otherBuffer `,` $hostBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($otherBuffer), type($hostBuffer),
          $cOtherDevice, $dHostDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferDeviceToHostOp::getSlot() { return getBHostRuntimeSlot(); }

    M::DeviceRefAttr BufferDeviceToHostOp::getDevice() { return getCOtherDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferDeviceToDeviceOp
//===----------------------------------------------------------------------===//

def BufferDeviceToDeviceOp : MGP_DeviceOp<"buffer.device_to_device", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Copies a non-host buffer into another non-host buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of one `!mgp.buffer` into
    another `!mgp.buffer`. The two buffers must have the same device label,
    but may be on distinct virtual devices.

    The output chain is ready when the copy has been scheduled, but not
    necessarily when it has completed on the device.

    Example:
    ```mlir
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %src_buf : !mgp.buffer<, <"cuda", 3>>
      %dst_buf : !mgp.buffer<, <"cuda", 1>>
      %ch1 = mgp.buffer.device_to_device %ch0, %ctx[5, 3], %src_buf, %dst_buf
               : !mgp.buffer<, <"cuda", 3>> -> !mgp.buffer<, <"cuda", 1>>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$srcBuffer,
                       MGP_DeviceBufferType:$dstBuffer,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aSrcRuntimeSlot,     // 0
                       UI64Attr:$bDstRuntimeSlot,     // 1
                       M_DeviceRefAttr:$cSrcDevice,   // 2
                       M_DeviceRefAttr:$dDstDevice);  // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aSrcRuntimeSlot `,` $bDstRuntimeSlot `]` `,`
    $srcBuffer `,` $dstBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($srcBuffer), type($dstBuffer),
          $cSrcDevice, $dDstDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferDeviceToDeviceOp::getSlot() { return getASrcRuntimeSlot(); }

    M::DeviceRefAttr BufferDeviceToDeviceOp::getDevice() { return getCSrcDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferUnsizeOp
//===----------------------------------------------------------------------===//

// NOTE: This op does not need a device attribute or runtime slot, and is a
//       no-op at runtime.
def BufferUnsizeOp : MGP_Op<"buffer.unsize", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    // The result is a shallow copy of the input.
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAliasOpInterface>,
]> {
  let summary = "Forget the size of a statically sized buffer.";
  let description = [{
    Returns a dynamically sized shallow copy of the statically-sized input
    buffer.

    Note that no copying is done, and the result `AsyncValueRef<GML::DeviceRef>`
    is simply a move of the input.

    Example:
    ```mlir
    %in_buf : !mgp.buffer<256, <"cuda", 3>>
    %out_buf = mgp.buffer.unsize %in_buf : !mgp.buffer<256, <"cuda", 3>>
    %out_buf : !mgp.buffer<, <"cuda", 3>>
    ```
  }];

  let arguments = (ins MGP_DeviceBufferType:$inBuffer);
  let results = (outs  MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{ $inBuffer attr-dict `:` qualified(type($inBuffer)) }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferResizeOp
//===----------------------------------------------------------------------===//

// NOTE: This op does not need a runtime slot. However it does need a device
//       attribute so as to dispatch to the device-appropriate assertion op.
def BufferResizeOp : MGP_Op<"buffer.resize", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    // The result is a shallow copy of the input.
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
    DeclareOpInterfaceMethods<MGP_HasDeviceOpInterface>,
    DeclareOpInterfaceMethods<MGP_BufferAliasOpInterface>,
]> {
  let summary = "Assert the size of a dynamically sized buffer.";
  let description = [{
    Returns a statically sized shallow copy of the dynamically-sized input
    buffer. An error will be returned if the input buffer is not of the
    expected size at runtime.

    Note that no copying is done, and the result `AsyncValueRef<GML::DeviceRef>`
    is simply a move of the input.

    Example:
    ```mlir
    %in_buf : !mgp.buffer<, <"cuda", 3>>
    %out_buf = mgp.buffer.resize %in_buf : !mgp.buffer<256, <"cuda", 3>>
    %out_buf : !mgp.buffer<256, <"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceBufferType:$inBuffer,
                       UI64Attr:$aSize,                // 0
                       MGP_OptDeviceRefAttr:$bDevice); // 1
  let results = (outs  MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inBuffer attr-dict
    `:` custom<StaticBufferTypeToSizeDeviceBuffer>(
          type($outBuffer), $aSize, $bDevice, type($inBuffer))
   }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    M::DeviceRefAttr BufferResizeOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferConcatOp
//===----------------------------------------------------------------------===//

def BufferConcatOp : MGP_DeviceOp<"buffer.concat", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Concatenates a sequence of input buffers into an output "
                "buffer.";
  let description = [{
    Concatenates a sequence of input buffers into an output buffer. If only a
    single operand is given then mimics an ordinary buffer copy. The the output
    buffer and all input buffers have statically known sizes then they are
    checked for consistency at compile time. A runtime error will be returned
    if the actual runtime sizes of input and output buffers are not consistent.

    Example:
    ```mlir
    %ctx : !mgp.context
    %in_ch : !mgp.chain<<"cuda", 3>>
    %out_buf : !mgp.buffer<256, <"cuda", 3>>
    %in0_buf : !mgp.buffer<128, <"cuda", 3>>
    %in1_buf : !mgp.buffer<, <"cuda", 3>>
    %in2_buf : !mgp.buffer<, <"cuda", 3>>
    %out_ch = mgp.buffer.concat %in_ch, %ctx[5], %out_buf, %in0_buf, %in1_buf, %in2_buf
              : (!mgp.buffer<128, <"cuda", 3>>,
                 !mgp.buffer<, <"cuda", 3>>,
                 !mgp.buffer<, <"cuda", 3>>) -> !mgp.buffer<256, <"cuda", 3>>
    %out_ch : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$outBuffer,
                       Variadic<MGP_DeviceBufferType>:$inBuffers,
                       UI64Attr:$aRuntimeSlot,         // 0
                       MGP_OptDeviceRefAttr:$bDevice); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,`  $theContext `[` $aRuntimeSlot `]` `,` $outBuffer `,`
    $inBuffers attr-dict
    `:` custom<BufferTypesToDevice>(type($inBuffers), type($outBuffer),
                                    $bDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferConcatOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferConcatOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferPlanOp
//===----------------------------------------------------------------------===//
def BufferPlanOp : MGP_Op<"buffer.plan", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Runtime memory planning for buffers";
  let description = [{
    Given the runtimeSize and lifetime information for a list of allocations,
    returns the high watermark size and offsets for each allocation.

    Alloc lifetimes are reprsented as the min/max pre/postorder indices of the
    uses of every allocation in the "dependency tree". An allocation A can reuse
    an allocation B if A.maxPre < B.minPre && A.minPost > B.maxPost.
    See MGPBufferLifetimeAnalysis.h for details.
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    Variadic<Index>:$runtimeSizes,
    UI64ArrayElementsAttr:$aAlignments,
    UI64ArrayElementsAttr:$bMinPre,
    UI64ArrayElementsAttr:$cMinPost,
    UI64ArrayElementsAttr:$dMaxPre,
    UI64ArrayElementsAttr:$eMaxPost
  );

  let results = (outs
    Index:$highWatermark,
    Variadic<Index>:$offsets
  );

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferSliceOp
//===----------------------------------------------------------------------===//

def BufferSliceOp : MGP_Op<"buffer.slice", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Borrows a slice of parent buffer on device with dynamically "
                "known offset and size.";
  let description = [{
    Returns a borrowed buffer whose underlying storage points into a section of
    the input buffer. The offset and size of the section is given by
    the offset and size arguments.

    Note that no copying is done, and the result buffer merely borrows from
    the (offset into) the input buffer.

    Example:
    ```mlir
    %ch:  !mgp.chain
    %ctx : !mgp.context
    %in_buf : !mgp.buffer<24>
    %offset : index
    %size : index
    %out_buf =
        mgp.buffer.slice %in_buf, %offset, %size : !mgp.buffer<24>
    %out_buf : !mgp.buffer<>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceBufferType:$inBuffer,
                       Index:$offset,
                       Index:$size
                       );

  let results = (outs MGP_DeviceBufferType:$outBuffer);

  let assemblyFormat = [{
    $inBuffer `,` $offset `,` $size attr-dict
      `:` type($inBuffer)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BufferToIndexOp
//===----------------------------------------------------------------------===//

def BufferToIndexOp : MGP_Op<"buffer.to_index", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Returns contents of the buffer to an `index`";
  let description = [{
    Returns contents of a 4- or 8-byte buffer as a 64-bit `index`.

    Sign extension is performed when upcasting from a 32-bit integer.

    Example:
    ```mlir
    %0 = mgp.buffer.to_index %ch0, %buf : !mgp.buffer<8>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$inBuffer);

  let results = (outs Index:$out);

  let assemblyFormat = [{
     $inChain `,` $inBuffer attr-dict `:` type($inBuffer)
  }];
}

//===----------------------------------------------------------------------===//
// BufferToBoolOp
//===----------------------------------------------------------------------===//

def BufferToBoolOp : MGP_DeviceOp<"buffer.to_bool"> {
  let summary = "Returns contents of a 1-byte buffer as an i1.";
  let description = [{
    Returns contents of a one byte buffer as an i1.

    Example:
    ```mlir
    %0 = mgp.to_bool %ch0, %ctx[5], %buf : !mgp.buffer<1, <"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       UI64Attr:$aRuntimeSlot,          // 0
                       MGP_OptDeviceRefAttr:$bDevice);  // 1
  let results = (outs I1:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $buffer attr-dict
      `:` custom<BufferTypeToDevice>(type($buffer), $bDevice, type($inChain))
  }];

  let extraClassDefinition = [{
    uint64_t BufferToBoolOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferToBoolOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferSetWithIndexOp
//===----------------------------------------------------------------------===//

def BufferSetWithIndexOp : MGP_DeviceOp<"buffer.set_with_index", [
    // TODO: #37615
    /* DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>, */
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Set a buffer with the values from index inputs.";
  let description = [{
    The runtime index values will be sign extended or truncated based on the
    given buffer's size. Therefore only buffer sizes of 4 and 8 times the number
    of index inputs are supported by the primitive.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf0 : !mgp.buffer<4, <"cuda", 3>>
    %idx0 : index
    %ch0 = mgp.buffer.set_with_index %ctx[5], %buf0, %idx0 : !mgp.buffer<4, <"cuda", 3>>
    %ch0 : !mgp.chain<<"cuda", 3>>

    %idx1 : index
    %idx2 : index
    %buf1 : !mgp.buffer<24, <"cuda", 3>>
    %ch1 = mgp.buffer.set_with_index %ctx[5], %buf1, %idx0, %idx1, %idx2 : !mgp.buffer<24, <"cuda", 3>>
    %ch1 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       Variadic<Index>:$values,
                       UI64Attr:$aRuntimeSlot,         // 0
                       MGP_OptDeviceRefAttr:$bDevice); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $theContext `[` $aRuntimeSlot `]` `,`  $buffer ( `,` $values^ )? attr-dict
    `:` custom<BufferTypeToDevice>(type($buffer), $bDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferSetWithIndexOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferSetWithIndexOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

def DebugTensorPrintOp : MGP_DeviceOp<"debug.tensor.print"> {
  let summary = "Prints a debug representation of an tensor. For debugging "
                "and testing only.";
  let description = [{
    Prints a debug representation of an tensor, given as a pair of an
    `!mgp.buffer` and a `!mgp.tensor_spec`. If a label attribute is supplied
    the tensor contents are printed with that label. Otherwise just the tensor
    metadata is printed. Printing the tensor contents may require an implicit
    device-to-host copy.

    For debugging and testing only.

    Note this op accepts and returns `!mef.chain`s since printing is managed
    by the host.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %buf: !mgp.buffer<2048, <"cuda", 3>>
    %spec: !mgp.tensor_spec<8x64xf32>
    %ch1 = mgp.debug.tensor.print %ch0, %ctx[5], %buf, %spec, "label"
           : !mgp.tensor<8x64xf32, <"cuda", 3>>
    %ch1: !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       MGP_TensorSpecType:$spec,
                       UI64Attr:$aRuntimeSlot,                       // 0
                       DefaultValuedAttr<StrAttr, "\"\"">: $bLabel,  // 1
                       MGP_OptDeviceRefAttr:$cDevice);               // 2
  let results = (outs ChainType: $outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $buffer `,` $spec
    ( `,` $bLabel^ )? attr-dict
    `:` custom<TensorTypeToDeviceBufferSpec>(
          $cDevice, type($buffer), type($spec))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t DebugTensorPrintOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr DebugTensorPrintOp::getDevice() { return getCDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// SyncOp
//===----------------------------------------------------------------------===//

def MGP_SyncOp : MGP_DeviceOp<"sync",
  [DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>]
> {
  let summary = "Schedules a device specific sync";
  let description = [{
    Returns a ready chain once the device has synchronized. In
    cpu this is a nop as the sync happens with the chain, on
    gpu this records a cuda event.
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_OpaqueType:$deviceContext,
                       UI64Attr:$aRuntimeSlot,                       // 0
                       MGP_OptDeviceRefAttr:$bDevice);               // 1
  let results = (outs ChainType: $outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    uint64_t SyncOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr SyncOp::getDevice() { return getBDevice(); }
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPDEVICEOPS_TD
