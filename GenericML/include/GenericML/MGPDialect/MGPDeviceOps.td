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
//   // Parses an #M.device_ref into $device, and assigns $chain... the
//   // type !mgp.chain<$device>.
//   custom<DeviceRef>($device, type($chain)...)
//
//   // Parses optionally type-annotated operands into $gatingOperands, using
//   // the default type !mgp.chain<$device>.
//   custom<ChainCreateParenOperandList>(
//     ref($device), $gatingOperands, type($gatingOperands)
//   )
//
//   // Parses an !mgp.buffer type and extracts its raw size and device
//   // attributes, and assigns $chain... the type !mgp.chain<$device>.
//   custom<BufferTypeToRawSizeDevice>(type($buffer)?, $rawSize, $device, type($chain)...)
//
//   // Parses a statically sized !mgp.buffer type and extracts its size and
//   // device attributes, and assigns $chain... the type !mgp.chain<$device>.
//   custom<StaticBufferTypeToSizeDevice>(type($buffer), $size, $device, type($chain)...)
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
//   // Parses an !mgp.tensor type and extracts its device and tensor shape.
//   custom<TensorTypeToDeviceShape>(type($tensor), $device, type($outSpec))
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
//   // and assigns $inChain the type !mgp.chain<$srcDevice> and $outChain
//   // the type !mgp.chain<$dstDevice>.
//   custom<SrcDstBufferTypesToDevices>(
//     type($srcBuffer), type($dstBuffer),
//     $srcDevice, $dstDevice, type($inChain), type($outChain)
//   )
//
//   // Parses a function type consisting of one input buffer and one output
//   // buffer type, checks their devices agree, assigns that device to $device,
//   // and assigns $inChain and $outChain the type !mgp.chain<$device>.
//   custom<SrcDstBufferTypesToSameDevice>(
//     type($srcBuffer), type($dstBuffer), $device,
//     type($inChain), type($outChain)
//   )
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

//===----------------------------------------------------------------------===//
// RuntimeCreateOp
//===----------------------------------------------------------------------===//

def RuntimeCreateOp : MGP_DeviceOp<"runtime.create"> {
  let summary = "Creates a runtime to coordinate execution on a particular "
                "device.";
  let description = [{
    Creates a device-specific runtime within an `!mgp.context` slot appropriate
    for the `#M.device_ref` attribute (the 'runtime slot'). The actual
    'physical' device to use is determined by a combination of runtime
    configuration options and the `DeviceSpecMap` produced by the already
    executed `mgp.check_runtime` op which has been stored in the given
    'device map slot' index.

    All subsequent operations for this device must reference this runtime same
    slot.

    There is no universal representation for device runtimes in the host C++
    runtime. Depending on device the op may accept additional attributes.

    Note that this operation accepts and returns `!mef.chain`s since it is
    part of device setup on the host.

    Example:
    ```mlir
      // Establish a CUDA runtime in slot 5, using the device map established
      // by an earlier mgp.check_runtime op in slot 2 to determine which
      // physical GPU to use. The physical GPU will be associated with the
      // virtual device id 3.
      %ch0 : !mef.chain
      %ctx : !mgp.context
      %ch1 = mgp.runtime.create %ch0, %ctx[5, 2], <"cuda", 3>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,        // 1
                       UI64Attr:$bDeviceSpecMapSlot,  // 2
                       M_DeviceRefAttr:$cDevice);     // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `,` $bDeviceSpecMapSlot `]` `,`
    $cDevice attr-dict
  }];

  let extraClassDefinition = [{
    uint64_t RuntimeCreateOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr RuntimeCreateOp::getDevice() { return getCDevice(); }
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
    (module index, function index) in `mgp.kernel.execute`, and constant
    by (module index, constant index) in `mgp.buffer.contant.load`.

    There is no universal representation for device modules in the host C++
    runtime, each device type may use it's own representation.

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
      %ch1 = cuda.module.load %ch0, %ctx[5],
               dense_resource<label> : !M.aligned_bytes<1000, align 16>,
               ["kernel1", "kernel2"], ["const1", "const2"],
               <"cuda", 3>
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
                       M_DeviceRefAttr:$eDevice);                                // 4
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $bImage `,`
    $cFunctionSymbols `,` $dConstantSymbols `,` $eDevice attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ModuleLoadOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr ModuleLoadOp::getDevice() { return getEDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

def ChainCreateOp : MGP_DeviceOp<"chain.create", [
    Pure,
    DeclareOpInterfaceMethods<GRT_ChainCreateOpInterface>,
    DeclareOpInterfaceMethods<GRT_GatedOnChainOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Returns an `!mgp.chain` gated by zero or more operands.";
  let description = [{
    Returns an `!mgp.chain` for a device which is 'gated' by zero or more
    operands. The gating operands may be of any type, however all device-generic
    operands must be on either the 'host' or 'other' device.

    Some devices may support the notion of a chain being 'finished' separately
    from it being 'ready'. See `!mgp.chain` for more details. For these
    devices, additional synchronization may be required on gating operands
    which are on the other device so that the result chain is only 'finished'
    when all 'other' device operands are 'finished'. Otherwise, this op is
    equivalent to `grt.chain.create`.

    It is valid for the 'other' and 'host' devices to be the same if
    creating a chain for the host device.

    Example:
    ```mlir
      %ctx : !mgp.context
      %ch0 : !mgp.chain<<"cuda", 3>>
      %ch1 : !mgp.chain<<"cpu", 0>>
      %ch2 : !mef.chain
      %val : !mgp.buffer<<"cuda", 3>, 64>
      // Returns a chain which is ready when %ch0, %ch1, %ch2 and %val are
      // ready, but which is 'finished' only when %ch0 and %val are 'finished'.
      // Uses the CUDA runtime established in slot 5, which must be for CUDA
      // virtual device id 3.
      %ch3 = mgp.chain.create.other %ctx[5], host <"cpu", 0>, other <"cuda", 3>,
             (
               %ch0,
               %ch1: !mgp.chain<<"cpu", 0>>,
               %ch2: !mef.chain,
               %val: !mgp.buffer<<"cuda", 3>, 64>
             )
      %ch3 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_ContextType:$theContext,
                       Variadic<AnyType>:$gatingOperands,
                       UI64Attr:$aOtherRuntimeSlot,    // 0
                       M_DeviceRefAttr:$bHostDevice,   // 1
                       M_DeviceRefAttr:$cOtherDevice); // 2
  let results = (outs MGP_DeviceChainType:$outChain);
  let assemblyFormat = [{
    $theContext `[` $aOtherRuntimeSlot `]` `,`
    `host` $bHostDevice `,`
    `other` custom<DeviceRef>($cOtherDevice, type($outChain)) `,`
    custom<ChainCreateParenOperandList>(
      ref($cOtherDevice), $gatingOperands, type($gatingOperands))
    attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ChainCreateOp::getSlot() { return getAOtherRuntimeSlot(); }

    M::DeviceRefAttr ChainCreateOp::getDevice() { return getCOtherDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// ChainDeviceToHostOp
//===----------------------------------------------------------------------===//

def ChainDeviceToHostOp : MGP_DeviceOp<"chain.device_to_host", [
    Pure,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Returns a `!mef.chain` which is ready when the given device "
                "`!mgp.chain` is 'finished'.";
  let description = [{
    Some non-host devices may need to represent the notion of a computation
    being 'finished', see `!mgp.chain` for more details. This op returns a
    host `!mef.chain` which is 'ready' only when the device `!mgp.chain`
    is 'finished'. Generally this may require additional synchronization between
    the non-host device and the host-device, which impose additional latency.

    It is valid for the device to be the host device, in which case this
    op is a no-op.

    Example:
    ```mlir
      %ch0 : !mgp.chain<<"cuda", 3>>
      %ctx : !mgp.context
      // Returns a chain for the host CPU, with appropriate synchronization
      // on chains for the CUDA GPU with virtual device id and runtime
      // established in slot 5. The host runtime has been established in slot 4.
      %ch1 = mgp.chain.device_to_host %ch0, %ctx[5], <"cuda", 3>
      %ch1 : !mef.chain
    ```
  }];

  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,    // 0
                       M_DeviceRefAttr:$bDevice); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,`
    custom<DeviceRef>($bDevice, type($inChain)) attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ChainDeviceToHostOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr ChainDeviceToHostOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// ChainHostToDeviceOp
//===----------------------------------------------------------------------===//

def ChainHostToDeviceOp : MGP_DeviceOp<"chain.host_to_device", [
    Pure,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
]> {
  let summary = "Returns an `!mgp.chain` which is 'finished' when the given "
                "`!mef.chain` is ready.";
  let description = [{
    Some non-host devices may need to represent the notion of a computation
    being 'finished', see `!mgp.chain` for more details. This op returns a
    device `!mgp.chain` which is 'finished' only when the operand `!mef.chain`
    is ready. Generally this does not require any device synchronization.

    It is valid for the device to be the host device, in which case this
    op is a no-op.

    Example:
    ```mlir
      %ctx : !mgp.context
      %ch0 : !mef.chain
      %ch1 = mgp.chain.host_to_device %ch0, %ctx[1], <"cuda", 3>
      %ch1 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,    // 0
                       M_DeviceRefAttr:$bDevice); // 1
  let results = (outs MGP_DeviceChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,`
    custom<DeviceRef>($bDevice, type($outChain)) attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t ChainHostToDeviceOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr ChainHostToDeviceOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferConstantOp
//===----------------------------------------------------------------------===//

def BufferConstantOp : MGP_DeviceOp<"buffer.constant", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
]> {
  let summary = "Allocates and initialized a buffer from a host attribute.";
  let description = [{
    Returns an `!mgp.buffer` on the given device initialized with the byte
    array in the given attribute (which will be loaded from the MEFFile holding
    the compiled model). This may require a host-to-device copy, which may
    be asynchronous.

    See also `mgp.buffer.constant.load` for loading a constant from a
    compiled module for the device.

    Example:
    ```mlir
      %ctx : !mgp.context
      // Returns a buffer on the CUDA GPU with virtual device id 3, whose
      // runtime should have been established in slot 5, which has been
      // initialized with the bytes in the dense resource with 'label'.
      %buf = mgp.buffer.constant %ctx[5],
               dense_resource<label>:!M.aligned_bytes<256, align 16>,
               <"cuda", 3>
      %buf : !mgp.buffer<<"cuda", 3>, 256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,      // 0
                       ElementsAttr:$bValue,        // 1
                       UI64Attr:$cRawAlign,         // 2
                       M_DeviceRefAttr:$dDevice);   // 3
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $theContext `[` $aRuntimeSlot  `]` `,`
    custom<ElementsAttrWithAlign>($bValue, $cRawAlign) `,`
    $dDevice attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$theContext, "uint64_t":$aRuntimeSlot,
                   "mlir::ElementsAttr":$bValue, "M::DeviceRefAttr":$dDevice)>
  ];

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let extraClassDefinition = [{
    uint64_t BufferConstantOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferConstantOp::getDevice() { return getDDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferConstantLoadOp
//===----------------------------------------------------------------------===//

def BufferConstantLoadOp : MGP_DeviceOp<"buffer.constant.load", [
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Retrieves a constant from an already loaded module for device.";
  let description = [{
    Retrieves a buffer from loaded module with given module and constant index.
    Generally speaking the cost of transferring the constant onto the device
    has been paid for when the module was loaded.

    Example:
    ```mlir
      %ctx : !mgp.context
      // Returns buffer on CUDA GPU with virtual device id 3, which is
      // associated with the CUDA runtime at slot 5. The constant is in
      // the module loaded with index 0, and is at constant index 2.
      %buf = mgp.buffer.constant.load %ctx[5], (0, 2) : !mgp.buffer<<"cuda", 3>, 256>
      %buf : !mgp.buffer<<"cuda", 3>, 256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,      // 0
                       UI64Attr:$bModuleIndex,      // 2
                       UI64Attr:$cConstantIndex,    // 3
                       UI64Attr:$dSize,             // 4
                       M_DeviceRefAttr:$eDevice);   // 5
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $theContext `[` $aRuntimeSlot `]` `,`
    `(` $bModuleIndex `,` $cConstantIndex `)` attr-dict
    `:` custom<StaticBufferTypeToSizeDevice>(type($outBuffer), $dSize,  $eDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferConstantLoadOp::isTrivial() { return true; }

    uint64_t BufferConstantLoadOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferConstantLoadOp::getDevice() { return getEDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocStaticOp
//===----------------------------------------------------------------------===//

def BufferAllocStaticOp : MGP_DeviceOp<"buffer.alloc.static", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
    DeclareOpInterfaceMethods<MGP_StaticAllocOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Allocates a buffer with statically known byte size on given device.";
  let description = [{
    One of a family of 4 buffer allocation primitives.

    Allocates (possibly asynchronously) an uninitialized buffer from a device's
    default memory pool with a statically known byte size and optional
    statically known alignment. The allocation uses the device-appropriate
    runtime established in the given `!mgp.context` slot. Not all devices
    support the notion of alignment.

    The buffer will be freed when the last reference to it is destroyed.

    Example:
    ```mlir
      %ch : !mgp.chain<<"cuda", 3>>
      %ctx : !mgp.context
      // Allocates 256 bytes using the CUDA runtime established in slot 5,
      // which should correspond to the the CUDA GPU with virtual device id 3.
      %buf = mgp.buffer.alloc.static %ch, %ctx[5] : !mgp.buffer<<"cuda", 3>, 256>
      %buf : !mgp.buffer<<"cuda", 3>, 256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,                                 // 0
                       UI64Attr:$bSize,                                        // 1
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$cRawAlign, // 2
                       M_DeviceRefAttr:$dDevice);                              // 3
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aRuntimeSlot `]`
    ( `,` `align` $cRawAlign^ )? attr-dict
    `:` custom<StaticBufferTypeToSizeDevice>(
          type($outBuffer), $bSize, $dDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferAllocStaticOp::isTrivial() { return true; }

    uint64_t BufferAllocStaticOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferAllocStaticOp::getDevice() { return getDDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocDynamicOp
//===----------------------------------------------------------------------===//

def BufferAllocDynamicOp : MGP_DeviceOp<"buffer.alloc.dynamic", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface, ["isTrivial"]>,
]> {
  let summary = "Allocates a buffer with dynamically provided byte size on "
                "given device.";
  let description = [{
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
      %buf = mgp.buffer.alloc.dynamic %ch, %ctx[5], %size : !mgp.buffer<<"cuda", 3>>
      %buf : !mgp.buffer<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       Index:$runtimeSize,
                       UI64Attr:$aRuntimeSlot,                                 // 0
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign, // 1
                       M_DeviceRefAttr:$cDevice);                              // 2
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aRuntimeSlot `]` `,` $runtimeSize
    ( `,` `align` $bRawAlign^ )? attr-dict
    `:` custom<DynamicBufferTypeToDevice>(
          type($outBuffer), $cDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferAllocDynamicOp::isTrivial() { return true; }

    uint64_t BufferAllocDynamicOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferAllocDynamicOp::getDevice() { return getCDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferReallocStaticOp
//===----------------------------------------------------------------------===//

def BufferReallocStaticOp : MGP_DeviceOp<"buffer.realloc.static", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
]> {
  let summary = "(Re)allocates or reuses a cached device buffer with statically "
                "known size.";
  let description = [{
    One of a family of 4 buffer allocation primitives.

    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer. The size must be known
    statically.  An optional static alignment constraint may also be given.

    The resulting buffer always borrows from the buffer held by the cache.
    The compiler must respect the following rules to avoid use-after-free
    errors and mutation-via-aliases at runtime:
     1. The 'one shot' rule: For all control flow paths, at most one
        mgp.buffer.realloc.* can execute for the same slot and buffer
        indexes.
        This rule is easy to follow since no two operations in the graph should
        share the same slot, and buffer indexes should correspond to op result
        indexes.
     2. The 'lifetime' rule: The lifetime of the context must exceed the
        lifetime of the returned buffer. This rule is easy to follow since the
        context is held by the Modular runtime M::ExecutableModel, which must
        outlive both all calls to execute and all uses of aliasing results of
        execute.

    Note that though the buffer will be cached, the contents of the buffer need
    not be preserved between calls to `mgp.buffer.realloc.*` in successive
    invocations of the model 'execute' body. This rule allows the runtime
    to avoid some buffer copies when extracting the buffer from a tensor
    which itself was created using a cached buffer. In other words, the cache
    is designed to avoid re-allocating buffers rather than re-calculating the
    contents of buffers.

    Note that alignment constraints are ignored when deciding if a cached buffer
    may be reused.

    Example:
    ```mlir
    %ctx : !mgp.context
    %ch : !mgp.chain<<"cuda", 3>>
    // Reallocate the buffer in slot index 42 and buffer index 2, using the
    // CUDA runtime in slot 5, which should correspond to the CUDA GPU with
    // virtual id 3.
    %buf = mgp.buffer.realloc.static %ch, %ctx[5, 42, 2] : !mgp.buffer<<"cuda", 3>, 64>
    %buf : !mgp.buffer<<"cuda", 3>, 64>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                   MGP_ContextType:$theContext,
                   UI64Attr:$aRuntimeSlot,                                   // 0
                   UI64Attr:$bBufferSlot,                                    // 1
                   UI64Attr:$cBufferIndex,                                   // 2
                   UI64Attr:$dSize,                                          // 3
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$eRawAlign,   // 4
                   M_DeviceRefAttr:$fDevice);                                // 5
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,`
    $theContext `[` $aRuntimeSlot `,` $bBufferSlot `,` $cBufferIndex `]`
    ( `,` `align` $eRawAlign^ )? attr-dict
    `:` custom<StaticBufferTypeToSizeDevice>(
            type($outBuffer), $dSize, $fDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferReallocStaticOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferReallocStaticOp::getDevice() { return getFDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferReallocDynamicOp
//===----------------------------------------------------------------------===//

def BufferReallocDynamicOp : MGP_DeviceOp<"buffer.realloc.dynamic", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
]> {
  let summary = "(Re)allocates or reuses a cached device buffer with dynamically "
                "known size.";
  let description = [{
    One of a family of 4 buffer allocation primitives.

    This operation has access to a buffer cache entry in the context given
    by the slot and buffer indexes.

    If a buffer of matching size already exists in the cache then it is
    returned directly. Otherwise a new buffer is allocated and the cache is
    updated, deleting any previously cached buffer. The size is given by
    an operand. An optional static alignment constraint may also be given.

    See `mgp.buffer.realloc.static` for additional details and restrictions.

    Example:
    ```mlir
    %ctx : !mgp.context
    %size : index
    %ch : !mgp.chain<<"cuda", 3>>

    // Reallocate the buffer in slot index 42 and buffer index 2, using the
    // CUDA runtime in slot 5, which should correspond to the CUDA GPU with
    // virtual id 3.
    %buf = mgp.buffer.realloc.dynamic %ch, %ctx[5, 42, 2], %size : !mgp.buffer<<"cuda", 3>>
    %buf : !mgp.buffer<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                   MGP_ContextType:$theContext,
                   Index:$runtimeSize,
                   UI64Attr:$aRuntimeSlot,                                 // 0
                   UI64Attr:$bBufferSlot,                                  // 1
                   UI64Attr:$cBufferIndex,                                 // 2
                   DefaultValuedAttr<UI64Attr, "kUnknownSize">:$dRawAlign, // 3
                   M_DeviceRefAttr:$eDevice);                              // 4
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,`
    $theContext `[` $aRuntimeSlot `,` $bBufferSlot `,` $cBufferIndex `]` `,`
    $runtimeSize ( `,` `align` $dRawAlign^ )? attr-dict
    `:` custom<DynamicBufferTypeToDevice>(
          type($outBuffer), $eDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferReallocDynamicOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferReallocDynamicOp::getDevice() { return getEDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

def BufferCacheOp : MGP_DeviceOp<"buffer.cache"> {
  let summary = "Moves an `!mgp.buffer` into a dedicated state slot.";
  let description = [{
    Stores the given `!mgp.buffer` into the state slot at the given index.
    It may be later retrieved by `mgp.buffer.get_cached`. The buffer operand
    must have no other references. Depending on the buffer's device, any
    computation associated with the buffer may need to be 'finished' before
    the store is executed.

    Note this op accepts and returns `!mef.chain`s since the buffer cache is
    held on the host.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf : !mgp.buffer<<"cuda", 3>>
    %ch0 : !mef.chain
    %ch1 = mgp.buffer.cache %ch0, %ctx[5], %buf : !mgp.buffer<<"cuda", 3>, 256>
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       UI64Attr:$aRuntimeSlot,     // 0
                       M_DeviceRefAttr:$bDevice);  // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $buffer attr-dict
    `:` custom<BufferTypeToDevice>(type($buffer), $bDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferCacheOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferCacheOp::getDevice() { return getBDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferGetCachedOp
//===----------------------------------------------------------------------===//

def BufferGetCachedOp : MGP_DeviceOp<"buffer.get_cached", [
    Pure,
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
    %buf = mgp.buffer.get_cached %ch0, %ctx[5] : !mgp.buffer<<"cuda", 3>, 256>
    %buf : !mgp.buffer<<"cuda", 3>, 256>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       UI64Attr:$aRuntimeSlot,      // 0
                       UI64Attr:$bRawSize,          // 1
                       M_DeviceRefAttr:$cDevice);   // 2
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` attr-dict
    `:` custom<BufferTypeToRawSizeDevice>(
          type($outBuffer), $bRawSize, $cDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferGetCachedOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferGetCachedOp::getDevice() { return getCDevice(); }
}];
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

    Note that the input chain is on the host device, and the output chain
    on the other device.

    Example:
    ```mlir
      %ch0 : !mgp.chain<<"cpu", 0>>
      %ctx : !mgp.context
      %host_buf : !mgp.buffer<<"cpu", 0>>
      %other_buf : !mgp.buffer<<"cuda", 3>>
      %ch1 = mgp.buffer.host_to_device %ch0, %ctx[4, 5], %host_buf, %other_buf
               : !mgp.buffer<<"cpu", 0>> -> !mgp.buffer<<"cuda", 3>>
      %ch1 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$hostBuffer,
                       MGP_DeviceBufferType:$otherBuffer,
                       UI64Attr:$aHostRuntimeSlot,     // 0
                       UI64Attr:$bOtherRuntimeSlot,    // 1
                       M_DeviceRefAttr:$cHostDevice,   // 2
                       M_DeviceRefAttr:$dOtherDevice); // 3
  let results = (outs MGP_DeviceChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aHostRuntimeSlot `,` $bOtherRuntimeSlot `]`
    `,` $hostBuffer `,` $otherBuffer attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(type($hostBuffer), type($otherBuffer),
          $cHostDevice, $dOtherDevice, type($inChain), type($outChain))
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

    Note that the input chain is on the other device, and the output chain
    on the host device.

    Example:
    ```mlir
      %ch0 : !mgp.chain<<"cuda", 3>>
      %ctx : !mgp.context
      %other_buf : !mgp.buffer<<"cuda", 3>>
      %host_buf : !mgp.buffer<<"cpu", 0>>
      %ch1 = host.buffer.device_to_host %ch0, %ctx[5, 4], %other_buf, %host_buf
               : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cpu", 0>>
      %ch1 : !mgp.chain<<"cpu", 0>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$otherBuffer,
                       MGP_DeviceBufferType:$hostBuffer,
                       UI64Attr:$aOtherRuntimeSlot,   // 0
                       UI64Attr:$bHostRuntimeSlot,    // 1
                       M_DeviceRefAttr:$cOtherDevice, // 2
                       M_DeviceRefAttr:$dHostDevice); // 3
  let results = (outs MGP_DeviceChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aOtherRuntimeSlot `,` $bHostRuntimeSlot `]`
    `,` $otherBuffer `,` $hostBuffer attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(type($otherBuffer), type($hostBuffer),
          $cOtherDevice, $dHostDevice, type($inChain), type($outChain))
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
  let summary = "Copies a buffer into a buffer.";
  let description = [{
    Copies (typically asynchronously) the contents of one `!mgp.buffer` into
    another `!mgp.buffer` on the same device.

    Example:
    ```mlir
      %ch0 : !mgp.chain<<"cuda", 3>>
      %ctx : !mgp.context
      %src_buf : !mgp.buffer<<"cuda", 3>>
      %dst_buf : !mgp.buffer<<"cuda", 3>>
      %ch1 = mgp.buffer.device_to_device %ch0, %ctx[5], %src_buf, %dst_buf
               : !mgp.buffer<<"cuda", 3>> -> !mgp.buffer<<"cuda", 3>>
      %ch1 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$srcBuffer,
                       MGP_DeviceBufferType:$dstBuffer,
                       UI64Attr:$aRuntimeSlot,    // 0
                       M_DeviceRefAttr:$bDevice); // 1
  let results = (outs MGP_DeviceChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $theContext `[` $aRuntimeSlot `]` `,`
    $srcBuffer `,` $dstBuffer attr-dict
    `:` custom<SrcDstBufferTypesToSameDevice>(
          type($srcBuffer), type($dstBuffer), $bDevice,
          type($inChain), type($outChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
     uint64_t BufferDeviceToDeviceOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferDeviceToDeviceOp::getDevice() { return getBDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// BufferUnsizeOp
//===----------------------------------------------------------------------===//

// NOTE: This op does not need a device attribute or runtime slot, it's just
//       for keeping the MLIR type system happy.
def BufferUnsizeOp : MGP_Op<"buffer.unsize", [
    DeclareOpInterfaceMethods<MGP_BufferUnsizeOpInterface>,
    DeclareOpInterfaceMethods<GRTMayAliasInterface>, // The retrieved element is shallow copied
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Returns a shallow copy of the input buffer, but with "
                "unknown size.";
  let description = [{
    Returns a shallow copy of the input buffer, but with unknown size.
    Validation errors will occur if input is not a static size.

    Note that no copying is done, and the result `AsyncValueRef<GML::DeviceRef>`
    is simply a move of the input.

    Example:
    ```mlir
    %in_buf : !mgp.buffer<<"cuda", 3>, 256>
    %out_buf = mgp.buffer.unsize %in_buf : !mgp.buffer<<"cuda", 3>, 256>
    %out_buf : !mgp.buffer<<"cuda", 3>>
    ```
  }];

  let arguments = (ins MGP_DeviceBufferType:$inBuffer);
  let results = (outs  MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{ $inBuffer attr-dict `:` qualified(type($inBuffer)) }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferUnsizeOp::mayAlias(size_t operandIndex, size_t resultIndex) {
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// BufferSplitBorrowOp
//===----------------------------------------------------------------------===//

def BufferSplitBorrowOp : MGP_DeviceOp<"buffer.split.borrow", [
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Splits an `!mgp.buffer` into multiple result buffers which "
                "borrow sections from the input.";
  let description = [{
    Splits an input `!mgp.buffer` into multiple result buffers which borrow
    sections from the input. Concatenating the result buffers would recover the
    contents of the input buffer. The sizes of each result buffer can be
    supplied either implicitly by the result buffer type's size, or explicitly
    as an index operand if the result buffer type is dynamically sized.

    A runtime error will be returned if the actual runtime sizes of input and
    result buffers and size operands are not consistent.

    Note that no copying is done, and the result buffers merely borrow from the
    (offsets into) the input buffer.

    The result buffers will inherit the mutablity and, if possible, alignment
    of the input buffer.

    The runtime will guarantee the input buffer will outlive the result
    buffers. However, currently the runtime does NOT check for overlapping
    reads/writes to the same contents, even in debug builds.

    Example:
    ```mlir
    %ch:  !mgp.chain<<"cuda", 3>>
    %ctx: !mgp.context
    %in_buf : !mgp.buffer<<"cuda", 3>, 256>
    %size1 : index
    %size2 : index
    %out0_buf, %out1_buf, %out2_buf =
        mgp.buffer.split.borrow %ch, %ctx[5], %in_buf, %size1, %size2
        : !mgp.buffer<<"cuda", 3>, 256>
           -> (!mgp.buffer<<"cuda", 3>, 128>,
               !mgp.buffer<<"cuda", 3>>,
               !mgp.buffer<<"cuda", 3>>)
    %out0_buf : !mgp.buffer<<<"cuda", 3>, 128>
    %out1_buf : !mgp.buffer<<"cuda", 3>>
    %out2_buf : !mgp.buffer<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$inBuffer,
                       Variadic<Index>:$runtimeSizes,
                       UI64Attr:$aRuntimeSlot,
                       UI64ArrayElementsAttr:$bOutBufferRawSizes,
                       M_DeviceRefAttr:$cDevice);
  let results = (outs Variadic<MGP_DeviceBufferType>:$outBuffers);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $inBuffer
      (`,` $runtimeSizes^)? attr-dict
      `:` custom<BufferTypesToBufferRawSizesDeviceOutBuffers>(
            type($inBuffer), $bOutBufferRawSizes, $cDevice,
            type($outBuffers), type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferSplitBorrowOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
      return operandIndex == 1;
    }

    uint64_t BufferSplitBorrowOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferSplitBorrowOp::getDevice() { return getCDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferSectionBorrowStaticOp
//===----------------------------------------------------------------------===//

def BufferSectionBorrowStaticOp : MGP_DeviceOp<"buffer.section.borrow.static", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Borrows a section of a parent buffer on device with statically "
                "known offset and size.";
  let description = [{
    Returns a borrowed buffer who's underlying storage points into a section of
    the input buffer. The offset and size of the section is given by
    attributes, and the buffer input and result types must have statically
    known sizes. Thus this op is really only useful in conjunction with static
    memory planning. A validation error is triggered if the requested section
    is out of range of the input buffer.

    Note that no copying is done, and the result buffer merely borrows from
    the (offset into) the input buffer.

    The result buffer will inherit the mutablity and, if possible, alignment
    of the input buffer.

    The runtime will guarantee the input buffer will outlive the result buffer.
    However, the runtime only checks for overlapping concurrent borrows, or
    overlapping reads/writes to the same contents, in debug builds.

    Example:
    ```mlir
    %ch:  !mgp.chain<<"cuda", 3>>
    %ctx : !mgp.context
    %in_buf : !mgp.buffer<<"cuda", 3>, 256>
    %out_buf =
        mgp.buffer.section.borrow.static %ch, %ctx[5], %in_buf, offset 128, size 64
          : !mgp.buffer<<"cuda", 3>, 256>
    %out_buf : !mgp.buffer<<"cuda", 3>, 64>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$inBuffer,
                       UI64Attr:$aRuntimeSlot,    // 0
                       UI64Attr:$bOffset,         // 1
                       UI64Attr:$cSize,           // 2
                       M_DeviceRefAttr:$dDevice); // 3
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $inBuffer `,`
    `offset` $bOffset `,` `size` $cSize attr-dict
    `:` custom<BufferTypeToDevice>(type($inBuffer), $dDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferSectionBorrowStaticOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
      return operandIndex == 2;
    }

    uint64_t BufferSectionBorrowStaticOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferSectionBorrowStaticOp::getDevice() { return getDDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferSectionBorrowDynamicOp
//===----------------------------------------------------------------------===//

def BufferSectionBorrowDynamicOp : MGP_DeviceOp<"buffer.section.borrow.dynamic", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
]> {
  let summary = "Borrows a section of parent buffer on device with dynamically "
                "known offset and size.";
  let description = [{
    Returns a borrowed buffer who's underlying storage points into a section of
    the input buffer. The offset and size of the section is given by
    attributes, which are then scaled by a single `index` operand. The input
    and output buffers must have unknown size. Thus this op is really only
    useful in conjunction with static memory planning when a set of buffers
    all share the same dependence on a single runtime dimension (such as for the
    dynamic batch size). A runtime error is triggered if the requested section
    is out of range of the input buffer.

    Note that no copying is done, and the result buffer merely borrows from
    the (offset into) the input buffer.

    The result buffer will inherit the mutablity and, if possible, alignment
    of the input buffer.

    The runtime will guarantee the input buffer will outlive the result buffer.
    However, the runtime only checks for overlapping concurrent
    borrows, or overlapping reads/writes to the same contents, in debug builds.

    Example:
    ```mlir
    %ch:  !mgp.chain<<"cuda", 3>>
    %ctx : !mgp.context
    %in_buf : !mgp.buffer<<"cuda", 3>>
    %scale : index
    %out_buf =
        cuda.buffer.section.borrow.dynamic %ch, %ctx[5], %in_buf, %scale,
          offset 128, size 64 : !mgp.buffer<<"cuda", 3>>
    %out_buf : !mgp.buffer<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$inBuffer,
                       Index:$scale,
                       UI64Attr:$aRuntimeSlot,       // 0
                       UI64Attr:$bUnscaledOffset,    // 1
                       UI64Attr:$cUnscaledSize,      // 2
                       M_DeviceRefAttr:$dDevice);    // 3
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $inBuffer `,` $scale `,`
    `offset` $bUnscaledOffset `,` `size` $cUnscaledSize attr-dict
    `:` custom<BufferTypeToDevice>(type($inBuffer), $dDevice, type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool BufferSectionBorrowDynamicOp::extendsLifetime(size_t operandIndex, size_t resultIndex) {
      return operandIndex == 1;
    }

    uint64_t BufferSectionBorrowDynamicOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr BufferSectionBorrowDynamicOp::getDevice() { return getDDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// TensorCreateOp
//===----------------------------------------------------------------------===//

def TensorCreateOp : MGP_DeviceOp<"tensor.create", [
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "Creates an `!mgp.tensor` from an `!mgp.tensor_spec` and "
                "`!mgp.buffer`.";
  let description = [{
    Creates an `!mgp.tensor` from an `!mgp.buffer` and a `!mgp.tensor_spec`. The
    construction is delayed until the given chain is ready, since the buffer is
    typically written into as a side-effect of an operation.

    Example:
    ```mlir
    %ctx : !mef.context
    %ch : !mgp.chain<<"cuda", 3>>
    %buf : !mgp.buffer<<"cuda", 3>, 256>
    %spec : !mgp.tensor_spec<4x16xf32>
    %tensor = mgp.tensor.create %ch, %ctx[5], %buf, %spec : !cuda.tensor<<"cuda", 3>, 4x16xf32>
    %tensor : !mgp.tensor<<"cuda", 3>, 4x16xf32>
    ```
  }];

  let arguments = (ins MGP_DeviceChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       MGP_TensorSpecType:$tensorSpec,
                       UI64Attr:$aRuntimeSlot,       // 0
                       M_DeviceRefAttr:$bDevice);    // 1
  let results = (outs MGP_DeviceTensorType: $outTensor);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $buffer `,` $tensorSpec attr-dict
    `:` custom<TensorTypeToDeviceBufferSpec>(type($outTensor), $bDevice,
          type($buffer), type($tensorSpec), type($inChain))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool TensorCreateOp::mayAlias(size_t operandIndex, size_t resultIndex) {
      return operandIndex == getBufferMutable().getOperandNumber();
    }

    uint64_t TensorCreateOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr TensorCreateOp::getDevice() { return getBDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// TensorExtractBufferOp
//===----------------------------------------------------------------------===//

def TensorExtractBufferOp : MGP_DeviceOp<"tensor.extract.buffer", [
    Pure,
    DeclareOpInterfaceMethods<GRTMayAliasInterface>,
]> {
  let summary = "Extracts the `!mgp.buffer` from an `!mgp.tensor`.";
  let description = [{
    Returns the `!mgp.buffer` component of tensor. The buffer is moved if
    the tensor has no other live references, otherwise a copy of the tensor's
    buffer is returned.

    Example:
    ```mlir
    %ctx : !mgp.context
    %tensor : !mgp.tensor<<"cuda", 3>, 8x64xf32>
    %buf = mgp.tensor.extract.buffer %ctx[5], %tensor : !mgp.tensor<<"cuda", 3>, 8x64xf32>
    %buf : !mgp.buffer<<"cuda", 3>, 2048>
    ```
  }];

  let arguments = (ins MGP_ContextType:$theContext,
                       MGP_DeviceTensorType:$tensor,
                       UI64Attr:$aRuntimeSlot,     // 0
                       M_DeviceRefAttr:$bDevice);  // 1
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $theContext `[` $aRuntimeSlot `]` `,` $tensor attr-dict
    `:` custom<TensorTypeToDeviceBuffer>(type($tensor), $bDevice, type($outBuffer))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    bool TensorExtractBufferOp::mayAlias(size_t operandIndex, size_t resultIndex) {
      return operandIndex == getTensorMutable().getOperandNumber();
    }

    uint64_t TensorExtractBufferOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr TensorExtractBufferOp::getDevice() { return getBDevice(); }
}];
}

//===----------------------------------------------------------------------===//
// TensorExtractShapeOp
//===----------------------------------------------------------------------===//

def TensorExtractShapeOp : MGP_DeviceOp<"tensor.extract.shape", [
    Pure,
]> {
  let summary = "Extracts the `!mgp.tensor_spec` from a `!mgp.tensor`.";
  let description = [{
    Returns the `!mgp.tensor_spec` of a tensor, with the dtype represented as
    `unknown`.

    Example:
    ```mlir
    %ctx : !mgp.context
    %tensor : !mgp.tensor<<"cuda", 3>, 8x64xf32>
    %shape = mgp.tensor.extract.shape %ctx[5], %tensor : !cuda.tensor<<"cuda", 3>, 8x64xf32>
    %shape : !mgp.tensor_spec<8x64xunk>
    ```
  }];

  let arguments = (ins MGP_ContextType: $theContext,
                       MGP_DeviceTensorType: $tensor,
                       UI64Attr: $aRuntimeSlot,    // 0
                       M_DeviceRefAttr:$bDevice);  // 1
  let results = (outs MGP_TensorSpecType:$outShape);
  let assemblyFormat = [{
    $theContext `[` $aRuntimeSlot `]` `,` $tensor attr-dict
    `:` custom<TensorTypeToDeviceShape>(type($tensor), $bDevice, type($outShape))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t TensorExtractShapeOp::getSlot() { return getARuntimeSlot(); }

    M::DeviceRefAttr TensorExtractShapeOp::getDevice() { return getBDevice(); }
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
    %buf: !mgp.buffer<<"cuda", 3>, 2048>
    %spec: !mgp.tensor_spec<8x64xf32>
    %ch1 = mgp.debug.tensor.print %ch0, %ctx[5], %buf, %spec, "label"
           : !mgp.tensor<<"cuda", 3>, 8x64xf32>
    %ch1: !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       MGP_TensorSpecType:$spec,
                       UI64Attr:$aRuntimeSlot,                        // 0
                       DefaultValuedAttr<StrAttr, "\"\"">: $bLabel,   // 1
                       M_DeviceRefAttr:$cDevice);                     // 2
  let results = (outs ChainType: $outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $aRuntimeSlot `]` `,` $buffer `,` $spec (`,` $bLabel^)?
    attr-dict
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
// KernelExecuteGenericOp
//===----------------------------------------------------------------------===//

def KernelExecuteGenericOp : MGP_DeviceOp<"kernel.execute.generic", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MGP_KernelExecuteOpInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_InOutSemanticsOpInterface>,
]> {
  let summary = "Executes a Mojo kernel.";
  let description = [{
    Executes the Mojo kernel with the given function and module index from the
    loaded module at the given `!mgp.context` slot. The kernel entry point is
    always executed on the host device, however it may in turn launch a kernel
    for a non-host device. All arguments must reside on the devices the kernel
    expects. Generally this means arguments are on the host device or on the
    device for the operation.

    Note that the summary attribute is only used for debugging and profiling.
    The actual kernel to execute is determined only by the given module and
    kernel indexes.

    Example:
    ```mlir
    %ch0 : !mgp.chain<<"cuda", 3>>
    %ctx : !mgp.context
    %buf0 : !mgp.buffer<<"cpu", 0>>
    %buf1 : !mgp.buffer<<"cuda", 3>>
    %buf2 : !mgp.buffer<<"cuda", 3>, 48>
    %dim0 : index
    %dim1 : index
    // Launches a CUDA kernel via a host Mojo shim:
    //  - The CUDA kernel will execute on the CUDA GPU with virtual device 3,
    //    which has its runtime established in slot 5.
    //  - The host shim can be retrieved from the module with index 7, kernel
    //    with index 9, using the host runtime established in slot 4.
    //  - The kernel takes two input buffers and one output buffer. The
    //    first buffer resides on the host, the other two are on the GPU.
    //  - The kernel takes two runtime dimensions.
    %ch1 = mgp.kernel.execute.generic %ch0,
             %ctx[4, 5], (7, 9),
             (in %buf0 : !mgp.buffer<<"cpu", 0>>
              in %buf1 : !mgp.buffer<<"cuda", 3>>
              out %buf2 : !mgp.buffer<<"cuda", 3>, 48>)
             (%dim0, %dim1),
             host <"cpu", 0> other <"cuda", 3> summary "kernel1" is_async false
    %ch1 : !mgp.chain<<"cuda", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins
    MGP_DeviceChainType:$inChain,
    MGP_ContextType:$theContext,
    Variadic<AnyTypeOf<[MGP_DeviceBufferType, MGP_List]>>:$args,
    Variadic<Index>:$inDimensions,
    UI64Attr:$aHostRuntimeSlot,                          // 0
    UI64Attr:$bOtherRuntimeSlot,                         // 1
    UI64Attr:$cHostModuleIndex,                          // 2
    UI64Attr:$dHostKernelIndex,                          // 3
    M_InOutSignatureAttr:$eInOutSignature,               // 4
    M_DeviceRefAttr:$fHostDevice,                        // 5
    M_DeviceRefAttr:$gOtherDevice,                       // 6
    StrAttr:$hSummary,                                   // 7
    DefaultValuedAttr<UI64Attr, "0">:$iNumOutDimensions, // 8
    DefaultValuedAttr<BoolAttr, "false">:$jIsAsync);     // 9
  let results = (outs MGP_DeviceChainType:$outChain,
                      Variadic<Index>:$outDimensions);
  let assemblyFormat = [{
    $inChain `,`
    $theContext `[` $aHostRuntimeSlot `,` $bOtherRuntimeSlot `]` `,`
    `(` $cHostModuleIndex `,` $dHostKernelIndex `)` `,`
    custom<InOutArgsSignature>($args, type($args), $eInOutSignature) `,`
    ( `(` $inDimensions^ `)` `,` )?
    `host` $fHostDevice
    `other` custom<DeviceRef>($gOtherDevice, type($inChain))
    `summary` $hSummary
    ( `out_dims` $iNumOutDimensions^ )?
    ( `is_async` $jIsAsync^ )?
    attr-dict
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t KernelExecuteGenericOp::getSlot() { return getBOtherRuntimeSlot(); }

    M::DeviceRefAttr KernelExecuteGenericOp::getDevice() { return getGOtherDevice(); }
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPDEVICEOPS_TD
