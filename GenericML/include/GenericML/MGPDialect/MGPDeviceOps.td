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
// overhead of dealing with device labels at runtime.
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
                       MGP_DeviceContextType:$deviceContext,
                       MGP_OptDeviceRefAttr:$bDevice,  // 1
                       StrAttr:$cTag,
                       StrAttr:$dFilePath); // 2
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` ( `device` $bDevice^ )?  attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{

    M::DeviceRefAttr DeviceContextProfileStartOp::getDevice() { return getBDevice(); }
  }];
}

def DeviceContextProfileEndOp : MGP_DeviceOp<"device.context.profile.end"> {
  let summary = "Shim to tag the start of a device context profiling run";
  let description = [{This is temporary op. to enable onnx profiling - Should use max-driver for this in future}];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceContextType:$deviceContext,
                       MGP_OptDeviceRefAttr:$bDevice,  // 1
                       StrAttr:$cTag,
                       StrAttr:$dFilePath); // 2
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $deviceContext `,` ( `device` $bDevice^ )?  attr-dict `:` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;
  let extraClassDefinition = [{

    M::DeviceRefAttr DeviceContextProfileEndOp::getDevice() { return getBDevice(); }
  }];
}


//===----------------------------------------------------------------------===//
// ModuleLoadOp
//===----------------------------------------------------------------------===//

def ModuleLoadOp : MGP_DeviceOp<"module.load",
    [DeclareOpInterfaceMethods<MGP_HasRuntimeSlotOpInterface>,]
> {
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
    by (module index, constant index) in `mgp.buffer.constant.load`.

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
               device <"gpu", 3>
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
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
      "getLiftableAttributes",
      "getIgnorableAttributes"
    ]>,
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
      !mgp.opaque<"weights_registry"> -> !mgp.buffer
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_OpaqueType:$weightsRegistry,
                       StrAttr:$bName,                                         // 0
                       UI64Attr:$cSize,                                        // 1
                       UI64Attr:$dAlign,                                       // 2
                       MGP_OptDeviceRefAttr:$eDevice,                          // 3
                       DefaultValuedAttr<BoolAttr, "false">:$gHasAlias,        // 4
                       DefaultValuedAttr<BoolAttr, "false">:$hIsPlaceholder);  // 5

  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain `,` $weightsRegistry
    `[`
    $bName `,` $cSize `,` `align` $dAlign
    oilist( `has_alias` $gHasAlias
          | `placeholder` $hIsPlaceholder
          | `device` $eDevice)
    `]`
    attr-dict `:` qualified(type($weightsRegistry))
    `->` qualified(type($outBuffer))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the buffer's alignment.
    std::optional<uint64_t> getOptAlign();
  }];

  let extraClassDefinition = [{
    SmallVector<Attribute> BufferConstantExternalOp::getLiftableAttributes() {
      return {getBNameAttr(), getCSizeAttr(), getDAlignAttr()};
    }

    // Ignore the devices.
    SmallVector<Attribute> BufferConstantExternalOp::getIgnorableAttributes() {
      return {getEDeviceAttr(), getGHasAliasAttr()};
    }

    M::DeviceRefAttr BufferConstantExternalOp::getDevice() { return getEDevice(); }

    bool BufferConstantExternalOp::isTrivial() { return false; }
  }];
}

//===----------------------------------------------------------------------===//
// BufferAllocOp
//===----------------------------------------------------------------------===//

def BufferAllocOp : MGP_DeviceOp<"buffer.alloc", [
    DeclareOpInterfaceMethods<MGP_BufferAllocOpInterface, ["getOutBuffer"]>,
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getIgnorableAttributes"]>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
    InferTypeOpAdaptor
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
      %ch : !mgp.chain<<"gpu", 3>>
      %ctx : !mgp.context
      %size : index
      // Allocates %size bytes using the CUDA runtime from slot 5, which
      // should correspond to the CUDA GPU with virtual device id 3.
      %buf = mgp.buffer.alloc %ch, %size : !mgp.buffer<<"gpu", 3>>
      %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       Index:$runtimeSize,
                       MGP_DeviceContextType:$deviceContext,
                       DefaultValuedAttr<UI64Attr, "kUnknownSize">:$bRawAlign, // 1
                       MGP_OptDeviceRefAttr:$cDevice);                         // 2
  let results = (outs MGP_DeviceBufferType:$outBuffer);
  let assemblyFormat = [{
    $inChain  `,` $runtimeSize `,` $deviceContext
    ( `,` `align` $bRawAlign^ )? attr-dict
    `:` custom<BufferTypeToDevice>(
          type($outBuffer), $cDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    uint64_t BufferAllocOp::getRawAlign() { return getBRawAlign(); }

    SmallVector<mlir::Attribute> BufferAllocOp::getIgnorableAttributes() {
      return {getBRawAlignAttr(), getCDeviceAttr()};
    }

    M::DeviceRefAttr BufferAllocOp::getDevice() { return getCDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferCacheOp
//===----------------------------------------------------------------------===//

def BufferCacheOp : MGP_Op<"buffer.cache"> {
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
    %buf : !mgp.buffer<<"gpu", 3>>
    %ch0 : !mef.chain
    // Cache the given buffer in slot index 42.
    %ch1 = mgp.buffer.cache %ch0, %ctx[5, 42], %buf
             : !mgp.buffer<<"gpu", 3>>
    %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceBufferType:$buffer,
                       UI64Attr:$bBufferSlot);
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $theContext `[` $bBufferSlot `]` `,`
    $buffer attr-dict `:` functional-type(operands, results)
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
      %buf0 : !mgp.buffer
      %ch2 = mgp.buffer.free %ch1, %buf0 : !mgp.buffer
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
      %host_buf : !mgp.buffer
      %other_buf : !mgp.buffer<<"gpu", 3>>
      %ch1 = mgp.buffer.host_to_device %ch0, %host_buf, %other_buf
               : !mgp.buffer -> !mgp.buffer<<"gpu", 3>>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$hostBuffer,
                       MGP_DeviceBufferType:$otherBuffer,
                       MGP_DeviceContextType:$deviceContext,
                       MGP_OptDeviceRefAttr:$cHostDevice,   // 2
                       MGP_OptDeviceRefAttr:$dOtherDevice); // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain `,` $hostBuffer `,` $otherBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($hostBuffer), type($otherBuffer),
          $cHostDevice, $dOtherDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

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
      %other_buf : !mgp.buffer<<"gpu", 3>>
      %host_buf : !mgp.buffer
      %ch1 = mgp.buffer.device_to_host %ch0, %other_buf, %host_buf
               : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$otherBuffer,
                       MGP_DeviceBufferType:$hostBuffer,
                       MGP_DeviceContextType:$deviceContext,
                       MGP_OptDeviceRefAttr:$cOtherDevice, // 2
                       MGP_OptDeviceRefAttr:$dHostDevice); // 3
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $otherBuffer `,` $hostBuffer `,` $deviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($otherBuffer), type($hostBuffer),
          $cOtherDevice, $dHostDevice, type($inChain)) `,` qualified(type($deviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

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
      %src_buf : !mgp.buffer<<"gpu", 3>>
      %dst_buf : !mgp.buffer<<"gpu", 1>>
      %ch1 = mgp.buffer.device_to_device %ch0, %ctx[5, 3], %src_buf, %dst_buf
               : !mgp.buffer<<"gpu", 3>> -> !mgp.buffer<<"gpu", 1>>
      %ch1 : !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$srcBuffer,
                       MGP_DeviceBufferType:$dstBuffer,
                       MGP_DeviceContextType:$srcDeviceContext,
                       MGP_DeviceContextType:$dstDeviceContext,
                       M_DeviceRefAttr:$cSrcDevice,   // 0
                       M_DeviceRefAttr:$dDstDevice);  // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    $inChain  `,` $srcBuffer `,` $srcDeviceContext`,` $dstBuffer `,` $dstDeviceContext attr-dict
    `:` custom<SrcDstBufferTypesToDevices>(
          type($srcBuffer), type($dstBuffer),
          $cSrcDevice, $dDstDevice, type($inChain)) `,` qualified(type($srcDeviceContext))`,` qualified(type($dstDeviceContext))
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    M::DeviceRefAttr getSrcDevice();
    M::DeviceRefAttr getDstDevice();
    }];

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferDeviceToDeviceOp::getDevice() { return getDDstDevice(); }
    M::DeviceRefAttr BufferDeviceToDeviceOp::getSrcDevice() { return getCSrcDevice(); }
    M::DeviceRefAttr BufferDeviceToDeviceOp::getDstDevice() { return getDDstDevice(); }
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
    %in_ch : !mgp.chain<<"gpu", 3>>
    %out_buf : !mgp.buffer<<"gpu", 3>>
    %in0_buf : !mgp.buffer<<"gpu", 3>>
    %in1_buf : !mgp.buffer<<"gpu", 3>>
    %in2_buf : !mgp.buffer<<"gpu", 3>>
    %out_ch = mgp.buffer.concat %in_ch, %out_buf, %in0_buf, %in1_buf, %in2_buf
              : (!mgp.buffer<<"gpu", 3>>,
                 !mgp.buffer<<"gpu", 3>>,
                 !mgp.buffer<<"gpu", 3>>) -> !mgp.buffer<<"gpu", 3>>
    %out_ch : !mgp.chain<<"gpu", 3>>
    ```
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$outBuffer,
                       Variadic<MGP_DeviceBufferType>:$inBuffers,
                       MGP_DeviceContextType:$ctx,
                       MGP_OptDeviceRefAttr:$bDevice); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{

    M::DeviceRefAttr BufferConcatOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// BufferPlanOp
//===----------------------------------------------------------------------===//
def BufferPlanOp : MGP_Op<"buffer.plan", [
    InferTypeOpAdaptor
]> {
  let summary = "Runtime memory planning for buffers";
  let description = [{
    Given the runtimeSize and lifetime information for a list of allocations,
    returns the high watermark size and offsets for each allocation.

    Alloc lifetimes are represented as the min/max pre/postorder indices of the
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
    Pure,
    InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<GRTExtendsLifetimeInterface>,
    DeclareOpInterfaceMethods<MGP_LiftableBufferCreateOpInterface>,
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
    %in_buf : !mgp.buffer
    %offset : index
    %size : index
    %out_buf =
        mgp.buffer.slice %in_buf, %offset, %size : !mgp.buffer
    %out_buf : !mgp.buffer
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
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Returns contents of the buffer to an `index`";
  let description = [{
    Returns contents of a 4- or 8-byte buffer as a 64-bit `index`.

    Sign extension is performed when upcasting from a 32-bit integer.

    Example:
    ```mlir
    %0 = mgp.buffer.to_index %ch0, %buf : !mgp.buffer
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

def BufferToBoolOp : MGP_DeviceOp<"buffer.to_bool", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "Returns contents of a 1-byte buffer as an i1.";
  let description = [{
    Returns contents of a one byte buffer as an i1.

    Example:
    ```mlir
    %0 = mgp.to_bool %ch0, %ctx[5], %buf : !mgp.buffer<<"gpu", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$buffer,
                       MGP_OptDeviceRefAttr:$bDevice);  // 1
  let results = (outs I1:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $inChain `,` $buffer attr-dict
      `:` custom<BufferTypeToDevice>(type($buffer), $bDevice, type($inChain))
  }];

  let extraClassDefinition = [{

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
    DeclareOpInterfaceMethods<EmitsMojoInterface>,
    DeclareOpInterfaceMethods<KernelTerminatorProvider>,
]> {
  let summary = "Set a buffer with the values from index inputs.";
  let description = [{
    The runtime index values will be sign extended or truncated based on the
    given buffer's size. Therefore only buffer sizes of 4 and 8 times the number
    of index inputs are supported by the primitive.

    Example:
    ```mlir
    %ctx : !mgp.context
    %buf0 : !mgp.buffer<<"gpu", 3>>
    %idx0 : index
    %ch0 = mgp.buffer.set_with_index %buf0, %idx0 : !mgp.buffer<<"gpu", 3>>
    %ch0 : !mgp.chain<<"gpu", 3>>

    %idx1 : index
    %idx2 : index
    %buf1 : !mgp.buffer<<"gpu", 3>>
    %ch1 = mgp.buffer.set_with_index %buf1, %idx0, %idx1, %idx2 : !mgp.buffer<<"gpu", 3>>
    %ch1 : !mgp.chain<<"gpu", 3>>
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins MGP_DeviceBufferType:$buffer,
                       Variadic<Index>:$values,
                       MGP_OptDeviceRefAttr:$bDevice); // 1
  let results = (outs ChainType:$outChain);
  let assemblyFormat = [{$buffer ( `,` $values^ )? attr-dict
    `:` custom<BufferTypeToDevice>(type($buffer), $bDevice)
  }];
  let hasVerifier = 1;

  let extraClassDefinition = [{
    M::DeviceRefAttr BufferSetWithIndexOp::getDevice() { return getBDevice(); }
  }];
}

//===----------------------------------------------------------------------===//
// DebugTensorPrintOp
//===----------------------------------------------------------------------===//

def DebugTensorPrintOp : MGP_Op<"debug.tensor.print", [
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, [
      "getLiftableAttributes"
    ]>,
    DeclareOpInterfaceMethods<EmitsMojoInterface>,
    DeclareOpInterfaceMethods<KernelTerminatorProvider>
  ]> {
  let summary = "Prints a debug representation of an tensor. For debugging "
                "and testing only.";
  let description = [{
    Prints a debug representation of an tensor, given as a pair of an
    `!mgp.buffer` and a `!mgp.tensor_spec`. If a label attribute is supplied
    the tensor contents are printed with that labelm and tensor contents. Otherwise
    just the tensor metadata is printed. Printing the tensor contents may requires
    an explicit device-to-host copy. Only CPU buffers can be printed with contents.

    For debugging and testing only.

    Note this op accepts and returns `!mef.chain`s since printing is managed
    by the host.

    Example:
    ```mlir
    %ch0: !mef.chain
    %ctx: !mgp.context
    %buf: !mgp.buffer<<"gpu", 3>>
    %spec: !mgp.tensor_spec<8x64xf32>
    %ch1 = mgp.debug.tensor.print %ch0, %ctx[5], %buf, %spec, "label"
           : !mgp.tensor<8x64xf32, <"gpu", 3>>
    %ch1: !mef.chain
    ```
  }];

  // CAUTION: The attribute arguments must be in alphabetical order by their
  // names.
  let arguments = (ins ChainType:$inChain,
                       MGP_DeviceBufferType:$buffer,
                       MGP_TensorSpecType:$spec,
                       DefaultValuedAttr<StrAttr, "\"\"">: $bLabel);
  let results = (outs ChainType: $outChain);

  let extraClassDefinition = [{
    SmallVector<Attribute> DebugTensorPrintOp::getLiftableAttributes() {
      return {getBLabelAttr()};
    }
  }];

  let assemblyFormat = [{
    $inChain `,` $buffer `,` $spec
    ( `,` $bLabel^ )? attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// SyncOp
//===----------------------------------------------------------------------===//

def MGP_SyncOp : MGP_DeviceOp<"sync", [
    DeclareOpInterfaceMethods<MGP_LiftableChainCreateOpInterface>,
    DeclareOpInterfaceMethods<MGP_SelectiveLowerAttrOpInterface, ["getIgnorableAttributes"]>,
    Pure
]> {
  let summary = "Schedules a device specific sync";
  let description = [{
    Returns a ready chain once the device has synchronized. In
    cpu this is a nop as the sync happens with the chain, on
    gpu this records a gpu event.
  }];

  let arguments = (ins ChainType:$inChain,
                       MGP_ContextType:$theContext,
                       MGP_DeviceContextType:$deviceContext,
                       MGP_OptDeviceRefAttr:$bDevice);               // 1
  let results = (outs ChainType: $outChain);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
        functional-type(operands, results)
  }];

  let extraClassDefinition = [{
    SmallVector<mlir::Attribute> SyncOp::getIgnorableAttributes() {
      return {getBDeviceAttr()};
    }

    M::DeviceRefAttr SyncOp::getDevice() { return getBDevice(); }
  }];
}

#endif  // GENERICML_MGPDIALECT_MGPDEVICEOPS_TD
