//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for MGP ops which may also be used by other dialects which
// have similar buffer and tensor ops.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPINTERFACES_TD
#define GENERICML_MGPDIALECT_MGPOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def MGP_BufferAllocOpInterface : OpInterface<"MGPBufferAllocOpInterface"> {
  let description = [{
    Interface for ops which allocate or reallocate (via an !mgp.context slot)
    buffers.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the input chain gating this op's allocation.",
      "mlir::Value", "getInChain", (ins)
    >,
    InterfaceMethod<
      "Sets the input chain gating this op's allocation.",
      "void", "setInChain", (ins "mlir::Value":$chain)
    >,
    InterfaceMethod<
      "Returns the buffer allocated by this op.",
      "mlir::Value", "getOutBuffer", (ins)
    >,
  ];
}

def MGP_StaticAllocOpInterface : OpInterface<"MGPStaticAllocOpInterface"> {
  let description = [{
    Interface for ops which statically allocate buffers.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the input chain gating this op's allocation.",
      "mlir::Value", "getInChain", (ins)
    >,
    InterfaceMethod<
      "The size of the allocation",
      "uint64_t", "getSize", (ins)
    >,
    InterfaceMethod<
      "The alignment of the allocation",
      "uint64_t", "getRawAlign", (ins)
    >,
    InterfaceMethod<
      "Returns the buffer allocated by this op.",
      "mlir::Value", "getOutBuffer", (ins)
    >,
  ];
}

def MGP_DynamicAllocOpInterface : OpInterface<"MGPDynamicAllocOpInterface"> {
  let description = [{
    Interface for ops which dynamically allocate buffers.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the input chain gating this op's allocation.",
      "mlir::Value", "getInChain", (ins)
    >,
InterfaceMethod<
      "The runtime size of the allocation",
      "mlir::TypedValue<::mlir::IndexType>", "getRuntimeSize", (ins)
    >,
    InterfaceMethod<
      "The alignment of the allocation",
      "uint64_t", "getRawAlign", (ins)
    >,
    InterfaceMethod<
      "Returns the buffer allocated by this op.",
      "mlir::Value", "getOutBuffer", (ins)
    >,
  ];
}

def MGP_LiftableBufferCreateOpInterface : OpInterface<"MGPLiftableBufferCreateOpInterface"> {
  let description = [{
    Interface for ops which may return a buffer as their only result and can be
    safely moved from the 'execute' to the 'init' block. The buffer may be from
    an allocation or a constant. The buffer can be cached within an
    `!mgp.context` slot if it is still needed by the 'execute' block.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the buffer created by the liftable op, or null if op is not liftable.",
      "mlir::Value", "getOutBuffer", (ins),
      "return $_op->getNumResults() == 1 ? $_op->getResult(0) : nullptr;"
    >,
    InterfaceMethod<
      "Returns true if the op is 'trivial', in the sense that the op latency is "
      "comparable to that of retrieving a buffer from an `!mgp.context` slot.",
      "bool", "isTrivial", (ins), "", "return false;"
    >,
  ];
}

def MGP_LiftableChainCreateOpInterface : OpInterface<"MGPLiftableChainCreateOpInterface"> {
  let description = [{
    Interface for ops which may return a chain-like type as their only result and
    can be safely moved from the 'execute' to the 'init' block. The chain
    indicates the op has completed. Any execution serialization enforced by the
    result chain must be maintained after the op is moved.

    If the op performs observable side effects then 'safely moved' means it is
    ok for those side effects to occur once on 'init' instead of once per
    'execute'.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the chain created by the liftable op, or null if op is not liftable.",
      "mlir::Value", "getOutChain", (ins),
      "return $_op->getNumResults() == 1 ? $_op->getResult(0) : nullptr;"
    >,
  ];
}


def MGP_InOutSemanticsOpInterface : OpInterface<"MGPInOutSemanticsOpInterface"> {
  let description = [{
    Interface for ops which may mutate some of their operands.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the InOutSemantics for the operand at the given index.",
      "::M::InOutSignatureAttr::InOutSemantics", "getInOutSemantics", (ins "size_t":$index)
    >,
  ];
}

def MGP_KernelExecuteOpInterface : OpInterface<"MGPKernelExecuteOpInterface"> {
  let description = [{
    Interface for ops which execute a kernel, possibly indirectly.
  }];
  let cppNamespace = "::M::MGP";
}

def MGP_TensorExtractOpInterface : OpInterface<"MGPTensorExtractOpInterface"> {
  let description = [{
    Interface for ops which extract a buffer and a tensor spec from a tensor.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the resulting buffer",
      "::mlir::Value", "getBufferResult", (ins), "return $_op.getBufferResult();"
    >,
    InterfaceMethod<
      "Get the resulting tensor spec",
      "::mlir::Value", "getTensorSpecResult", (ins), "return $_op.getTensorSpecResult();"
    >,
    InterfaceMethod<
      "Get the input tensor",
      "::mlir::Value", "getTensor", (ins), "return $_op.getTensor();"
    >,
    InterfaceMethod<
      "Get the input tensor as a mutable operand",
      "::mlir::OpOperand&", "getTensorMutable", (ins), "return $_op.getTensorMutable();"
    >
  ];
}

def MGP_CreateTensorOpInterface : OpInterface<"MGPCreateTensorOpInterface"> {
  let description = [{
    Interface for ops which create a tensor from a buffer, spec, and chain.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the input buffer",
      "::mlir::Value", "getBuffer", (ins), "return $_op.getBuffer();"
    >,
    InterfaceMethod<
      "Get the input chain",
      "::mlir::Value", "getInChain", (ins), "return $_op.getInChain();"
    >,
    InterfaceMethod<
      "Get the input tensor spec",
      "::mlir::Value", "getTensorSpec", (ins), "return $_op.getTensorSpec();"
    >,
    InterfaceMethod<
      "Get the output tensor",
      "::mlir::Value", "getResult", (ins), "return $_op.getResult();"
    >
  ];
}

def MGP_BufferUnsizeOpInterface : OpInterface<"MGPBufferUnsizeOpInterface"> {
  let description = [{
    Interface for ops which unsize and alias a buffer.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the input buffer",
      "::mlir::Value", "getInBufferValue", (ins), "return $_op.getInBuffer();"
    >,
    InterfaceMethod<
      "Get the output buffer",
      "::mlir::Value", "getOutBufferValue", (ins), "return $_op.getOutBuffer();"
    >,
  ];
}

def MGP_HasDeviceOpInterface : OpInterface<"MGPHasDeviceOpInterface"> {
  let description = [{
    Interface for ops which have an #M.device_ref attribute.

    Some ops actually involve both a 'host' and 'other' device. For these ops
    this interface provides the 'other' device.

    Ops with this interface are dispatched at runtime to an op registered
    with the name:
    ```
       op->getName() + "." + op->getDevice().getLabel()
    ```
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the #M.device_ref used by the op.",
      "M::DeviceRefAttr", "getDevice", (ins)
    >,
  ];
}

def MGP_HasRuntimeSlotOpInterface : OpInterface<"MGPHasRuntimeSlotOpInterface"> {
  let description = [{
    Interface for ops which access a device-specific runtime via an !mgp.context
    slot index.

    Some ops actually involve both a 'host' and 'other' device. For these ops
    this interface provides the 'other' device runtime. This is the same
    convention used by `MGP_HasDeviceOpInterface`, thus the runtime slot and
    devices will always match.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the runtime slot index used by the op.",
      "uint64_t", "getSlot", (ins)
    >,
    InterfaceMethod<
      "Returns the !mgp.context used by the op.",
      "::mlir::TypedValue<::M::MGP::ContextType>", "getTheContext", (ins)
    >,
  ];
}

#endif // GENERICML_MGPDIALECT_MGPOPINTERFACES_TD
