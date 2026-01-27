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
    InterfaceMethod<
      "The alignment of the allocation",
      "uint64_t", "getRawAlign", (ins)
    >,
  ];
}

def MGP_BufferAliasOpInterface : OpInterface<"MGPBufferAliasOpInterface"> {
  let description = [{
    Interface for ops which will take an input buffer and alias it with the output directly
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the aliased buffer.",
      "mlir::Value", "getInBuffer", (ins)
    >,
    InterfaceMethod<
      "Returns the alias to the buffer.",
      "mlir::Value", "getOutBuffer", (ins)
    >
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
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getNumResults() == 1 ? $_op->getResult(0) : nullptr;
      }]
    >,
    InterfaceMethod<
      "Returns true if the op is 'trivial', in the sense that the op latency is "
      "comparable to that of retrieving a buffer from an `!mgp.context` slot.",
      "bool", "isTrivial", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return false;
      }]
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
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getNumResults() == 1 ? $_op->getResult(0) : nullptr;
      }]
    >,
  ];
}

def MGP_SelectiveLowerAttrOpInterface : OpInterface<"MGPSelectiveLowerAttrOpInterface"> {
  let description = [{
    Interface for ops that require selectively converting a static attribute into
    runtime values when lowering to mogg.call.

    The interface categorize attributes into 3 categories:
    1. the "liftable" attributes, which need to be lifted to runtime value to avoid
       unnecessary elaboration.
    2. the "ignorable" attribute, which will simply be dropped during lowering.
    3. the "parameterizable" attribute, which need to be lowered into static parameters.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the vector of attributes that need to be lifted.",
      "mlir::SmallVector<mlir::Attribute>", "getLiftableAttributes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      "Returns the vector of attributes that need to be ignored.",
      "mlir::SmallVector<mlir::Attribute>", "getIgnorableAttributes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return {};
      }]
    >,
    InterfaceMethod<
      "Returns the vector of attributes that need to be parameterized.",
      "mlir::SmallVector<mlir::NamedAttribute>", "getParameterizableAttributes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        mlir::SmallVector<mlir::NamedAttribute> ret;
        // Use cast to force dynamic dispatch
        auto liftables = cast<MGP::MGPSelectiveLowerAttrOpInterface>(*$_op).getLiftableAttributes();
        auto ignorables = cast<MGP::MGPSelectiveLowerAttrOpInterface>(*$_op).getIgnorableAttributes();
        for (auto attr : $_op->getAttrs()) {
          // Skip liftable/ignorable attrs.
          if (llvm::find(liftables, attr.getValue()) != liftables.end())
             continue;
          if (llvm::find(ignorables, attr.getValue()) != ignorables.end())
             continue;
          ret.push_back(attr);
        }
        return ret;
      }]
    >
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

def MGP_TensorExtractOpInterface : OpInterface<"MGPTensorExtractOpInterface"> {
  let description = [{
    Interface for ops which extract a buffer and a tensor spec from a tensor.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the resulting buffer",
      "::mlir::Value", "getBufferResult", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getBufferResult();
      }]
    >,
    InterfaceMethod<
      "Get the resulting tensor spec",
      "::mlir::Value", "getTensorSpecResult", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getTensorSpecResult();
      }]
    >,
    InterfaceMethod<
      "Get the input tensor",
      "::mlir::Value", "getTensor", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getTensor();
      }]
    >,
    InterfaceMethod<
      "Get the input tensor as a mutable operand",
      "::mlir::OpOperand&", "getTensorMutable", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getTensorMutable();
      }]
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
      "::mlir::Value", "getBuffer", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getBuffer();
      }]
    >,
    InterfaceMethod<
      "Get the input chain",
      "::mlir::Value", "getInChain", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getInChain();
      }]
    >,
    InterfaceMethod<
      "Get the input tensor spec",
      "::mlir::Value", "getTensorSpec", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getTensorSpec();
      }]
    >,
    InterfaceMethod<
      "Get the output tensor",
      "::mlir::Value", "getResult", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getResult(0);
      }]
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
      "::mlir::Value", "getInBufferValue", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getInBuffer();
      }]
    >,
    InterfaceMethod<
      "Get the output buffer",
      "::mlir::Value", "getOutBufferValue", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op.getOutBuffer();
      }]
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
