//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for MGP types which may also be used by other dialects which
// have similar buffer and tensor types.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPEINTERFACES_TD
#define GENERICML_MGPDIALECT_MGPTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def MGP_TensorTypeInterface : TypeInterface<"MGPTensorTypeInterface"> {
  let description = [{
    Interface for types which represent runtime tensors.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the tensor spec", "::M::MGP::TensorSpecType", "getTensorSpec", (ins), "return $_type.getTensorSpec();"
    >
  ];
}

def MGP_BufferTypeInterface : TypeInterface<"MGPBufferTypeInterface"> {
  let description = [{
    Types that represent runtime buffers.
  }];
  let cppNamespace = "::M::MGP";
}


def MGP_HasDeviceTypeInterface : TypeInterface<"MGPHasDeviceTypeInterface"> {
  let description = [{
    Types which carry an `#M.device_ref` attribute.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the type's device.", "::M::DeviceRefAttr", "getDevice", (ins)
    >
  ];
}

#endif  // GENERICML_MGPDIALECT_MGPTENSORTYPEINTERFACE_TD
