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
