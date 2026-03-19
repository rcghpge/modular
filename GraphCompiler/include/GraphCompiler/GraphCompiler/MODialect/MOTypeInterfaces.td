//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines type interfaces in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPEINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"
include "Support/MDialect/MAttrs.td"

def MO_HasDeviceTypeInterface : TypeInterface<"MOHasDeviceTypeInterface"> {
  let cppNamespace = "::M::MO";
  let description = [{
    Interface for types with an optional `#M.device_ref` attribute that
    describes where values with the type reside.
  }];
  let methods = [
    InterfaceMethod<
      [{
        Returns the `#M.device_ref` annotation for this type, or null if
        no such annotation.
      }],
      "DeviceRefAttr", "getDeviceRef", (ins )
    >,

    InterfaceMethod<
      [{
        Returns a copy of the type with the given device annotation.
      }],
      "Type", "cloneWithNewDevice", (ins "DeviceRefAttr":$device), "",
      [{
        return $_type.cloneWith(device);
      }]
    >,
  ];
}

def MO_ShapedTypeInterface : TypeInterface<"MOShapedTypeInterface"> {
  let cppNamespace = "::M::MO";
  let description = [{
    Interface for shaped types that have data type and shape information.
  }];
  let methods = [
    InterfaceMethod<
      [{
        Returns the element data type of this shaped type.
      }],
      "DType", "getDtype", (ins )
    >,

    InterfaceMethod<
      [{
        Returns the rank of this shaped type.
      }],
      "size_t", "getRank", (ins )
    >,

    InterfaceMethod<
      [{
        Returns the rank of this shaped type, excluding unknown dimensions.
      }],
      "size_t", "getRankNonZero", (ins )
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPEINTERFACES_TD
