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

def MO_KernelOperandTypeInterface : TypeInterface<"MOKernelOperandTypeInterface"> {
  let cppNamespace = "::M::MO";
  let description = [{
    Classifies a type as a kernel-operand category (tensor / buffer / scalar /
    opaque) for operand inference, so inference queries the type instead of
    hard-coding `isa<MO::...>`. MOGG codegen handles implement this too, which is
    what lets inference run on the advanced-fusion path. Rank / on-host
    validation of a scalar match is left to `MOShapedTypeInterface` /
    `MOHasDeviceTypeInterface`; a type that carries neither (an opaque handle) is
    trusted (already validated upstream).
  }];
  let methods = [
    InterfaceMethod<
      [{ Returns the kernel-operand category of this type. }],
      "::M::MO::KernelOperandKind", "getKernelOperandKind", (ins)
    >,

    InterfaceMethod<
      [{ For an `Opaque`-kind type, its symbol name; empty otherwise. }],
      "mlir::StringAttr", "getKernelOperandOpaqueName", (ins), "",
      [{ return {}; }]
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOTYPEINTERFACES_TD
