//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines interfaces on tensor types in the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTENSORTYPEINTERFACE_TD
#define GENERICML_MGPDIALECT_MGPTENSORTYPEINTERFACE_TD

include "mlir/IR/OpBase.td"

def MGP_TensorTypeInterface : TypeInterface<"MGPTensorTypeInterface"> {
  let description = [{
    Types that represent runtime tensors.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the tensor spec", "::M::MGP::CpuTensorSpecType", "getTensorSpec", (ins), "return $_type.getTensorSpec();"
    >
  ];
}

#endif  // GENERICML_MGPDIALECT_MGPTENSORTYPEINTERFACE_TD
