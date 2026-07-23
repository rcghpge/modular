//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD

include "mlir/IR/OpBase.td"


def ConvertibleToMGPTypeInterface : TypeInterface<"ConvertibleToMGPTypeInterface"> {
  let description = [{
    This interface allows foreign types to be converted in the MOToMGP pass.
  }];

  let cppNamespace = "::M::MGP";

  let methods = [
    InterfaceMethod<[{
      Returns the converted type of this type.
    }],
    "::mlir::Type", "convertType">,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_CONVERTIBLETOMGPTYPEINTERFACE_TD
