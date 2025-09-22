//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
#define GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD

include "mlir/IR/OpBase.td"

def EmitsMojoInterface : OpInterface<"EmitsMojoInterface"> {
  let description = [{
    An op that implements this interface will ultimately lower to Mojo text.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Return the body of the kernel, which should be the code we will compile",
      "mlir::LogicalResult", "emitMojo", (ins "M::MOGG::EmitMojoState&":$state)
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_INTERFACES_EMITSMOJOINTERFACE_TD
