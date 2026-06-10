//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD

include "mlir/IR/OpBase.td"

def MOGG_KernelCallInterface : OpInterface<"KernelCallInterface"> {
  let description = [{
    An interface that represents ops that call into Mojo kernels directly.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Inject a dictionary attribute for kernel parameters",
      "void", "setKernelParams", (ins "mlir::DictionaryAttr":$params),
      "$_op->setAttr(\"kernel_param\", params);"
    >,
    InterfaceMethod<
      "Returns the value that represents the lowered KGEN mojo object",
      "mlir::DictionaryAttr", "getKernelParams", (ins),
      "return $_op->template getAttrOfType<DictionaryAttr>(\"kernel_param\");"
    >,
    InterfaceMethod<
      "Returns the sym_name of the kernel decl this op references, or null "
      "if this op does not directly reference a kernel decl.",
      "mlir::StringAttr", "getKernelSymNameAttr", (ins),
      "return $_op->template getAttrOfType<mlir::StringAttr>(\"kernelSymName\");"
    >,
  ];
}

def MOGG_StubCallOpInterface : OpInterface<"StubCallOpInterface"> {
  let description = [{
    Common interface for ops that call an outlined `mogg.kernel.stub` by symbol
    (`mogg.call.stub` and `mogg.call.stub.async`). The callee symbol is stored
    as the `$name` `FlatSymbolRefAttr`, so the default implementations forward
    to the generated `getNameAttr`/`setNameAttr` accessors.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns the symbol reference of the callee kernel stub.",
      "mlir::FlatSymbolRefAttr", "getCalleeSymbol", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getNameAttr(); }]
    >,
    InterfaceMethod<
      "Sets the symbol reference of the callee kernel stub.",
      "void", "setCalleeSymbol", (ins "mlir::FlatSymbolRefAttr":$callee),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ $_op.setNameAttr(callee); }]
    >,
  ];
}

def MOGG_KernelTerminator : OpInterface<"KernelTerminator"> {
  let description = [{
    A terminator for `mogg.kernel` operation.

    A kernel terminator represents a call to a particular registered mojo
    function with the following calling convention:

    ```mojo
    // mogg.kernel_terminator input(%1, %2) dps_output(%3) -> mo.opaque(...)
    def kernel[...]  (
       out : OutputTensor[...] // <- %3

       in0 : InputTensor[...]  // <- %1
       in1 : Opaque[...]       // <- %2

    ) -> Opaque[...] {         // <- kernel returns.
       ...
    }
    ```
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
     InterfaceMethod<
      "Returns the device context that this op uses. Can be nullptr if not needed.",
      "mlir::Value", "getDeviceContext", (ins)
    >,
    InterfaceMethod<
      "Returns the dps output values of the kernel (a argument but with dps semantics)",
      "mlir::SmallVector<mlir::Value>", "getKernelDPSOutputs", (ins)
    >,
    InterfaceMethod<
      "Returns the inputs values of the kernel",
      "mlir::SmallVector<mlir::Value>", "getKernelInputs", (ins)
    >,
    InterfaceMethod<
      "Returns the type of kernel returned value (either a by_val/by_ref return)",
      "mlir::Type", "getKernelReturnOutput", (ins)
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
