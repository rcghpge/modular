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
      "$_op.setKernelParamAttr(params);"
    >,
    InterfaceMethod<
      "Returns the value that represents the lowered KGEN mojo object",
      "mlir::DictionaryAttr", "getKernelParams", (ins),
      "return $_op.getKernelParamAttr();"
    >,
    InterfaceMethod<
      "Returns the sym_name of the kernel decl this op references.",
      "mlir::StringAttr", "getKernelSymNameAttr", (ins),
      "return $_op.getKernelSymNameAttr();"
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

def MOGG_TensorCreateInterface : OpInterface<"TensorCreateInterface"> {
  let description = [{
    Common interface for the five `mogg._tensor.create*` ops (the plain
    tensor and each view kind: slice/broadcast/transpose/reshape) -- all
    carry `$elementType`/`$mut` as identically-named attributes, so a
    capture's producer can be read through this instead of dispatched on
    each concrete op type.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns the element type of the produced tensor.",
      "::mlir::Type", "getElementType", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getElementType(); }]
    >,
    InterfaceMethod<
      "Returns whether the produced tensor is mutable.",
      "bool", "getMut", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getMut(); }]
    >,
    InterfaceMethod<
      "Sets whether the produced tensor is mutable.",
      "void", "setMut", (ins "bool":$mut), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ $_op.setMut(mut); }]
    >,
    InterfaceMethod<
      [{Clone this create op with a new mutability, adjusting the result kind
        (`!mogg.tensor_` when made immutable) so the buffer-implies-mutable
        invariant holds. Returns the clone; the original is left for DCE.}],
      "::M::MOGG::TensorCreateInterface", "cloneWithMut",
      (ins "::mlir::OpBuilder &":$builder, "bool":$mut), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        auto cloned = ::mlir::cast<ConcreteOp>(builder.clone(*$_op.getOperation()));
        // Concrete `setMut` writes the `mut` property `getMut` reads (and errs
        // at compile time if an op ever drops it); a raw `setAttr` would not.
        cloned.setMut(mut);
        ::mlir::Value result = cloned->getResult(0);
        // A buffer result is mutable memory; retype to `!mogg.tensor_` when the
        // clone becomes immutable to keep `!mogg.buffer_ => mut`.
        if (!mut && ::mlir::isa<::M::MOGG::BufferType>(result.getType()))
          result.setType(::M::MOGG::TensorType::get(cloned->getContext()));
        return ::mlir::cast<::M::MOGG::TensorCreateInterface>(
            cloned.getOperation());
      }]
    >,
  ];

  // A `!mogg.buffer_` result is mutable memory -- `getKernelOperandKind()`
  // reads the mutable `Buffer` kind straight off the type -- so it must be
  // marked mutable. Enforced on the interface (rather than one op) so every
  // current and future `mogg._tensor.create*` producer is covered.
  //
  // The converse is intentionally NOT checked: a `!mogg.tensor_` may be
  // mutable too -- a DPS output (`io = Output`) and a mutable view are both
  // mutable `!mogg.tensor_` -- so `mut` does not pin the kind down the other
  // way, and only the plain `mogg._tensor.create` even carries the `$io` that
  // scoping an input-only immutability rule would need.
  //
  // Defined out of line in MOGGOps.cpp so it can name `!mogg.buffer_`, which
  // this generated interface header cannot see.
  let verify = [{
    return ::M::MOGG::verifyTensorCreateInterfaceMutability($_op);
  }];
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
