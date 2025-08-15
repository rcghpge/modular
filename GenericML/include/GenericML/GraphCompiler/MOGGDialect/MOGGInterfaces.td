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
  ];
}

def MOGG_KernelTerminator : OpInterface<"KernelTerminator"> {
  let description = [{
    A terminator for `mogg.experimental.kernel` operation.

    A kernel terminator represents a call to a particular registered mojo
    function with the following calling convention:

    ```mojo
    // mogg.kernel_terminator input(%1, %2) dps_output(%3) -> mo.opaque(...)
    fn kernel[...]  (
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

def MOGG_EmitsMojoInterface : OpInterface<"EmitsMojoInterface"> {
  let description = [{
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Return the body of the kernel, which should be the code we will compile",
      "mlir::LogicalResult", "emitMojo", (ins "M::MOGG::EmitMojoState&":$state)
    >,
  ];
}

def MOGG_CompilableKernel : OpInterface<"CompilableKernel"> {
  let description = [{
    Implemented by operations which represent a compilable kernel so multiple
    operations can be compiled rather than just MOGG::KernelOp.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Return the body of the kernel, which should be the code we will compile",
      "mlir::Block&", "getBody", (ins), "return $_op.getBody().front();"
    >,
    InterfaceMethod<
      "Get output parameter decls",
      "::llvm::ArrayRef<M::KGEN::ParamDeclAttr>", "getOutputParamDecls", (ins),
      "return $_op.getOutputParamDecls();"
    >,
    InterfaceMethod<
      "Get output parameter decls attr",
      "::M::KGEN::ParamDeclArrayAttr", "getOutputParamDeclsAttr", (ins),
      "return $_op.getOutputParamDeclsAttr();"
    >,
    InterfaceMethod<
      "Get the device attr for this kernel",
      "::M::DeviceRefAttr", "getExecutionDevice", (ins),
      "return $_op.getExecutionDevice();"
    >,
    InterfaceMethod<
      "Returns true if kernel is to be implemented as a Mojo async function.",
      "bool", "getIsAsync", (ins),
      "return $_op.getIsAsync().value_or(false);"
    >,
    InterfaceMethod<
      "Returns the attached constant resources to the kernel",
      "mlir::ArrayAttr", "getConstantResources", (ins), "return $_op.getConstantResourcesAttr();"
    >,
  ];
}

def MOGG_MojoObjectInfosInterface : OpInterface<"MojoObjectInfosInterface"> {
  let description = [{
    Implemented by operations which might take a mojo object.
    Carry extra informations about the mojo object.
    These informations are extracted first when manipulating the kernels,
    but they are needed later when lowering to KGEN.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns true if the mojo object is in a register / value, or false if it's in memory",
      "bool", "isMojoRegisterType", (ins), "return $_op.getIsRegisterType();"
    >,
    InterfaceMethod<
      "Returns the parametrized methods needed for this mojo object (or none if not found)",
      "std::optional<mlir::DictionaryAttr>", "getMojoTypeMethodsDict", (ins), "return $_op.getTypeMethodsDict();"
    >,
    InterfaceMethod<
      "Returns the value that represents the lowered KGEN mojo object",
      "mlir::Value", "getKGENMojoValue", (ins), "return $_op.getKGENMojoValueImpl();"
    >,
    InterfaceMethod<[{
      Returns the method `methodName` stored in `typeMethodsDict`.
      Returns nullptr if there isn't a methods dict, or `methodName` isn't in the vtable.
    }], "mlir::TypedAttr", "getMethodOfMojoType", (ins "StringRef":$methodName),
    [{}], [{
      if (!$_op.getTypeMethodsDict().has_value())
        return nullptr;
      DictionaryAttr methodsDict = *$_op.getTypeMethodsDict();
      return dyn_cast_or_null<TypedAttr>(methodsDict.get(methodName));
    }]>
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
