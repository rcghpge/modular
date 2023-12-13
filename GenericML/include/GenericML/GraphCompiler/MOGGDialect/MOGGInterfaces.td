//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGINTERFACES_TD

include "mlir/IR/OpBase.td"

def MOGG_CompilableKernel : OpInterface<"CompilableKernel"> {
  let description = [{
    Implemented by operations which represent a compilable kernel so multiple
    operations can be compiled rather than just MOGG::KernelOp.
  }];

  let cppNamespace = "::M::MOGG";

  let methods = [
    InterfaceMethod<
      "Returns true if this kernel contains KGEN IR from the new experimental path",
      "bool", "isKGEN", (ins), "return $_op.getIsKGEN().value_or(false);"
    >,
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
      "Get the device attr for this kernel",
      "::M::DeviceRefAttr", "getDeviceRef", (ins), 
      "return $_op.getDeviceRef();"
    >,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
