//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for MGP ops which may also be used by other dialects which
// have similar buffer and tensor ops.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPOPINTERFACES_TD
#define GENERICML_MGPDIALECT_MGPOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def MGP_BufferAllocOpInterface : OpInterface<"MGPBufferAllocOpInterface"> {
  let description = [{
    Interface for ops which allocate or reallocate (via an !mgp.context slot)
    buffers.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns the input chain gating this op's allocation.",
      "mlir::Value", "getInChain", (ins)
    >,
    InterfaceMethod<
      "Sets the input chain gating this op's allocation.",
      "void", "setInChain", (ins "mlir::Value":$chain)
    >,
    InterfaceMethod<
      "Returns the buffer allocated by this op.",
      "mlir::Value", "getOutBuffer", (ins)
    >,
  ];
}

def MGP_LiftableBufferCreateOpInterface : OpInterface<"MGPLiftableBufferCreateOpInterface"> {
  let description = [{
    Interface for ops which return a buffer, either by allocation or
    construction (eg a constant). Such ops can typically be moved to the
    'init' block, and the resulting buffer can be cached within an
    `!mgp.context` slot if necessary.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Returns true if op is liftable to the 'init' block.",
      "bool", "isLiftable", (ins), "return true;"
    >,
  ];
}

def MGP_KernelExecuteOpInterface : OpInterface<"MGPKernelExecuteOpInterface"> {
  let description = [{
    Interface for ops which execute a kernel, possibly indirectly.
  }];
  let cppNamespace = "::M::MGP";
}

def MGP_TensorExtractOpInterface : OpInterface<"MGPTensorExtractOpInterface"> {
  let description = [{
    Interface for ops which extract a buffer and a tensor spec from a tensor.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the resulting buffer",
      "::mlir::Value", "getBufferResult", (ins), "return $_op.getBufferResult();"
    >,
    InterfaceMethod<
      "Get the resulting tensor spec",
      "::mlir::Value", "getTensorSpecResult", (ins), "return $_op.getTensorSpecResult();"
    >,
    InterfaceMethod<
      "Get the input tensor",
      "::mlir::Value", "getTensor", (ins), "return $_op.getTensor();"
    >
  ];
}

def MGP_CreateTensorOpInterface : OpInterface<"MGPCreateTensorOpInterface"> {
  let description = [{
    Interface for ops which create a tensor from a buffer, spec, and chain.
  }];
  let cppNamespace = "::M::MGP";
  let methods = [
    InterfaceMethod<
      "Get the input buffer",
      "::mlir::Value", "getBuffer", (ins), "return $_op.getBuffer();"
    >,
    InterfaceMethod<
      "Get the input chain",
      "::mlir::Value", "getInChain", (ins), "return $_op.getInChain();"
    >,
    InterfaceMethod<
      "Get the input tensor spec",
      "::mlir::Value", "getTensorSpec", (ins), "return $_op.getTensorSpec();"
    >,
    InterfaceMethod<
      "Get the output tensor",
      "::mlir::Value", "getResult", (ins), "return $_op.getResult();"
    >
  ];
}

#endif // GENERICML_MGPDIALECT_MGPOPINTERFACES_TD
