//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// An interface for chains.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD
#define GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD

include "mlir/IR/OpBase.td"

def GRT_ChainOpInterface : OpInterface<"GRTChainOpInterface"> {
  let description = [{
    Operation that creates a chain.
  }];
  let cppNamespace = "::M::GRT";
  let methods = [
    InterfaceMethod<
      "Get the resulting chain",
      "::mlir::Value", "getChainResult", (ins), "return $_op.getResult();"
    >
  ];
}

#endif // GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD
