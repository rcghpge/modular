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

def GRT_ChainCreateOpInterface : OpInterface<"GRTChainCreateOpInterface"> {
  let description = [{
    Operation that creates a new chain.
  }];
  let cppNamespace = "::M::GRT";
  let methods = [
    InterfaceMethod<
      "Get the resulting chain",
      "::mlir::Value", "getChainResult", (ins), "return $_op.getResult();"
    >,
    InterfaceMethod<
      "Get the input chains that this chain depends on",
      "::mlir::ValueRange", "getInputChains", (ins)
    >
  ];
}

def GRT_GatedOnChainOpInterface : OpInterface<"GRTGatedOnChainOpInterface"> {
  let description = [{
    Chain operations that depend on other chains/inputs. Allows changing what
    is being depended on. Ops implementing this interface must have the same
    type input chains as output chains.
  }];
  let cppNamespace = "::M::GRT";
  let methods = [
    InterfaceMethod<
      "Set the inputs that this chain depends on",
      "void", "setInputChains", (ins "::mlir::ValueRange":$inChains)
    >
  ];
}

#endif // GRAPHRT_GRTDIALECT_GRTCHAININTERFACE_TD
