//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Interfaces for MEF types which may also be used by other dialects which
// have similar chain types.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD
#define GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def MEF_ChainTypeInterface : TypeInterface<"MEFChainTypeInterface"> {
  let description = [{
    Interface for types which support chain-like operations.
  }];
  let cppNamespace = "::M::MEF";
}

#endif  // GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD
