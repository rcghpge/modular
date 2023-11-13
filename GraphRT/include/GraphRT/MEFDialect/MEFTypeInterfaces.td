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

def MEF_AnyTypeInterface : TypeInterface<"MEFAnyTypeInterface"> {
  let description = [{
    This interface allows foreign types to be used as model input and output
    types. These types are represented in the runtime by AnyAsyncValueRefs of
    arbitrary type.
  }];
  let cppNamespace = "::M::MEF";
}

#endif  // GRAPHRT_MEFDIALECT_MEFTYPEINTERFACES_TD
