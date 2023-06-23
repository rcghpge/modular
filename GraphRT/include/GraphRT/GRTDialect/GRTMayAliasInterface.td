//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD
#define GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTMayAliasInterface : OpInterface<"GRTMayAliasInterface"> {
  let description = [{
    This interface indicates an operation may return aliases to its operands.
    That is, a result may be a pointer to an operand AsyncValue.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";
}

#endif // GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD
