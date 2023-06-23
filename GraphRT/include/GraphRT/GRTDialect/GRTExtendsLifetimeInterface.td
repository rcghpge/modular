//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD
#define GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTExtendsLifetimeInterface : OpInterface<"GRTExtendsLifetimeInterface"> {
  let description = [{
    This interface indicates an operation may extend the lifetime of its
    operands to include the lifetime of its results. That is, a result
    may point to an AsyncValue which holds a pointer to an operand AsyncValue.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";
}

#endif // GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD
