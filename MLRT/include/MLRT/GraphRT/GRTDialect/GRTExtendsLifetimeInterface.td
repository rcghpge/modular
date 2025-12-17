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

    However, !mef.chain operands and results are not subject to this rule.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns true if operand with index operandIndex has its lifetime
        extended to include that of the result with resultIndex.
      }],
      "bool", "extendsLifetime", (ins "size_t": $operandIndex, "size_t": $resultIndex)
    >,
  ];

}

#endif // GRAPHRT_GRTDIALECT_GRTEXTENDSLIFETIMEINTERFACE_TD
