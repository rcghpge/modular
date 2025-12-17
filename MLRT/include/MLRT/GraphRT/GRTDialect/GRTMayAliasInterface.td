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
    Here 'alias' includes:
      - 'direct' aliasing: a result may be a pointer to an operand AsyncValue.
         For example, the identity op.
      - 'internal' aliasing: a result is a pointer to a fresh AsyncValue which
         shares an underlying resource with an operand AsyncValue. For example,
         an op to create a tensor from a buffer which may move the buffer if
         it has no other references.

    The AsyncDataflowDependencies analysis requires this information so as
    to soundly approximate the 'exclusive' relation between SSA values.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns true if operand with index operandIndex may alias with result
        resultIndex.
      }],
      "bool", "mayAlias", (ins "size_t": $operandIndex, "size_t": $resultIndex)
    >,
  ];
}

#endif // GRAPHRT_GRTDIALECT_GRTMAYALIASINTERFACE_TD
