//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD
#define GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD

include "mlir/IR/OpBase.td"

def ChainIndicatesFinishedInterface : OpInterface<"ChainIndicatesFinishedInterface"> {
  let cppNamespace = "::M::MEF";

  let description = [{
    This interface indicates that the op is FINISHED execution when its
    `!mef.chain` results are ready, even though the ops other results may not
    yet be finished. The op must have at least one `!mef.chain` result. Here
    'FINISHED' means the op no longer holds any references to its operands,
    and any sub-tasks started by the op have completed.

    Ops without this interface are assumed to follow a stricter rule and be
    FINISHED when ANY of their results are ready. This stricter rule applies
    irrespective of whether the op is marked as `NonStrictInterface`.

    This interface is not used at runtime, only compile time (by dataflow
    analysis).

    See also NonStrictInterface, which describes how op operands interact with
    op execution.
  }];
}

#endif // GRAPHRT_MEFDIALECT_MEFCHAININDICATESFINISHED_TD
