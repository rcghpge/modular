//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD
#define GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD

include "mlir/IR/OpBase.td"

def NonStrictInterface : OpInterface<"NonStrictInterface"> {
  let cppNamespace = "::M::MEF";

  let description = [{
    This interface indicates to the MEF execution engine that the op may be
    scheduled to BEGIN execution before all of its operands are ready. The op
    itself then takes responsibility for correctly sequencing computation as its
    arguments become ready.

    Ops without this interface are only scheduled once all operands are ready.
    There's nothing the op implementation needs to do in this case.

    This interface is used at both runtime (by the MEF executor) and compile
    time (by dataflow analysis).

    See also ChainIndicatesFinishedInterface, which describes how op result
    connect with op execution.
  }];
}

#endif // GRAPHRT_MEFDIALECT_MEFNONSTRICTINTERFACE_TD
