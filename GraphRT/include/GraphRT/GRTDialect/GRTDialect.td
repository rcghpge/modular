//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the GraphRTDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTDIALECT_TD
#define GRAPHRT_GRTDIALECT_GRTDIALECT_TD

include "mlir/IR/DialectBase.td"

def GRTDialect : Dialect {
  let name = "grt";

  let summary =
    "The 'grt' dialect holds definitions of widely used primitive ops.";
  let description = [{
    This is the `grt` dialect, which holds commonly used primitives for
    working with integers and implementing control flow.
  }];

  let cppNamespace = "::M::GRT";

  // We depend on the MEF dialect for the chain type.
  let dependentDialects = ["::M::MEF::MEFDialect"];
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // GRAPHRT_GRTDIALECT_GRTDIALECT_TD
