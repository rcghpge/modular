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

  // https://discourse.llvm.org/t/psa-new-improved-fold-method-signature-has-landed-please-update-your-downstream-projects/67618
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // GRAPHRT_GRTDIALECT_GRTDIALECT_TD
