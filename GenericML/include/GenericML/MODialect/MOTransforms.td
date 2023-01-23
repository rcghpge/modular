//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def ResolveUnknownParameters : Pass<"resolve-unknown-parameters", "ModuleOp"> {
  let summary = "Attempts to resolve unknown tensor shape and dimension parameters.";
  let description = [{
    Pass to parameterize `mo` ops. That is, replaces unknown shape and dimension
    parameters (i.e. `?`) in `!mo.tensor` instances with parameter references.
  }];
}

#endif // GENERICML_MODIALECT_MOTRANSFORMS_TD
