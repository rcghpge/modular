//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOTRANSFORMS_TD
#define GENERICML_MODIALECT_MOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def ForwardShapeInferenceMaterialization : Pass<"materialize-fw-shape-funcs", "ModuleOp"> {
  let summary = "Materializes the forward shape inference functions.";
  let description = [{
    Pass to materialize forward shape inference functions for ops that have
    them, and add invocations to the graph. The ops need to be fully parametric
    (i.e. unknown shapes or dimensions in the arguments and results are not
    allowed). The shape inference functions are not evaluated.
  }];
}

def ResolveUnknownParameters : Pass<"resolve-unknown-parameters", "ModuleOp"> {
  let summary = "Attempts to resolve unknown tensor shape and dimension parameters.";
  let description = [{
    Pass to parameterize `mo` ops. That is, replaces unknown shape and dimension
    parameters (i.e. `?`) in `!mo.tensor` instances with parameter references.
  }];
}

#endif // GENERICML_MODIALECT_MOTRANSFORMS_TD
