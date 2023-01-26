//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
#define GENERICML_MGPDIALECT_MGPTRANSFORMS_TD

include "mlir/Pass/PassBase.td"
include "MGPOps.td"

def PeepholeTensorExtract : Pass<"peephole-tensor-extract", "MGP::ModelOp"> {
  let summary = "Removes redundant tensor extract operations from the graph.";
  let description = [{
    If a tensor is created only to be extracted then we 
    should elide both and just refer to the underlying
    buffer and spec.
  }];
}

#endif // GENERICML_MGPDIALECT_MGPTRANSFORMS_TD
