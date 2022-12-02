//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"

def MOToPrimitives : Pass<"mo-to-primitives", "ModuleOp"> {
  let summary = "Convert `mo` dialect to primitives.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to `mgp` and
    `bef` primitives.
  }];
  let dependentDialects = ["BEF::BEFDialect",
                           "MGP::MGPDialect",
                           "GRT::GRTDialect"];
}
