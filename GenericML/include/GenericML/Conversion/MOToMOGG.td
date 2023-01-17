//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_MOTOMOGG_TD
#define GENERICML_CONVERSION_MOTOMOGG_TD

include "mlir/Pass/PassBase.td"

def MOToMOGG : Pass<"mo-to-mogg", "ModuleOp"> {
  let summary = "Convert `mo` dialect to mogg ops.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to registered
    `mogg` dialect ops, performing kernel selection along the way.

    TODO: Expand this to also perform lowering search
  }];
  let dependentDialects = ["MOGG::MOGGDialect"];
}

#endif // GENERICML_CONVERSION_MOTOMOGG_TD
