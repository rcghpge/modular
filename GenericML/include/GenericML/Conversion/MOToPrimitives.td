//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_CONVERSION_MOTOPRIMITIVES_TD
#define GENERICML_CONVERSION_MOTOPRIMITIVES_TD

include "mlir/Pass/PassBase.td"

def MOToPrimitives : Pass<"mo-to-primitives", "ModuleOp"> {
  let summary = "Convert `mo` dialect to primitives.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to `mgp` and
    `bef` primitives.
  }];
  let options = [
    Option<"compiledFrameworkLabel", "compiled-framework-label", "std::string",
      /*default=*/"\"tfl\"",
      "The label for resulting MGP graph. Default is tfl.">];
  let dependentDialects = ["BEF::BEFDialect",
                           "MGP::MGPDialect",
                           "GRT::GRTDialect"];
}

#endif // GENERICML_CONVERSION_MOTOPRIMITIVES_TD
