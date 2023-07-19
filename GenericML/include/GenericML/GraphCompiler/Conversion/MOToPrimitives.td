//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD

include "mlir/Pass/PassBase.td"

def MOToPrimitives : Pass<"mo-to-primitives", "ModuleOp"> {
  let summary = "Convert `mo` dialect to primitives.";
  let description = [{
    This pass performs a conversion from known `mo` dialect ops to `mgp` and
    `mef` primitives.
  }];
  let options = [
    Option<"compiledFrameworkLabel", "compiled-framework-label", "std::string",
      /*default=*/"\"tf\"",
      "The label for resulting MGP graph. Default is tf.">,
    Option<"minimumCpuAlignment", "min-cpu-alignment", "size_t",
      /*default=*/"kPreferredMemoryAlignment",
     "Minimum alignment for all CPU buffers (both constants and allocated)">,
    Option<"includeHostTargetInfo", "include-host-target-info", "bool",
      /*default=*/"false",
     "If true, the generated model will capture the TargetInfoAttr of the "
     "compile-time host within the generated mgp.model. A runtime check will "
     "then be made to ensure the runtime host satisfies the assumptions "
     "made at compile time.">,
  ];
  let dependentDialects = ["MEF::MEFDialect",
                           "MGP::MGPDialect",
                           "GRT::GRTDialect"];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOPRIMITIVES_TD
