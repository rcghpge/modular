//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD

include "mlir/Pass/PassBase.td"

def MOToMAP : Pass<"mo-to-map", "mlir::ModuleOp"> {
  let summary = "Lower `mo` dialect ops into `map` kernels.";
  let description = [{
    This pass lowers known `mo` dialect ops into one `map` kernel each,
    as the entry point of the advanced fusion pipeline.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::MAP::MAPDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTOMAP_TD
