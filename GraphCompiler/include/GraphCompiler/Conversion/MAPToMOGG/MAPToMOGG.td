//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD
#define GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD

include "mlir/Pass/PassBase.td"

def MAPToMOGG : Pass<"map-to-mogg", "mlir::ModuleOp"> {
  let summary = "Lower `map` dialect kernels into `mogg` codegen ops.";
  let description = [{
    Lowers `map` dialect kernels into `mogg` codegen ops: dynamic dims become
    SSA values, contexts are added, and tensors are built from pointers. This is
    where the fusion-focused `map` IR crosses into the code-generation-focused
    `mogg` IR.

    This is scaffolding gated on `M::MO::useAdvancedFusion()`; it currently does
    nothing.
  }];

  let dependentDialects = [
    "M::MAP::MAPDialect",
    "M::MOGG::MOGGDialect",
    "M::MO::MODialect",
    "M::MOSH::MOSHDialect",
  ];
}

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MAPTOMOGG_TD
