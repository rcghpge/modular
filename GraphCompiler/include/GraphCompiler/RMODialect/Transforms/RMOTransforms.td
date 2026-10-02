//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the transforms for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD

include "mlir/Pass/PassBase.td"

def LegalizeRMOOps : Pass<"mo-legalize-rmo-operators", "GraphOp"> {
  let summary = "Legalize RMO operators to mo operators.";
  let description = [{
    `MOAnalogue` ops lower automatically to their MO op by rebinding each
    operand and result to the types from `inferMOTypes`. Other RMO ops lower to
    compositions of RMO ops.
  }];

  let dependentDialects = [
    "M::MO::MODialect",
    "M::RMO::RMODialect"
  ];

  let options = [
    Option<"skipRMORebind", "skip-rmo-rebinds", "bool", "false",
           "Whether to skip legalize rmo rebind operations. Useful for debugging.">,
  ];
}

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_TD
