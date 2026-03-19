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
    There are broadly two categories of RMO operators. The first are those
    analogous to an existing MO operator. These are lowered by properly
    rebinding each input and result to the types determined in the `MOAnalogue`
    interface. These operators have a lowering automatically generated as long
    as the `MOAnalogue` operator is implemented.

    The next are RMO operators which lower to other RMO operators.
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
