//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD

include "mlir/IR/DialectBase.td"
include "KGEN/KGENDialect/KGENDialect.td"

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

def MODialect : Dialect {
  let name = "mo";

  let summary = "Modular high level graph operations.";
  let description = [{
    Common framework-independent graph level IR and compiler optimizations
    for the Modular stack.
    The MO Dialect is intended to provide a centralized dialect for accessing
    Modular optimizations. It will provide an HLO-like IR that framework graphs
    can lower to, common graph level optimziation passes, tools for kernel
    selection, and lowering to multi-backend execution through MGP.

    (Rough design is currently in
    https://www.notion.so/modularai/Graph-Compiler-Design-Points-8db653075b7e4994a0916ef12f8b9083
    which will be moved into GenericML/docs/mo.md as things firm up.)
  }];

  let cppNamespace = "::M::MO";
  let dependentDialects = [
    "MOSH::MOSHDialect",
    "MIP::MIPDialect",
  ];
  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
  private:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect types.
    void registerTypes();
    /// Register all dialect ops.
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
