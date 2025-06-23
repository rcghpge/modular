//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD

include "mlir/IR/DialectBase.td"
include "KGEN/KGENDialect/KGENDialect.td"
include "Support/MDialect/MDialect.td"

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

def MODialect : Dialect {
  let name = "mo";

  let summary = "MAX Graph operations.";
  let description = [{
    The MO Dialect represents a MAX graph and its operations. It is a frontend
    dialect that serves as input to the graph compiler.
  }];

  let cppNamespace = "::M::MO";
  let dependentDialects = [
    "M::MDialect",
    "MOSH::MOSHDialect",
  ];
  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

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
