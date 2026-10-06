//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD

include "mlir/IR/DialectBase.td"
include "Mojo/KGENDialect/KGENDialect.td"
include "Support/MDialect/MDialect.td"

//===----------------------------------------------------------------------===//
// MODialect
//===----------------------------------------------------------------------===//

def MODialect : Dialect {
  // TODO(GEX-4321): Migrate assembly formats to strict properties before
  // upstream removes this opt-out.
  let useStrictPropertiesInAssemblyFormat = 0;
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
    void registerAttributes();
    void registerTypes();
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
