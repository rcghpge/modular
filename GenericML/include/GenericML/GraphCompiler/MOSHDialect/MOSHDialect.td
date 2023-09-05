//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MOSHDialect
//===----------------------------------------------------------------------===//

def MOSHDialect : Dialect {
  let name = "mosh";

  let summary = "Modular high level shape dialect.";
  let description = [{
    Dialect to provide types and operations for writing shape functions and
    shape inference infrastructure.

    This dialect should not depend on the MO dialect.
  }];

  let cppNamespace = "::M::MOSH";
  let dependentDialects = [
    "MDialect",
    "KGEN::KGENDialect",
    "KGEN::POP::POPDialect",
    "mlir::index::IndexDialect",
    "mlir::arith::ArithDialect",
  ];
  let useDefaultAttributePrinterParser = 1;
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
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // GENERICML_GRAPHCOMPILER_MOSHDIALECT_MOSHDIALECT_TD
