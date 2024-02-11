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

  let summary = "MAX Graph operations.";
  let description = [{
    The MO Dialect represents a MAX graph and its operations. It is a frontend
    dialect that serves as input to the graph compiler.
  }];

  let cppNamespace = "::M::MO";
  let dependentDialects = [
    "MOSH::MOSHDialect",
    "MIP::MIPDialect",
  ];
  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  public:
    /// Return the module that stores the shape functions implemented in mojo.
    ModuleOp getShapeFuncModule();
  private:
    /// Register all dialect attributes.
    void registerAttributes();
    /// Register all dialect types.
    void registerTypes();
    /// Register all dialect ops.
    void registerOperations();
    /// Module for storing the shape functions implemented in mojo.
    OwningOpRef<ModuleOp> shapeFuncModule;
  }];

  let useDefaultTypePrinterParser = 1;
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
