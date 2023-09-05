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
    The MO Dialect is intended to provide a centralized dialect for accessing
    Modular optimizations. It will provide an HLO-like IR that framework graphs
    can lower to, common graph level optimziation passes, tools for kernel
    selection, and lowering to multi-backend execution through MGP. For
    unsupported operators from external frameworks, the default framework
    implementation will be used. This will ensure long-tail coverage of user
    models.

    MO will initially focus purely on value semantics (all ops look like
    return_value = op(input_values...)), and use explicit broadcasting for
    backend lowering simplicity.
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
