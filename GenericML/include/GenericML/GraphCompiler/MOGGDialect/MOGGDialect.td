//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD

include "mlir/IR/DialectBase.td"
//include "GenericML/GraphCompiler/MODialect/MODialect.td"

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

def MOGGDialect : Dialect {
  let name = "mogg";

  let summary = "Modular operations graph generators.";
  let description = [{
  }];

  let cppNamespace = "::M::MOGG";
  let dependentDialects = ["::M::MO::MODialect"];

  let extraClassDeclaration = [{
  private:
    /// Register all dialect ops.
    void registerOperations();
    /// Register all dialect types.
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 0;
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
