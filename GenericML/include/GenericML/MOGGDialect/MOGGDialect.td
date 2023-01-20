//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOGGDIALECT_MOGGDIALECT_TD
#define GENERICML_MOGGDIALECT_MOGGDIALECT_TD

include "mlir/IR/DialectBase.td"
//include "GenericML/MODialect/MODialect.td"

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
  }];

  let useDefaultTypePrinterParser = 0;

  // https://discourse.llvm.org/t/psa-new-improved-fold-method-signature-has-landed-please-update-your-downstream-projects/67618
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // GENERICML_MOGGDIALECT_MOGGDIALECT_TD
