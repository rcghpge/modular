//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MOGGDialect
//===----------------------------------------------------------------------===//

def MOGGDialect : Dialect {
  let name = "mogg";

  let summary = "Modular operations graph generators.";
  let description = [{
    MOGG specifies which kernel implements each op, which gives fusion enough
    information about the kernels' numerics to decide whether fusing is safe.
    Fusion embeds kernels in the output region of the root kernel. Selected
    kernels then lower, generally in buffer semantics, to MO for integration
    with framework kernels, memory optimization, and lowering to MEF.
  }];

  let cppNamespace = "::M::MOGG";
  let dependentDialects = ["::M::MO::MODialect", "::M::DebugInfo::DebugInfoDialect"];
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerOperations();
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGDIALECT_TD
