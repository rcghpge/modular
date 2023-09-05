//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the MEFDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_MEFDIALECT_MEFDIALECT_TD
#define GRAPHRT_MEFDIALECT_MEFDIALECT_TD

include "mlir/IR/DialectBase.td"

def MEFDialect : Dialect {
  let name = "mef";

  let summary = "The 'mef' dialect for the ModularExecutorFormat";
  let description = [{
    This dialect defines the `mef` dialect, which captures operations core to
    the GraphRT "MEF" executor.
  }];

  let cppNamespace = "::M::MEF";

  let useDefaultTypePrinterParser = 1;
  // TODO: Remove this once dialects have been updated to use properties.
  let usePropertiesForAttributes = 0;
}

#endif // GRAPHRT_MEFDIALECT_MEFDIALECT_TD
