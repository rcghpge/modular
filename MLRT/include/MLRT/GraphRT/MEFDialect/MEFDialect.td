//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the MEFDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD

include "mlir/IR/DialectBase.td"

def MEFDialect : Dialect {
  // TODO(GEX-4321): Migrate assembly formats to strict properties before
  // upstream removes this opt-out.
  let useStrictPropertiesInAssemblyFormat = 0;
  let name = "mef";

  let summary = "The 'mef' dialect for the ModularExecutorFormat";
  let description = [{
    This dialect defines the `mef` dialect, which captures operations core to
    the GraphRT "MEF" executor.
  }];

  let cppNamespace = "::M::MEF";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD
