//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the BEFDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_BEFDIALECT_BEFDIALECT_TD
#define GRAPHRT_BEFDIALECT_BEFDIALECT_TD

include "mlir/IR/DialectBase.td"

def BEFDialect : Dialect {
  let name = "bef";

  let summary = "The 'bef' dialect for the BinaryExecutorFormat";
  let description = [{
    This dialect defines the `bef` dialect, which captures operations core to
    the GraphRT "BEF" executor.
  }];

  let cppNamespace = "::M::BEF";

  let useDefaultTypePrinterParser = 1;
}

#endif // GRAPHRT_BEFDIALECT_BEFDIALECT_TD
