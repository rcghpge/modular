//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This declares the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

def MAPDialect : Dialect {
  let name = "map";

  let summary = "Modular Access Primitives dialect.";
  let description = [{
    MAP (Modular Access Primitives) expresses executable loop nests / kernels
    that load and store tensors and perform computation. It works together with
    the companion IMAP (Index Modular Access Primitives) dialect, which
    expresses index transformations.

    This is scaffolding: the dialect currently defines no operations, attributes
    or types.
  }];

  let cppNamespace = "::M::MAP";

  let useDefaultTypePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD
