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
    that load and store tensors and perform computation. It has two op
    families: `map.iter.opaque` wraps a hand-written Mojo kernel with a fusible
    store epilogue (`map.iter.epilogue` / `map.call.opaque`), while `map.iter.foreach`
    expresses the loop body explicitly as scalar `map.load` / `map.store` /
    `map.call.elementwise` ops per index (`map.index<N>`).
  }];

  let cppNamespace = "::M::MAP";
  let dependentDialects = [
    "::M::MO::MODialect",
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD
