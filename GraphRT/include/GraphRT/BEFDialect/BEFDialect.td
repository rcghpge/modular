//===- BEFDialect.td - BEF dialect definition --------------*- tablegen -*-===//
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

def BEFDialect : Dialect {
  let name = "bef";

  let summary = "Types and operations for the BEF dialect";
  let description = [{
    This dialect defines the `bef` dialect, which captures operations and
    attributes core to the GraphRT "BEF" executor.
  }];

  let cppNamespace = "::M::BEF";
}

#endif // GRAPHRT_BEFDIALECT_BEFDIALECT_TD
