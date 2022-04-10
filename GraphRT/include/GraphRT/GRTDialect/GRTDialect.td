//===- GraphRT/GRTDialect/GRTDialect.td --------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This contains the GraphRTDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTDIALECT_TD
#define GRAPHRT_GRTDIALECT_GRTDIALECT_TD

def GRTDialect : Dialect {
  let name = "grt";

  let summary =
    "The 'grt' dialect holds definitions of widely used primitive ops.";
  let description = [{
    This is the `grt` dialect, which holds commonly used primitives for
    working with integers and implementing control flow.
  }];

  let cppNamespace = "::M::GRT";

  // Use LLVM-style accessor prefixes.
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // GRAPHRT_GRTDIALECT_GRTDIALECT_TD
