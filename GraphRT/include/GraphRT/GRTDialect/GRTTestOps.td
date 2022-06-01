//===- GraphRT/GRTDialect/GRTTestOps.td --------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the test operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// TestcaseOp
//===----------------------------------------------------------------------===//

def TestcaseOp : GRTOp<"testcase",
          [Symbol, IsolatedFromAbove,
           SingleBlockImplicitTerminator<"BEF::OutputOp">]> {
  let summary = "A GRT testcase";
  let description = [{
    The 'grt.testcase' op provides a container that takes no arguments and can
    produce any result. It holds a single region with a single block that
    contains the code under test. The executor can iterate the 'grt.testcase'
    ops in a BEF file to run all the tests in the BEF file.
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}
