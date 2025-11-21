//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the test operations in the GRT dialect.
//
//===----------------------------------------------------------------------===//

include "GraphRT/GRTDialect/GRTOpsBase.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// TestcaseOp
//===----------------------------------------------------------------------===//

def TestcaseOp : GRTOp<"testcase",
          [Symbol, IsolatedFromAbove,
           SingleBlockImplicitTerminator<"MEF::OutputOp">]> {
  let summary = "A GRT testcase";
  let description = [{
    The 'grt.testcase' op provides a container that takes no arguments and can
    produce any result. It holds a single region with a single block that
    contains the code under test. The executor can iterate the 'grt.testcase'
    ops in a MEF file to run all the tests in the MEF file.
  }];

  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// DebugPrintNumRefs
//===----------------------------------------------------------------------===//

def DebugPrintNumRefs : GRTOp<"debug.print_num_refs", [
    AllTypesMatch<["input", "result"]>]> {
  let summary = "Prints the number of references to its operand, and returns "
                "the operand.";
  let description = [{
    This op prints to stdout the number of references to its operand,
    and returns the operand unchanged. It is intended only for use in
    unit tests for the correct handling of unique references.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
}
