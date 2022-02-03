//===- BEF.td - BEF dialect definition ---------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the BEF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef BEF_TD
#define BEF_TD

include "mlir/IR/OpBase.td"
include "GraphRT/BEFDialect/BEFDialect.td"

// Base class for the operation in this dialect.
class BEFOp<string mnemonic, list<Trait> traits = []> :
    Op<BEFDialect, mnemonic, traits>;

def ReturnOp : BEFOp<"return", [Terminator]> {
  let summary = "BEF return operation";
  let description = [{
    The "bef.return" operation captures the return values for a BEF function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the BEF function
    that contains the operation. For example:

      func @foo() : i32, f8 {
        ...
        bef.return %0, %1 : i32, f8
      }
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  // TODO: Verifier.
}

#endif // BEF_TD
