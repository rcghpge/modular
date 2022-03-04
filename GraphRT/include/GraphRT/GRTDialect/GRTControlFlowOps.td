//===- GraphRT/GRTDialect/GRTControlFlowOps.td -------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

def ChainNewOp : GRTOp<"chain.new"> {
  let summary = "Return a new !grt.chain";

  let arguments = (ins);
  let results   = (outs ChainType:$result);
  let assemblyFormat = "attr-dict";
}

def CallOp : GRTOp<"call"> {
  let summary = "Call a BEF function";

  let arguments = (ins SymbolRefAttr:$fn, Variadic<AnyType>:$operands);
  let results   = (outs Variadic<AnyType>:$results);
  let assemblyFormat =
    "$fn `(` $operands `)` attr-dict `:` functional-type(operands, results)";
}
