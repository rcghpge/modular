//===- GraphRTControlFlowOps.td ----------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GraphRT dialect.
//
//===----------------------------------------------------------------------===//

def ChainNewOp : GraphRTOp<"chain.new"> {
  let summary = "Return a new !graphrt.chain";

  let arguments = (ins);
  let results   = (outs ChainType:$result);
  let assemblyFormat = "attr-dict";
}
