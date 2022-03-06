//===- GraphRT/GRTDialect/GRTIntegerOps.td -----------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines integer operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantI32Op
//===----------------------------------------------------------------------===//

def ConstantI32Op : GRTOp<"constant.i32"> {
  let summary = "constant 32-bit integer";
  let description = [{
    The "graphrt.constant.i32" operation produces a 32-bit result in an int32_t
    AsyncValue.
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// AddI32Op
//===----------------------------------------------------------------------===//

def AddI32Op : GRTOp<"add.i32"> {
  let summary = "Add two int32_t values";
 
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// PrintI32Op
//===----------------------------------------------------------------------===//

def PrintI32Op : GRTOp<"print.i32"> {
  let summary = "Print an int32_t value";
 
  let arguments = (ins I32:$value, ChainType:$chain);
  let results = (outs ChainType:$result);
  let assemblyFormat = "$value `,` $chain attr-dict";
}
