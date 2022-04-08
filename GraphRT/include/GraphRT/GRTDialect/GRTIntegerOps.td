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
// ConstantI1Op
//===----------------------------------------------------------------------===//

def ConstantI1Op : GRTOp<"constant.i1"> {
  let summary = "constant 1-bit integer";
  let description = [{
    The "graphrt.constant.i1" operation produces a 1-bit result in a bool
    AsyncValue.
  }];

  let arguments = (ins I1Attr:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantI32Op
//===----------------------------------------------------------------------===//

def ConstantI32Op : GRTOp<"constant.i32"> {
  let summary = "constant 32-bit integer";
  let description = [{
    The "graphrt.constant.i32" operation produces a 32-bit result in an int32_t
    AsyncValue.
  }];

  let arguments = (ins SI32Attr:$value);
  let results = (outs SI32:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// AddI32Op
//===----------------------------------------------------------------------===//

def AddI32Op : GRTOp<"add.i32"> {
  let summary = "Add two int32_t values";
 
  let arguments = (ins SI32:$lhs, SI32:$rhs);
  let results = (outs SI32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// LessEqualI32Op
//===----------------------------------------------------------------------===//

def LessEqualI32Op : GRTOp<"lessequal.i32"> {
  let summary = "Compare two int32_t values for `<=` and return a bool";
 
  let arguments = (ins SI32:$lhs, SI32:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// PrintI32Op
//===----------------------------------------------------------------------===//

def PrintI32Op : GRTOp<"print.i32"> {
  let summary = "Print an int32_t value";
 
  let arguments = (ins SI32:$value, ChainType:$chain);
  let results = (outs ChainType:$result);
  let assemblyFormat = "$value `,` $chain attr-dict";
}
