//===- GraphRTOps.td ---------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in GraphRT dialect.
//
//===----------------------------------------------------------------------===//

def ConstantI32Op : GraphRTOp<"constant.i32"> {
  let summary = "constant 32-bit integer";
  let description = [{
    The "graphrt.constant.i32" operation produces a 32-bit result in an int32_t
    AsyncValue.
  }];

  let arguments = (ins  I32Attr:$value);
  let results   = (outs I32:$result);
  let assemblyFormat = "$value attr-dict";
 }
