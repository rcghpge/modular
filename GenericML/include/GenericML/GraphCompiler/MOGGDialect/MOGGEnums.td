//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines attributes in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD

include "mlir/IR/EnumAttr.td"

def MOGG_BindLambdaKind : I32EnumAttr<"BindLambdaKind", "lambda kind", [
  I32EnumAttrCase<"InputLambda", 0, "input_lambda">,
  I32EnumAttrCase<"OutputLambda", 1, "output_lambda">,
  I32EnumAttrCase<"ElemwiseLambda", 2, "elemwise_lambda">
]> {
  let cppNamespace = "::M::MOGG";
}


def MOGG_IOKind : I32EnumAttr<"IOKind", "IO type of a mojo kernel operand", [
  I32EnumAttrCase<"Unknown", 32, "Unknown">, // Can't do -1.
  I32EnumAttrCase<"Output", 0, "Output">,
  I32EnumAttrCase<"Input", 1, "Input">,
  I32EnumAttrCase<"FusedInput", 2, "FusedInput">,
  I32EnumAttrCase<"FusedOutput", 3, "FusedOutput">,
  I32EnumAttrCase<"FusedComputeOutput", 31, "FusedComputeOutput">
]> {
  let cppNamespace = "::M::MOGG";
}


#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
