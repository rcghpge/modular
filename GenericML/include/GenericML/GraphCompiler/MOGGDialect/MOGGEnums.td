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

def MOGG_ABIKind : I32EnumAttr<"ABIKind", "ABI kind", [
  I32EnumAttrCase<"MOGG", 0, "MOGG">,
  I32EnumAttrCase<"MGP", 1, "MGP">,
]> {
  let cppNamespace = "::M::MOGG";
}


#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGENUMS_TD
