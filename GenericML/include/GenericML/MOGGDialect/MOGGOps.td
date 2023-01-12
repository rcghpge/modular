//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MOGG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MOGGDIALECT_MOOPS_TD
#define GENERICML_MOGGDIALECT_MOOPS_TD

include "GenericML/MOGGDialect/MOGGDialect.td"
include "GenericML/MODialect/MOTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MOGG_Op<string mnemonic, list<Trait> traits = []>
    : Op<MOGGDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MOGG subgraph ops
//===----------------------------------------------------------------------===//

def MOGG_ExtOp : MOGG_Op<"ext.op", [HasParent<"MO::GraphOp">]> {
  let summary = "MOGG representation of an external library kernel";
  let description = [{
    This is a generic subgraph to represent calls to kernel libraries.

    It contains the name of the kernel, variadic inputs and outputs, and a body
    region for holding the externally called library for lowering.
  }];

  let arguments = (ins
    StrAttr:$opName,
    Variadic<MO_Tensor>:$operands
  );

  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $opName attr-dict `(`($operands^)?`)` $body
      (`:` type($operands)^)? (`->` type($results)^)?
  }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MOGG_OutputOp
    : MOGG_Op<"output", [Terminator]> {
  let summary = "MOGG output operation";
  let description = [{
    The "mogg.output" operation specifies the output values for a region in a
    MOGG operation. It is intendend to allow the regions of mogg operations to
    grow in complexity with specific outputs.

    ```mlir
      mogg.op foo() {
      ... mogg.output e0, % 1 : i32, f8
    } : i32, f8
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // GENERICML_MOGGDIALECT_MOGGOPS_TD
