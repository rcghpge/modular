//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MODIALECT_MOOPS_TD
#define GENERICML_MODIALECT_MOOPS_TD

include "GenericML/MODialect/MOTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
    : Op<MODialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_Op<"graph", [HasParent<"ModuleOp">, FunctionOpInterface,
                                 Symbol, SymbolTable,IsolatedFromAbove
  // TODO : Re-enable TransformableGraphInterface
  // DeclareOpInterfaceMethods<TransformableGraphInterface>
]> {
  let summary = "Function that contains a MO graph with other ops mixed in";
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getFunctionBody;
    using FunctionOpInterfaceTrait::front;
    using FunctionOpInterfaceTrait::end;
    Block *getBodyBlock() { return &getFunctionBody().front(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let builders = [
    /// Create a new mtfl.graph with an empty body.
    OpBuilder<(ins
      "StringRef":$name,
      "TypeRange":$inputTypes,
      "TypeRange":$resultTypes
    )>,
  ];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MO_OutputOp
    : MO_Op<"output", [Terminator, ReturnLike, HasParent<"MO::GraphOp">]> {
  let summary = "MO output operation";
  let description = [{
    The "mo.output" operation specifies the output values for a MO
    function. The operation takes variable number of operands and produces
    no results. The operand number and types must match the signature of
    the mo.graph that contains the operation. For example:

    ```mlir
      mo.graph @foo() : i32, f8 {
      ... mo.output e0, % 1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_Op<"constant"> {
  let summary = "constant MO operation";
  let description = [{
    Op to store constant tensors in model.
  }];

  let arguments = (ins ElementsAttr:$value);
  let results = (outs MO_Tensor:$result);
}


//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def MO_AddOp : MO_Op<"add", [Pure]> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
}

def MO_AbsOp : MO_Op<"abs", [SameOperandsAndResultType, Idempotent, Pure]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
}

def MO_ReluOp : MO_Op<"relu", [SameOperandsAndResultType, Idempotent, Pure]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensors.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
}

#endif // GENERICML_MODIALECT_MOOPS_TD
