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
include "Support/MDialect/MAttrs.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
    : Op<MODialect, mnemonic, traits>;


class MO_ElementWiseBinary<string name, list<Trait> traits = []>
  : MO_Op<name, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
  
  let assemblyFormat = [{
    `(`$input_x `,` $input_y `)` attr-dict `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_Op<"graph", [HasParent<"ModuleOp">, FunctionOpInterface,
                                 Symbol, SymbolTable,IsolatedFromAbove
  // TODO : Re-enable TransformableGraphInterface
  // DeclareOpInterfaceMethods<TransformableGraphInterface>
]> {
  let summary = "Function that contains a MO graph with other ops mixed in";
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
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

def MO_ConstantOp : MO_Op<"constant", [ConstantLike, Pure]> {
  let summary = "constant MO operation";
  let description = [{
    Op to store constant tensors in model.
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr, [AnyAttrOf<[M_ArrayElementsAttr, Builtin_DenseResourceElementsAttr]>]
    >:$value);
  let results = (outs MO_Tensor:$result);

  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def MO_AddOp : MO_ElementWiseBinary<"add"> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.
  }];
}

def MO_AbsOp : MO_Op<"abs", [SameOperandsAndResultType, Idempotent, Pure]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
}

def MO_DivOp : MO_ElementWiseBinary<"div"> {
  let summary = "Divide operation";
  let description = [{
    Returns x / y, where x and y are input tensors.
  }];
}

def MO_MulOp : MO_ElementWiseBinary<"mul"> {
  let summary = "Multiply operation";
  let description = [{
    Returns x * y, where x and y are input tensors.
  }];
}


def MO_ReluOp : MO_Op<"relu", [SameOperandsAndResultType, Idempotent, Pure]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensors.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
}

def MO_SubOp : MO_ElementWiseBinary<"sub"> {
  let summary = "Subtract operation";
  let description = [{
    Returns x - y, where x and y are input tensors.
  }];
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

def MO_BroadcastToOp : MO_Op<"broadcast_to", [Pure]> {
  let summary = "BroadcastTo Operation";
  let description = [{
    Broadcast the input_x tensor to the shape input_shape.

    The Broadcasting follows numpy semantics to determine validity of the
    two shapes.
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_shape);
  let results = (outs MO_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//


class MO_DotBase<string name, list<Trait> traits = []>
  : MO_Op<name, traits # [SameOperandsAndResultElementType]> {
  let arguments = (ins MO_Tensor:$input_a, MO_Tensor:$input_b);
  let results = (outs MO_Tensor:$result);


  let description = [{
    Base class for all generic dot products, including matmuls.

    Currenly we support the following cases:

    Both inputs are 2D matrices: matmul
      (M, K) * (K, N) -> (M, N)

    However over time we will support more of the generic matmul cases as
    outlined in these:
    * https://pytorch.org/docs/stable/generated/torch.matmul.html
    * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

  }];

  
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

// Normal 2D matmul.
def MO_MatmulOp : MO_DotBase<"matmul"> {
  let summary = "Matmul";
  let description = [{
    Perform matrix multiplication on two 2D tensors.
  }];
}


#endif // GENERICML_MODIALECT_MOOPS_TD
