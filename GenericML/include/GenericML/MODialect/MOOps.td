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

include "GenericML/MODialect/MOInterfaces.td"
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

class MO_ElementWiseOp<string name, list<Trait> traits = []>
    : MO_Op<name, traits # [Pure, SameOperandsAndResultType]> {
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

class MO_ElementWiseUnaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseBinaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
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

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

def MO_AddOp : MO_ElementWiseBinaryOp<"add"> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.
  }];
}

def MO_AbsOp : MO_ElementWiseUnaryOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.
  }];
}

def MO_DivOp : MO_ElementWiseBinaryOp<"div"> {
  let summary = "Divide operation";
  let description = [{
    Returns x / y, where x and y are input tensors.
  }];
}

def MO_GeluOp : MO_ElementWiseUnaryOp<"gelu"> {
  let summary = " Gaussian Error Linear Unit operation";
  let description = [{
    Returns GELU(x) = (x / 2) * (1 + erf(x/sqrt(2))), where erf denotes the
    (Gaussian) error function.
  }];
}

def MO_MulOp : MO_ElementWiseBinaryOp<"mul"> {
  let summary = "Multiply operation";
  let description = [{
    Returns x * y, where x and y are input tensors.
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensors.
  }];
}

def MO_SoftmaxOp : MO_Op<"softmax", [SameOperandsAndResultType, Pure]> {
  let summary = "Perfoms softmax on tensor `input`";
  let description = [{
    Returns softmax probability scores following:
     `exp(input) / sum(exp(input))`

    Axis is a vector representing the axes to reduce.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_SubOp : MO_ElementWiseBinaryOp<"sub"> {
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
    Base class for all generic dot products (i.e. tensor contractions with two
    inputs), for example matmuls.
  }];

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MatmulOp : MO_DotBase<"matmul", [MO_MatmulLike]> {
  let summary = "Matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two 2D tensors.
  }];
}

#endif // GENERICML_MODIALECT_MOOPS_TD
