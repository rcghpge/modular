//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD

include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/KGENDialect/KGENInterfaces.td"
include "KGEN/KGENDialect/KGENTypes.td"
include "GenericML/GraphCompiler/MODialect/MOInterfaces.td"
include "GenericML/GraphCompiler/MODialect/MOTypes.td"
include "GenericML/GraphCompiler/MOSHDialect/MOSHTypes.td"
include "Support/MDialect/MAttrs.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for ops in this dialect.
class MO_Op<string mnemonic, list<Trait> traits = []>
  : Op<MODialect, mnemonic, traits>;

// Base class for ops that work with tensors in the MO dialect.
class MO_TensorOp<string mnemonic, list<Trait> traits = []>
  : MO_Op<mnemonic, [MO_TensorOpInterface] # traits>;

// Base class for ops working with control flow
class MO_ControlOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<mnemonic, [MO_ControlOpInterface] # traits>;

//===----------------------------------------------------------------------===//
// IndexToTensorOp
//===----------------------------------------------------------------------===//

def MO_IndexToTensorOp : MO_Op<"index.to_tensor", [Pure]> {
  let summary = "Cast from `index` to scalar-like `!mo.tensor`.";
  let description = [{
    Return a scalar tensor that contains the input index value.

    Example:

    ```mlir
    %c: index
    %scalarT = mo.index.to_tensor(%c) -> !mo.tensor<[], si64>
    ```
  }];

  let arguments = (ins Index:$input);
  let results = (outs MO_RankedScalarIndexTensor:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

def MO_TransferOp : MO_Op<"transfer", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "Make an operand tensor available on a specified (virtual) "
                "device.";
  let description = [{
    Represents a possible copy or aliasing operation to make the contents of the
    operand tensor available on the (virtual) device given by the operand
    `#M.device_ref` attribute. The result tensor type is identical to the
    operand tensor type except for the device annotation.

    The operand tensor type must have a device annotation. It is valid for the
    source and destination devices to be identical. It is implementation defined
    as to whether the result tensor is a copy or alias of the operand tensor.

    Example:
    ```mlir
    %arg : !mo.tensor<[N, 8], f32> { device = #M.device_ref<"cuda", 3> }
    %res = mo.transfer %arg : !mo.tensor<!mo.tensor<[N, 8], f32>
             { device = #M.device_ref<"cuda", 3> } to <"cuda", 1>
    %res : !mo.tensor<[N, 8], f32> { device = #M.device_ref<"cuda", 1> }
    ```
  }];

  let arguments = (ins MO_Tensor:$input, M_DeviceRefAttr:$destDevice);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    $input `:` qualified(type($input)) `to` $destDevice attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def MO_GraphOp : MO_TensorOp<"graph", [
  HasParent<"ModuleOp">, FunctionOpInterface, Symbol, SymbolTable,
  IsolatedFromAbove, SingleBlockImplicitTerminator<"OutputOp">,
  DeclareOpInterfaceMethods<MO_Parameterization>,
  RegionKindInterface, HasOnlyGraphRegion,
  // needed to leverage KGEN parameter infra
  DeclInterface, FuncInterface
]> {
  let summary = "Function that contains a MO graph with other ops mixed in";
  let description = [{
    This op represents a computation graph that consists of
    - input/output shape parameters
    - input tensors
    - computations on input tensors and intermediate results
    - a terminating output op that returns the result tensors

    The `counter` property will be used to generate new parameter names that are
    unique among all parameter names inside this graph.

    Example:

    ```mlir
    mo.graph @example<D1 -> D2>(%arg: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
      // ... intermediate computations ...
      %res : !mo.tensor<[D3], f32>
      mo.output<D3> %arg : !mo.tensor<[D3], f32>
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<KGEN_SignatureType>:$signature,
    TypeAttrOf<FunctionType>:$functionType,
    KGEN_ParamDeclArrayAttr:$inputParams,
    KGEN_ParamDeclArrayAttr:$resultParams,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    IntProperty<"int64_t">:$counter
  );
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::front;
    using FunctionOpInterfaceTrait::end;
    Block *getBodyBlock() { return &getBody().front(); }
    OutputOp getOutputOp();

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the callable region of this function.
    Region *getCallableRegion() { return &getBodyRegion(); }

    /// We aren't using always_inline functionality in MO.
    KGEN::InlineLevel getInlineLevel() {
      return KGEN::InlineLevel::Never;
    }

    /// Return a new dimension parameter of `IndexType`, with a name that starts
    /// with `prefix` and differs from existing parameter names inside this
    /// graph op.
    KGEN::ParamDeclAttr getNewDimParam(StringRef prefix);

    /// Return a new dimension parameter of `MOSH::ShapeType`, with a name that
    /// starts with `prefix` and differs from existing parameter names inside
    /// this graph op.
    KGEN::ParamDeclAttr getNewShapeParam(StringRef prefix);

    /// Return a new dimension parameter of given type, with a name that starts
    /// with `prefix` and differs from existing parameter names inside this
    /// graph op.
    KGEN::ParamDeclAttr getNewParamOfType(StringRef prefix, Type type);

    /// Return a new name that's unique from existing parameters inside this
    /// graph op.
    std::string getNewParamName(StringRef prefix);

    /// Returns the #M.device_spec_collection annotation for this graph op, or
    /// null if no such annotation.
    DeviceSpecCollectionAttr getDeviceSpecs();

    /// Sets the #M.device_spec_collection annotation for this graph op,
    /// overwriting any existing annotation.
    void setDeviceSpecs(DeviceSpecCollectionAttr deviceSpecs);
  }];

  let builders = [
    /// Create a new mo.graph with an empty body.
    OpBuilder<(ins
      "StringRef":$name,
      "TypeRange":$inputTypes,
      "TypeRange":$resultTypes
    )>,
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def MO_OutputOp : MO_ControlOp<"output", [
  Terminator, ReturnLike, HasParent<"MO::GraphOp">,
  DeclareOpInterfaceMethods<ParamOpInterface>
]>, NoParamDecls {
  let summary = "MO output operation";
  let description = [{
    The "mo.output" operation specifies the output parameters and values for a
    MO function. The operation takes variable number of operands and produces
    no results. The operand number and types must match the signature of
    the mo.graph that contains the operation (after substituting the bindings to
    signature output type). For example:

    Example:

    ```mlir
    mo.graph @no_params(%arg0: !mo.tensor<?, f32>) -> (!mo.tensor<?, f32>) {
      mo.output %arg0 : !mo.tensor<?, f32>
    }

    mo.graph @with_params<D1 -> D2>(
        %arg0: !mo.tensor<[D1], f32>) -> (!mo.tensor<[D2], f32>) {
      mo.output<D1> %arg0 : !mo.tensor<[D1], f32>
    }
    ```

    Note that in the parameterized example, the output operands type annotation
    don't match the graph's output type textually, but they match after
    substituting the bindings into graph's output type.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands,
    DefaultValuedAttr<
      KGEN_ParameterExprArrayAttr, "ArrayRef<TypedAttr>({})">:$parameters
  );

  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : MO_ControlOp<"if",
          [SingleBlockImplicitTerminator<"MO::YieldOp">]> {
  let summary = "if operation";
  let description = [{
    The "mo.if" operation takes an i1 condition, a 'then' block and an 'else'
    block. If the condition is true, the 'then' block is run and the "mo.if"
    operation returns the results of that block, otherwise the "else" block is
    run and its values are returned.

    The blocks have access to all outer SSA values. The blocks must return
    values using the "mo.yield' op, and the returned values must match the
    types given in the "mo.if" result signature.

    Example:

    ```mlir
      %res = mo.if (%cond : !mo.tensor<[], bool>) -> !mo.tensor<?, f32> {
        %v1 = mo.add(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                               ) -> !mo.tensor<?, f32>
        mo.yield %v1 : !mo.tensor<?, f32>
      } else {
        %v2 = mo.sub(%x, %y) : (!mo.tensor<?, f32>, !mo.tensor<?, f32>
                                 ) -> !mo.tensor<?, f32>
        mo.yield %v2 : !mo.tensor<?, f32>
      }
    ```
  }];

  let arguments = (ins MO_CondLike:$cond);
  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);

  let assemblyFormat = [{
    `(` $cond `:` qualified(type($cond)) `)` (`->` type($results)^)?
    $thenRegion `else` $elseRegion attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : MO_ControlOp<"while",
          [SingleBlockImplicitTerminator<"MO::YieldOp">,
          IsolatedFromAbove]> {
  let summary = "while control flow operation";
  let description = [{
    The "mo.while" operation takes 'cond' and 'body' blocks. While the 'cond'
    block evaluates to the true condition, the 'body' block is executed.

    Both the 'cond' and 'body' blocks have access to the outer SSA values listed
    in the "mo.while" signature. The 'cond' block must return a single
    `!mo.tensor<[], bool>` 'scalar' condition result, using a "mo.yield"
    operation. The 'body' block must return results with the same types as
    the "mo.while" signature, again using a "mo.yield". It is not possible
    for the 'cond' or 'body' blocks to access outer SSA values not listed in
    the signature, even if those values never need to be 'updated' by the 'body'
    block.

    The results of the "mo.while" op are the results of the last executed
    'body' block, or the "mo.while" signature arguments if 'cond' never
    evaluated to true.

    Example:

    ```mlir
      %x: !mo.tensor<[2], f32>
      %y: !mo.tensor<[], f32>
      %res:2 = mo.while (%x: !mo.tensor<[2], f32>, %y: !mo.tensor<[], f32>) {
        %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
        %shape = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
        %mean0 = mo.mean (%x, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
        %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
        %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
        mo.yield %cond : !mo.tensor<[], bool>
      } do {
        %new_x = mo.add(%x, %x) : !mo.tensor<[2], f32>
        mo.yield %new_x, %y : !mo.tensor<[2], f32>, !mo.tensor<f32>
      }
      %res#1: !mo.tensor<[2], f32>
    ```
  }];

  let arguments = (ins Variadic<MO_Tensor>:$inputs);
  let results = (outs Variadic<MO_Tensor>:$results);
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion);

  // The assemblyFormat is almost:
  //   custom<ParenOperandList>($inputs) attr-dict $condRegion `do` $bodyRegion
  // However since inputs are used as both referential and binding occurrences
  // we must instead use a custom parser.
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def MO_YieldOp
    : MO_ControlOp<"yield", [
        Terminator, ReturnLike, Pure, ParentOneOf<["IfOp", "WhileOp"]>
      ]> {
  let summary = "MO yield operation";
  let description = [{
    The "mo.yield" operation specifies the output values for MO control flow
    blocks. The operation takes variable number of operands and produces no
    results.

    Example:

    ```mlir
    mo.if $cond : !mo.tensor<[], bool> (!mo.tensor<?, f32>) {
      mo.yield %arg0 : !mo.tensor<?, f32>
    } else {
      mo.yield %arg1 : !mo.tensor<?, f32>
    }
    ```
  }];

  let arguments = (ins Variadic<MO_Tensor>:$operands);

  let assemblyFormat = [{
    `` attr-dict ($operands^ `:` type($operands))?
  }];

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def MO_RebindOp : MO_ControlOp<"rebind", [
    DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let summary = "tensor parameter rebind op";
  let description = [{
    Op to represent "rebinding" a value of `!mo.tensor<...>` type to another
    `!mo.tensor<...>`. Currently, this op can be ued to convert all attributes
    of `!mo.tensor` types, inlcuding shape, element type, and metadata.

    Examples:

    ```mlir
    %0 = mo.rebind(%arg0) : !mo.tensor<[?, ?], f32> -> !mo.tensor<[N, 1], f32>
    %1 = mo.rebind(%0) : !mo.tensor<[N, 1], f32> -> !mo.tensor<Sh, f32>
    %2 = mo.rebind(%1) : !mo.tensor<Sh, f32> -> !mo.tensor<[], f32>
    %3 = mo.rebind<() -> K, M>(%0) :
      !mo.tensor<[N, 1], f32> -> !mo.tensor<[K, M], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def MO_ConstantOp : MO_TensorOp<"constant", [ConstantLike, Pure]> {
  let summary = "constant MO operation";
  let description = [{
    Op to store constant tensors in model.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<1, 2, 3, 4> : tensor<2x2xsi32>
    } : !mo.tensor<[2, 2], si32>
    ```
  }];

  let arguments = (ins
    ConfinedAttr<
      ElementsAttr,
      [AnyAttrOf<[M_ArrayElementsAttr,
                  Builtin_DenseResourceElementsAttr,
                  M_AlignedBytesAttr]>]
    >:$value);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = "attr-dict `:` qualified(type($result))";

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;

  let builders = [
    /// Create a new rank-1 tensor mo.constant of int32 value.
    OpBuilder<(ins "ArrayRef<int32_t>":$values)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int32 value.
    OpBuilder<(ins "int32_t":$value)>,
    /// Create a new rank-1 tensor mo.constant of int64 value.
    OpBuilder<(ins "ArrayRef<int64_t>":$values)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of int64 value.
    OpBuilder<(ins "int64_t":$value)>,
    /// Create a new rank-1 tensor mo.constant of any float type.
    OpBuilder<(ins "ArrayRef<APFloat>":$values, "Type":$elementType)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of given float type.
    OpBuilder<(ins "APFloat":$value, "Type":$elementType )>,
    /// Create a new rank-0 (scalar) tensor mo.constant of f32 type.
    OpBuilder<(ins "float":$value)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of f64 type.
    OpBuilder<(ins "double":$value)>,
    /// Create a new rank-1 tensor mo.constant of boolean value.
    OpBuilder<(ins "ArrayRef<bool>":$values)>,
    /// Create a new rank-0 (scalar) tensor mo.constant of boolean value.
    OpBuilder<(ins "bool":$value)>,
  ];

}

//===----------------------------------------------------------------------===//
// Elementwise ops
//===----------------------------------------------------------------------===//

class MO_ElementWiseOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure, MO_SameOperandsAndResultType]> {
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

class MO_ElementWiseUnaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseUnary]> {
  let arguments = (ins MO_Tensor:$input);
}

class MO_ElementWiseFloatUnaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseUnaryOp<name, traits> {
  let arguments = (ins MO_FloatTensor:$input);
}

class MO_ElementWiseBinaryOp<string name, list<Trait> traits = []>
    : MO_ElementWiseOp<name, traits # [MO_ElementWiseBinary]> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_ComparisonOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [Pure, MO_SameOperandsElementType,
                            MO_SameOperandsAndResultShape]> {
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class MO_ComparisonUnaryOp<string name, list<Trait> traits = []>
    : MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x);
}

class MO_ComparisonBinaryOp<string name, list<Trait> traits = []>
    : MO_ComparisonOp<name, traits> {
  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
}

class MO_LogicalBinaryOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [
  Commutative, Pure, MO_SameOperandsAndResultType, MO_ElementWiseBinary
]> {
  let arguments = (ins MO_BooleanMask:$input_x, MO_BooleanMask:$input_y);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_AddOp : MO_ElementWiseBinaryOp<"add", [Commutative]> {
  let summary = "Add operation";
  let description = [{
    Returns x + y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasFolder = 1;
}

def MO_AbsOp : MO_ElementWiseUnaryOp<"abs", [Idempotent]> {
  let summary = "Absolute value operation";
  let description = [{
    Returns abs(x), where x is the input tensors.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_DivOp : MO_ElementWiseBinaryOp<"div"> {
  let summary = "Divide operation";
  let description = [{
    Returns x / y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.div(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasFolder = 1;
}

def MO_ExpOp : MO_ElementWiseUnaryOp<"exp"> {
  let summary = "Exponential function";
  let description = [{
    Returns exp(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ErfOp : MO_ElementWiseUnaryOp<"erf"> {
  let summary = "Absolute value operation";
  let description = [{
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_GeluOp : MO_ElementWiseUnaryOp<"gelu"> {
  let summary = "Gaussian Error Linear Unit operation";
  let description = [{
    Returns GELU(x) = (x / 2) * (1 + erf(x/sqrt(2))), where erf denotes the
    (Gaussian) error function.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_LogOp : MO_ElementWiseUnaryOp<"log"> {
  let summary = "Natural Logarithm";
  let description = [{
    Computes the natural logorithm `log(x)`.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_Log1pOp : MO_ElementWiseUnaryOp<"log1p"> {
  let summary = "Natural Logarithm of 1 + x";
  let description = [{
    Calculate log(1 + x) maintaining accuracy for small
    x where 1 + x = 1 in floating point rounding.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SigmoidOp : MO_ElementWiseUnaryOp<"sigmoid"> {
  let summary = "Logistic Sigmoid operation";
  let description = [{
    Returns $sigmoid(x) = \frac{1}{1+e^{-x}} = \fract{e^x}{e^x + 1} to compute
    the logistic sigmoid operations.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MaxOp : MO_ElementWiseBinaryOp<"max", [Commutative]> {
  let summary = "Maximum operation";
  let description = [{
    Returns max(x, y), where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.max(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_MinOp : MO_ElementWiseBinaryOp<"min", [Commutative]> {
  let summary = "Minimum operation";
  let description = [{
    Returns min(x, y), where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.min(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ModOp : MO_ElementWiseBinaryOp<"mod"> {
  let summary = "Mod operation";
  let description = [{
    Returns x mod y, where x and y are input tensors. Supports floating point
    and integral types.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], si32>
    %rhs: !mo.tensor<[2, 3], si32>
    %res = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], si32>
    ```
  }];
}

def MO_MulOp : MO_ElementWiseBinaryOp<"mul", [Commutative]> {
  let summary = "Multiply operation";
  let description = [{
    Returns x * y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.mul(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_NegativeOp : MO_ElementWiseUnaryOp<"negative"> {
  let summary = "Returns the negative of the input.";
  let description = [{
    Negates the argument: -x

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.negative(%arg) : !mo.tensor<[2, 3], f32>
  }];

  let arguments = (ins MO_SignedTensor:$input);
}

def MO_PowOp : MO_TensorOp<"pow", [Pure, MO_SameOperandsAndResultShape]> {
  let summary = "Power operation";
  let description = [{
    Computes elementwise x ** y, where x is the base tensor and y is the
    exponent tensor.

    Example:

    ```mlir
    %x: !mo.tensor<[2, 3], f32>
    %y: !mo.tensor<[2, 3], f32>
    %res = mo.pow(
        %x: !mo.tensor<[2, 3], f32>,
        %y: !mo.tensor<[2, 3], f32>
    ) : !mo.tensor<[2, 3], f32>

    %x: !mo.tensor<[2, 3], f32>
    %y_int: !mo.tensor<[2, 3], si32>
    %res = mo.pow(
        %x: !mo.tensor<[2, 3], f32>,
        %y: !mo.tensor<[2, 3], si32>
    ) : !mo.tensor<[2, 3], f32>
    %res = mo.pow(%x, %y_int) : !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_x, MO_Tensor:$input_y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` $input_x `:` qualified(type($input_x))
    `,` $input_y `:` qualified(type($input_y)) `)` attr-dict
    `:` qualified(type($result))
  }];
}

def MO_CeilOp : MO_ElementWiseFloatUnaryOp<"ceil", [Idempotent]> {
  let summary = "Ceiling operation";
  let description = [{
    Returns the smallest largest integer greater than x.

    E.g.
       -3.5 --> -3
        3.5 -->  4

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_FloorOp : MO_ElementWiseFloatUnaryOp<"floor", [Idempotent]> {
  let summary = "Floor operation";
  let description = [{
    Returns the elementwise largest integer not greater than x.

    E.g.
       -3.5 --> -4
        3.5 -->  3

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TruncOp : MO_ElementWiseFloatUnaryOp<"trunc", [Idempotent]> {
  let summary = "Truncation operation";
  let description = [{
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    E.g.
       -3.5 --> -3
        3.5 -->  3

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RoundOp : MO_ElementWiseFloatUnaryOp<"round", [Idempotent]> {
  let summary = "Round operation";
  let description = [{
    Returns the elementwise nearest integer, with ties going away from zero.

    E.g.
       -3.5 --> -4
       -2.5 --> -3
        2.5 -->  3
        3.5 -->  4

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RoundEvenOp : MO_ElementWiseFloatUnaryOp<"roundeven", [Idempotent]> {
  let summary = "Round even operation";
  let description = [{
    Returns the elementwise neareset integer, with ties going toward the
    nearest even number.

    E.g.
       -3.5 --> -4
       -2.5 --> -2
        2.5 -->  2
        3.5 -->  4

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.roundeven(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_ReluOp : MO_ElementWiseUnaryOp<"relu", [Idempotent]> {
  let summary = "Rectified linear unit operation";
  let description = [{
    Returns max(0, x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_RsqrtOp : MO_ElementWiseUnaryOp<"rsqrt"> {
  let summary = "Inverse Square Root operation";
  let description = [{
    Returns 1/sqrt(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SqrtOp : MO_ElementWiseUnaryOp<"sqrt"> {
  let summary = "Square Root operation";
  let description = [{
    Return sqrt(x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_CastOp : MO_TensorOp<"cast", [Pure, MO_SameOperandsAndResultShape]> {
  let summary = "Cast operation";
  let description = [{
    Return cast[dtype](x), where x is the input tensor.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], i32>
    %res = mo.cast(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;

  let builders = [
    /// Cast DType interface
    OpBuilder<(ins "TensorValue":$input, "DType":$dtype)>,
  ];
}

def MO_IsNaNOp : MO_ComparisonUnaryOp<"is_nan"> {
  let summary = "IsNaN operation";
  let description = [{
    Returns true if x is not a number.

    Example:

    ```mlir
    %x: !mo.tensor<[2, 3], f32>
    %res = mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                           ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_IsInfOp : MO_ComparisonUnaryOp<"is_inf"> {
  let summary = "IsInf operation";
  let description = [{
    Returns true if x is infinite.

    Example:

    ```mlir
    %x: !mo.tensor<[2, 3], f32>
    %res = mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                           ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_EqualOp : MO_ComparisonBinaryOp<"equal", [Commutative]> {
  let summary = "Equal operation";
  let description = [{
    Returns elementwise x == y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                   !mo.tensor<[2, 3], f32>
                                  ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let hasFolder = 1;
}

def MO_GreaterOp : MO_ComparisonBinaryOp<"greater"> {
  let summary = "Greater operation";
  let description = [{
    Returns elementwise x > y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                     !mo.tensor<[2, 3], f32>
                                    ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let hasFolder = 1;
}

def MO_GreaterEqualOp : MO_ComparisonBinaryOp<"greater_equal"> {
  let summary = "Greater Equal operation";
  let description = [{
    Returns elementwise x >= y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                           !mo.tensor<[2, 3], f32>
                                          ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let hasFolder = 1;
}

def MO_NotEqualOp : MO_ComparisonBinaryOp<"not_equal", [Commutative]> {
  let summary = "Not Equal operation";
  let description = [{
    Returns elementwise x != y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                       !mo.tensor<[2, 3], f32>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let hasFolder = 1;
}

def MO_AndOp : MO_LogicalBinaryOp<"and"> {
  let summary = "Logical And operation";
  let description = [{
    Returns elementwise x `and` y, where x and y are input boolean tensors, and
    the resulting element is true if both the lhs and rhs elements are true.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], bool>
    %rhs: !mo.tensor<[2, 3], bool>
    %res = mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                 !mo.tensor<[2, 3], bool>
                                ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

def MO_NotOp : MO_TensorOp<"not", [
  Pure, MO_SameOperandsAndResultType, MO_ElementWiseUnary
]> {
  let summary = "Logical Not operation";
  let description = [{
    Returns elementwise `not` on given input, where input is a boolean tensor,
    and the resulting element is true if the input element is false.

    Example:

    ```mlir
    %in: !mo.tensor<[2, 3], bool>
    %res = mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$input);
  let results = (outs MO_BooleanMask:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

def MO_XorOp : MO_LogicalBinaryOp<"xor"> {
  let summary = "Logical Xor operation";
  let description = [{
    Returns elementwise x `xor` y, where x and y are input boolean tensors, and
    the resulting element is true if only one of the lhs and rhs elements is
    true.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], bool>
    %rhs: !mo.tensor<[2, 3], bool>
    %res = mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                 !mo.tensor<[2, 3], bool>
                                ) -> !mo.tensor<[2, 3], bool>
    ```
  }];
}

// TODO: Use an AllElementTypesMatch<["x", "y", "result"]> trait for MO_Tensor.
def MO_SelectOp : MO_TensorOp<"select", [Pure, MO_SameOperandsAndResultShape]>{
  let summary = "Select operation";
  let description = [{
    Return cond ? x : y (element-wise), where cond, x and y are input tensors.

    Example:

    ```mlir
    %cond: !mo.tensor<[2, 3], bool>
    %x: !mo.tensor<[2, 3], f32>
    %y: !mo.tensor<[2, 3], f32>
    %res = mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
  }];
  let arguments = (ins MO_BooleanMask:$cond, MO_Tensor:$x, MO_Tensor:$y);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def MO_SliceOp : MO_TensorOp<"slice", [
  MO_DefaultParameterization,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Slice operation";
  let description = [{
    Return a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements.

    Example:
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
  }];
  let arguments = (ins MO_Tensor:$input,
                       MO_IndicesVectorIfRanked:$start,
                       MO_IndicesVectorIfRanked:$stop,
                       MO_IndicesVectorIfRanked:$step,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_SubOp : MO_ElementWiseBinaryOp<"sub"> {
  let summary = "Subtract operation";
  let description = [{
    Returns x - y, where x and y are input tensors.

    Example:

    ```mlir
    %lhs: !mo.tensor<[2, 3], f32>
    %rhs: !mo.tensor<[2, 3], f32>
    %res = mo.sub(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_TanhOp : MO_ElementWiseUnaryOp<"tanh"> {
  let summary = "Tanh function";
  let description = [{
    Computes the hyperbolic tangent of the input tensor elements.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Softmax-like ops
//===----------------------------------------------------------------------===//

class MO_SoftmaxLikeOp<string name, list<Trait> traits = []>
    : MO_TensorOp<name, traits # [MO_SameOperandsAndResultType, Pure]> {
  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_LogSoftmaxOp : MO_SoftmaxLikeOp<"logsoftmax"> {
  let summary = "Perfoms logsoftmax on tensor `input`";
  let description = [{
    Returns logsoftmax probability scores following:
     `log(softmax(input))`

    The scores are computed along the last axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

def MO_SoftmaxOp : MO_SoftmaxLikeOp<"softmax"> {
  let summary = "Perfoms softmax on tensor `input`";
  let description = [{
    Returns softmax probability scores following:
     `exp(input) / sum(exp(input))`

    The scores are computed along the last axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = mo.softmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Reduction ops
//===----------------------------------------------------------------------===//

class MO_ReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
    name, traits # [MO_ReductionTraits]> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;

  let builders = [
    /// Build reduction op and automatically infer output type if it is not given.
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$axis, CArg<"TensorType", "{}">:$outputTy)>,
    OpBuilder<(ins "TensorValue":$input, "int64_t":$axis, CArg<"TensorType", "{}">:$outputTy)>
  ];
}

def MO_MeanOp : MO_ReductionOp<"mean"> {
  let summary = "Mean reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their mean value, and change that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.mean(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMulOp : MO_ReductionOp<"reduce.mul"> {
  let summary = "Product reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their product, and change that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.mul(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
  let hasFolder = 1;
}

def MO_ReduceAddOp : MO_ReductionOp<"reduce.add"> {
  let summary = "Sum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their sum, and change that axis's
    dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.add(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
  let hasFolder = 1;
}

def MO_ReduceMinOp : MO_ReductionOp<"reduce.min"> {
  let summary = "Minimum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their minimum, and change that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.min(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

def MO_ReduceMaxOp : MO_ReductionOp<"reduce.max"> {
  let summary = "Maximum reduction operation";
  let description = [{
    Reduce `input` elements across `axis` to their maximum, and change that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.reduce.max(%arg, %axis) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Sliding window ops
//===----------------------------------------------------------------------===//

def MO_AvgPoolOp : MO_TensorOp<"avg_pool", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]>{
  let summary = "Average pooling operation";
  let description = [{
    Apply average pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
    %fs = mo.constant {
      value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %st = mo.constant {
      value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %di = mo.constant {
      value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %pa = mo.constant {
      value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %res = mo.avg_pool(%arg) [
        filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
    ] : (
      !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
    ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_MaxPoolOp : MO_TensorOp<"max_pool", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]>{
  let summary = "Max pooling operation";
  let description = [{
    Apply max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
    %fs = mo.constant {
      value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %st = mo.constant {
      value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %di = mo.constant {
      value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %pa = mo.constant {
      value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %res = mo.max_pool(%arg) [
        filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
    ] : (
      !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
    ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLikeIfRanked:$filter_shape,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `)`
    `[` `filter_shape` `=` $filter_shape `,` `strides` `=` $strides `,`
        `dilations` `=` $dilations `,` `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_ConvOp : MO_TensorOp<"conv", [
  MO_SlidingWindow,
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]>{
  let summary = "Convolution operation";
  let description = [{
    Compute the convolution of the input with the given filter, strides, and
    dilations.

    For now the op only supports 2d convolution (so input and filter must be
    4D), with the following layout assumptions:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e., (height, width, in_channels, out_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    The input, output and filter tensors' ranks must match if statically known.

    This op currently only supports strides and dilations on the input.

    Example:

    ```mlir
    %st = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
      : !mo.tensor<[2], si64>
    %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>}
      : !mo.tensor<[2], si64>
    %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
      : !mo.tensor<[4], si64>
    %res = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa] : (
      !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
      !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
    ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$filter,
    MO_ShapeLikeIfRanked:$strides,
    MO_ShapeLikeIfRanked:$dilations,
    MO_ShapeLikeIfRanked:$paddings,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` $input `,` $filter `)`
    `[` `strides` `=` $strides `,` `dilations` `=` $dilations `,`
                    `paddings` `=` $paddings `]` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape related ops
//===----------------------------------------------------------------------===//

class MO_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
  : MO_TensorOp<mnemonic,
    traits # [DeclareOpInterfaceMethods<MO_DefaultParameterization>]
> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BroadcastToOp : MO_ShapeModifierOp<"broadcast_to"> {
  let summary = "BroadcastTo Operation";
  let description = [{
    Broadcast the input tensor to the shape newShape.

    The broadcasting follows numpy semantics to determine validity of the
    two shapes.

    Example:

    ```mlir
    %from: !mo.tensor<[3], f32>
    %to = mo.constant {
      value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %res = mo.broadcast_to(%from, %to) : (
      !mo.tensor<[3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
    ```
  }];
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let builders = [
    /// Create a ShapeOf + BroadcastToOp with a "BroadcastLike" interface.
    /// The second argument is not the shape to broadcast to, but a tensor
    /// whose shape we want to broadcast to.
    /// The third optional parameter should represent the shape of the
    /// second arg (e.g. from ShapeOfOp). If not given a ShapeOfOp will be
    /// inserted. It is provided to avoid Op duplication before CSE.
    OpBuilder<(ins "TensorValue":$input, "TensorValue":$broadcastLike, CArg<"TensorValue", "{}">:$shapeOfInput)>,
  ];
}

def MO_ReshapeOp : MO_ShapeModifierOp<"reshape", [DeclareOpInterfaceMethods<MO_Parameterization>]> {
  let summary = "Reshape operation";
  let description = [{
    Return a tensor with the same underlying data, but different shape.

    The first argument is the tensor to reshape.  The second tensor is the
    shape to reshape the first tensor to.  The second tensor may contain a
    single "-1" element, which signifies that that dimension should be
    automatically computed.

    Example:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %shape = mo.constant {
      value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```

    Auto-sizing example:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %shape = mo.constant {
      value = #M.dense_array<3, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %arg2 = mo.reshape(%arg1, %shape) : !mo.tensor<[3, 2], si64>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_UnsqueezeShapeOp : MO_ShapeModifierOp<"unsqueeze_shape"> {
  let summary = "Calculates the resulting shape for unsqueeze-type ops.";
  let description = [{
    Calculates the shape from unsqueeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N + M`.

    The indices in the given list map to the new vector of length `N + M` where
    the indicated dimensions are replaced with `1`. The remaining dimension in
    the original input shape vector are copied over in the non-1 dimensions.

    This operator supports negative indexing. Example:

    $input_shape: [10, 11, 12]
    $padding_indices: [-6, 3, 5] # -6 is equivalent to 0
    $result: [1, 10, 11, 1, 12, 1]

    ```mlir
      %input_shape : !mo.tensor<[3], si32>
      %indices : !mo.tensor<[4], si32>
      %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[7], si32>
    ```
  }];

  let arguments = (ins
    MO_ShapeLike:$input_shape,
    MO_ShapeLike:$padding_indices,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;

  let hasCanonicalizeMethod = 1;
}

def MO_ShapeOfOp : MO_TensorOp<"shape_of", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>
]> {
  let summary = "ShapeOf Operation";
  let description = [{
    Get the shape of a tensor.

    Example:

    ```mlir
    // statically ranked
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %shape1 = mo.shape_of(%arg1) : !mo.tensor<[3], si64>

    // dynamically ranked
    %arg2: !mo.tensor<?, f32>
    %shape2 = mo.shape_of(%arg2) : !mo.tensor<[?], si64>
    ```
  }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let builders = [
    /// Create a shape-of of op on the given SSA value.
    OpBuilder<(ins "TensorValue":$input)>,
  ];

  let hasVerifier = 1;
}

def MO_BroadcastShapeOp : MO_TensorOp<"broadcast_shape", [
  Commutative, DeclareOpInterfaceMethods<MO_DefaultParameterization>
]> {
  let summary = "Broadcast Shape Operation";
  let description = [{
    Given two tensor shapes, return the shape that both broadcast to under
    the numpy rules:

    1. Make the shapes the same rank by adding 1 to the front
       of the shorter shape
    2. Ensure each dimension is equal, either by matching them or promoting a 1
       to a larger number from the other shape

    Example 1:

    ```mlir
    %arg1: !mo.tensor<[1, 2, 3], f32>
    %arg2: !mo.tensor<[4, 2, 1], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[4, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, shape1 will return [4, 2, 3]

    Example 2:

    ```mlir
    %arg1: !mo.tensor<[10, 2, 1], f32>
    %arg2: !mo.tensor<[5], f32>
    %inshape1 = mo.shape_of(%arg1) : (
      !mo.tensor<[10, 2, 1], f32>) -> !mo.tensor<[3], si64>
    %inshape2 = mo.shape_of(%arg2) : (
      !mo.tensor<[5], f32>) -> !mo.tensor<[1], si64>
    %shape1 = mo.broadcast_shape(%inshape1, %inshape2) : !mo.tensor<[3], si64>
    ```
    In this example, shape1 will return [10, 2, 5]
  }];

  let arguments = (ins MO_ShapeLike:$input_x,
                       MO_ShapeLike:$input_y,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_ShapeLike:$shape);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

/// Used to model MO ops that accept both MO_Tensor and MOSH_ShapeType.
class MOShape_Op<string mnemonic, list<Trait> traits = []>
    : Op<MODialect, "shape." # mnemonic, traits> {
  string cppNamespace = MODialect.cppNamespace # "::Shape";
}

def MOShape_ReshapeOp : MOShape_Op<"reshape", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Reshape operation";
  let description = [{
    Return a tensor with the same underlying data, but different shape. The
    semantics of this op are nearly identical to those of `mo.reshape` (see the
    documentation of that op for details), with the only difference that
    `mo.shape.reshape` accepts a `!mosh.ape` value as the new shape instead of a
    shape-like tensor.

    Notably, this op differs from `MOStatic_ReshapeOp` in that this op takes the
    new shape as an explicit argument (i.e. SSA value), whereas
    `MOStatic_ReshapeOp` relies on its parametric result type to determine the
    new shape.

    Example:

    ```mlir
    %arg: !mo.tensor<[1, 2, 3], f32>
    %to: !mosh.ape
    %res = mo.shape.reshape(%arg, %to) : !mo.tensor<[3, 2], si64>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeType:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MOShape_BroadcastToOp : MOShape_Op<"broadcast_to", [
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
]> {
  let summary = "BroadcastTo operation";
  let description = [{
    Broadcast the input tensor to the shape newShape. The semantics of this op
    are nearly identical to those of `mo.broadcast_to` (see the documentation of
    that op for details), with the only difference that `mo.shape.broadcast_to`
    accepts a `!mosh.ape` value as the new shape instead of a shape-like tensor.

    Notably, this op differs from `MOStatic_BroadcastToOp` in that this op takes
    the new shape as an explicit argument (i.e. SSA value), whereas
    `MOStatic_BroadcastToOp` relies on its parametric result type to determine
    the new shape.

    Example:

    ```mlir
    %from: !mo.tensor<[3], f32>
    %to: !mosh.ape
    %res = mo.shape.broadcast_to(%from, %to) : (
      !mo.tensor<[3], f32>, !mosh.ape) -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MOSH_ShapeType:$newShape,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MOShape_FromTensorOp : MOShape_Op<"from_tensor", [Pure]> {
  let summary = "Cast from a shape-like `!mo.tensor` to `!mosh.ape`.";
  let description = [{
    Cast the input shape-like tensor to a shape value.

    Example:

    ```mlir
    %sht: !mo.tensor<[2], si64>
    %sh = mo.shape.from_tensor(%sht) : !mo.tensor<[2], si64>
    ```
  }];

  let arguments = (ins MO_ShapeLike:$input);
  let results = (outs MOSH_ShapeType:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `:` qualified(type($input))";
  let hasFolder = 1;
}

def MOShape_ToTensorOp : MOShape_Op<"to_tensor", [Pure]> {
  let summary = "Cast from a `!mosh.ape` to a shape-like `!mo.tensor`.";
  let description = [{
    Cast the input shape value to a shape-like tensor.

    Example:

    ```mlir
    %sh: !mosh.ape
    %sht = mo.shape.to_tensor(%sh) -> !mo.tensor<[2], si64>
    ```
  }];

  let arguments = (ins MOSH_ShapeType:$input);
  let results = (outs MO_ShapeLike:$result);

  let assemblyFormat = "`(` $input `)` attr-dict `->` qualified(type($result))";
  let hasFolder = 1;
}

class MOStatic_ShapeModifierOp<string mnemonic, list<Trait> traits = []>
    : MO_TensorOp<"static." # mnemonic, traits # [Pure]> {
  string cppNamespace = MODialect.cppNamespace # "::Static";

  let arguments = (ins MO_KnownTensor:$input);
  let results = (outs MO_KnownTensor:$result);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`
    qualified(type($input)) `->` qualified(type($result))
  }];
  let hasFolder = 1;
}

def MOStatic_BroadcastToOp : MOStatic_ShapeModifierOp<"broadcast_to"> {
  let summary = "Broadcast operation for parametric shapes";
  let description = [{
    Broadcast the input tensor to the result tensor. The shape of the input and
    result tensors must not be unknown or contain unknown dimensions, but can be
    parametric.

    This op only has limited compile-time check on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.broadcast_to` with a constant shape-like tensor instead).

    The broadcasting follows numpy semantics.

    Example:

    ```mlir
    %from: !mo.tensor<[3], f32>
    %res1 = mo.static.broadcast(%from)
      : !mo.tensor<[3], f32> -> !mo.tensor<[2, 3], f32>
    kgen.param.declare N = <...>
    %res2 = mo.static.broadcast(%from)
      : !mo.tensor<[3], f32> -> !mo.tensor<[N, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MOStatic_ReshapeOp : MOStatic_ShapeModifierOp<"reshape"> {
  let summary = "Reshape operation for parametric shapes";
  let description = [{
    Return a tensor with the same underlying data, but different shape. The
    shape of the input and result tensors must not be unknown or contain unknown
    dimensions, but can be parametric. We do not allow inferred dimensions
    (e.g. -1 in MO_ReshapeOp).

    The op has no compile-time or runtime checks on the validity of the target
    shape (we expect the user to add any necessary runtime checks); therefore it
    is not recommended for frontend conversion code to rely on this op (use
    `mo.reshape` with a constant shape-like tensor instead).

    Example:

    ```mlir
    %from: !mo.tensor<[2, 3], f32>
    %res = mo.static.reshape(%from)
      : !mo.tensor<[2, 3], f32> -> !mo.tensor<[2, 3], f32>
    ```
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MO_ConcatOp : MO_TensorOp<"concat", [
  DeclareOpInterfaceMethods<MO_Parameterization>,
  DeclareOpInterfaceMethods<MO_ParamDeclaration>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Concat Operation";
  let description = [{
    Concatenate the input tensors along a given dimension.

    `mo.concat` concatenates the `inputs` tensors into an output tensor. There
    must be at least 1 input tensor. The input tensors and output tensors all
    has the same shape except along the concatenation dimension `axis`. The size
    of the concatenation dimension in output tensor would have be the sum of
    sizes of the concatenation dimension in input tensors.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %arg0: !mo.tensor<[2, 3], f32>
    %arg1: !mo.tensor<[2, 5], f32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.concat[%axis: !mo.tensor<[], si64>](%arg0, %arg1) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>
    ) -> !mo.tensor<[2, 8], f32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarIndexTensor:$axis,
    Variadic<MO_Tensor>:$inputs,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def MO_SplitOp : MO_TensorOp<"split", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Split Operation";
  let description = [{
    Split the input tensor into multiple tensors along a given dimension.

    `mo.split` splits the tensor `input` into multiple output tensors.
    The number of output tensors is equal to the number of elements in
    `splitSizes`, which is a rank-1 tensor of integers.
    Each of the output tensors has the same shape as `input` except along the
    split dimension `axis`, where the size is given by the corresponding
    element in `splitSizes`.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
    %input: !mo.tensor<[2, 8], f32>
    %splitSizes = mo.constant {
      value = #M.dense_array<3, 5> : tensor<2xsi64>
    } : !mo.tensor<[2], si64>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>
    } : !mo.tensor<[], si64>
    %res = mo.split[%axis: !mo.tensor<[], si64>](%input, %splitSizes) : (
      !mo.tensor<[2, 8], f32>, !mo.tensor<[2], si64>
    ) -> (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 5], f32>)
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_IndicesVectorIfRanked:$splitSizes,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs Variadic<MO_Tensor>:$results);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls)
    `[` $axis `:` qualified(type($axis)) `]`
    `(` $input `,` $splitSizes `)` attr-dict `:` `(`
      qualified(type($input)) `,` qualified(type($splitSizes))
    `)` `->` `(` type($results) `)`
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def MO_TransposeOp : MO_TensorOp<"transpose", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Transpose Operation";
  let description = [{
    Return a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
    %input : !mo.tensor<[2, 3], f32>
    %perm : !mo.tensor<[2], si64>
    %res = mo.transpose(%input, %perm) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

    %input : !mo.tensor<[?, 5, ?], f32>
    %perm : !mo.tensor<[3], si32>
    %res = mo.transpose(%input, %perm) : (
      !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
    ) -> !mo.tensor<[?, ?, 5], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$perm,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MO_TileOp : MO_TensorOp<"tile", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Tile Operation";
  let description = [{
    Return a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
    %input : !mo.tensor<[2, 3], f32>
    %repeats : !mo.tensor<[2], si64>
    %res = mo.tile(%input, %repeats) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_ShapeLike:$repeats,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_PadOp : MO_TensorOp<"pad", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Pad Operation";
  let description = [{
    Pad the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assume input has rank `N`, the `paddings` tensor should have
    shape `(N, 2)`, where each row has the form `[before, after]`, indicating
    how many of `constant` to add before and after the contents of `input` in
    that dimension. The size of each dimension D of the padded output is:
    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
    %input: !mo.tensor<[2, 3], f32>
    %constant = mo.constant {
      value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
    %paddings = mo.constant {
      value = #M.dense_array<1, 0, 1, 1> : tensor<2x2xsi64>
    } : !mo.tensor<[2, 2], si64>
    %output   = mo.pad(%input, %paddings, %constant) : (
      !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 2], si64>, !mo.tensor<[], f32>
    ) -> !mo.tensor<[3, 5], f32>
    ```
  }];

  // TODO(#7987) support a padding_mode argument
  let arguments = (ins
    MO_Tensor:$input,
    MO_PaddingLike:$paddings,
    MO_RankedScalarTensor:$constant,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

def MO_GatherOp : MO_TensorOp<"gather", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Gather Operation";
  let description = [{
    Gather slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis.

    Example:

    ```mlir
    %input : !mo.tensor<[2, 2], f32>
    %indices: !mo.tensor<[2], si64>
    %axis = mo.constant {
      value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %res = mo.gather(%input, %indices, %axis) : (
      !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>, !mo.tensor<[], si64>
    ) -> !mo.tensor<[2, 2], f32>
    ```
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

// TODO(#20442): Remove with generic gather fusion.
def MO_GatherSumOp : MO_TensorOp<"gather_sum"> {
  let summary = "Gather sum op";
  let description = [{
    A temporary op for the gather sum pattern to target.
    Implicitly gather axis 0 and reduce axis 1.
  }];

  let arguments = (ins MO_NonScalarIfRanked:$input,
                       MO_IndicesTensor:$indices);
  let results = (outs MO_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

def MO_ScatterOp : MO_TensorOp<"scatter", [MO_ScatterLikeTraits]> {
  let summary = "Scatter Operation";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 4], f32>
    %updates: !mo.tensor<[2, 3], f32>
    %indices: !mo.tensor<[2, 3], si64>
    %res = mo.scatter(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
    ) -> !mo.tensor<[4, 4], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

class MO_ScatterReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
  "scatter." # name, traits # [MO_ScatterLikeTraits]
> {
  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterAddOp : MO_ScatterReductionOp<"add"> {
  let summary = "Scatter operation with addition as reduction function";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 4], f32>
    %updates: !mo.tensor<[2, 3], f32>
    %indices: !mo.tensor<[2, 3], si64>
    %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
    ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMaxOp : MO_ScatterReductionOp<"max"> {
  let summary = "Scatter operation with maximum as reduction function";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 4], f32>
    %updates: !mo.tensor<[2, 3], f32>
    %indices: !mo.tensor<[2, 3], si64>
    %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
    ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMinOp : MO_ScatterReductionOp<"min"> {
  let summary = "Scatter operation with minimum as reduction function";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 4], f32>
    %updates: !mo.tensor<[2, 3], f32>
    %indices: !mo.tensor<[2, 3], si64>
    %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
    ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

def MO_ScatterMulOp : MO_ScatterReductionOp<"mul"> {
  let summary = "Scatter operation with multiplication as reduction function";
  let description = [{
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 4], f32>
    %updates: !mo.tensor<[2, 3], f32>
    %indices: !mo.tensor<[2, 3], si64>
    %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 4], f32>, !mo.tensor<[1, 3], f32>, !mo.tensor<[1, 3], si64>
    ) -> !mo.tensor<[4, 4], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ScatterNdOp
//===----------------------------------------------------------------------===//

def MO_ScatterNdOp : MO_TensorOp<"scatter_nd", [MO_ScatterNdLikeTraits]> {
  let summary = "ScatterND Operation";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 2], f32>
    %updates: !mo.tensor<[1, 3, 2], f32>
    %indices: !mo.tensor<[1, 3, 1], si64>
    %res = mo.scatter_nd(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
    ) -> !mo.tensor<[4, 2], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

class MO_ScatterNdReductionOp<string name, list<Trait> traits = []> : MO_TensorOp<
  "scatter_nd." # name, traits # [MO_ScatterNdLikeTraits]
> {
  let arguments = (ins MO_Tensor:$input,
                       MO_NonScalarIfRanked:$updates,
                       MO_IndicesTensor:$indices,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_ScatterNdAddOp : MO_ScatterNdReductionOp<"add"> {
  let summary = "ScatterND operation with addition as reduction function";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 2], f32>
    %updates: !mo.tensor<[1, 3, 2], f32>
    %indices: !mo.tensor<[1, 3, 1], si64>
    %res = mo.scatter_nd.add(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
    ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMaxOp : MO_ScatterNdReductionOp<"max"> {
  let summary = "ScatterND operation with maximum as reduction function";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 2], f32>
    %updates: !mo.tensor<[1, 3, 2], f32>
    %indices: !mo.tensor<[1, 3, 1], si64>
    %res = mo.scatter_nd.max(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
    ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMinOp : MO_ScatterNdReductionOp<"min"> {
  let summary = "ScatterND operation with minimum as reduction function";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 2], f32>
    %updates: !mo.tensor<[1, 3, 2], f32>
    %indices: !mo.tensor<[1, 3, 1], si64>
    %res = mo.scatter_nd.min(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
    ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

def MO_ScatterNdMulOp : MO_ScatterNdReductionOp<"mul"> {
  let summary = "ScatterND operation with multiplication as reduction function";
  let description = [{
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
    %input:   !mo.tensor<[4, 2], f32>
    %updates: !mo.tensor<[1, 3, 2], f32>
    %indices: !mo.tensor<[1, 3, 1], si64>
    %res = mo.scatter_nd.mul(%inputs, %updates, %indices) : (
      !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
    ) -> !mo.tensor<[4, 2], f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

def MO_ArgNonzeroOp : MO_TensorOp<"arg_nonzero", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>
]> {
  let summary = "Returns locations of nonzero values in a tensor";
  let description = [{
    Return a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
    } : !mo.tensor<[3, 3], si32>
    %1 = mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si32>
    ```
    }];

  let arguments = (ins MO_Tensor:$input, MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_I64Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArgMaxOp / ArgMinOp
//===----------------------------------------------------------------------===//

// Base class for ArgMin and ArgMax.
class MO_ArgMinMaxOp<string mnemonic> : MO_ReductionOp<mnemonic> {
  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$axis,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_I64Tensor:$result);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def MO_ArgMaxOp : MO_ArgMinMaxOp<"arg_max"> {
  let summary = "Returns the indices of the maximum value in a tensor";
  let description = [{
    This op is equivalent to reduce_max, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical maximum values, the lowest index is returned.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
    } : !mo.tensor<[2, 2], si32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<si32>
    } : !mo.tensor<[], si32>
    %1 = mo.arg_max(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

def MO_ArgMinOp : MO_ArgMinMaxOp<"arg_min"> {
  let summary = "Returns the indices of the minimum value in a tensor";
  let description = [{
    This op is equivalent to reduce_min, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical minimum values, the lowest index is returned.

    Example:

    ```mlir
    %0 = mo.constant {
      value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
    } : !mo.tensor<[2, 2], si32>
    %axis = mo.constant {
      value = #M.dense_array<1> : tensor<si32>
    } : !mo.tensor<[], si32>
    %1 = mo.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    }];
}

//===----------------------------------------------------------------------===//
// Range
//===----------------------------------------------------------------------===//

def MO_RangeOp : MO_TensorOp<"range", [
  MO_SameOperandsAndResultElementType,
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>
]> {
  let summary = "Create a sequence of numbers";
  let description = [{
    The sequence goes from `start` with increments of size `step` up to (but not
    including) `limit`. All arguments are mandatory and must have the same
    element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
    %limit : !mo.tensor<[], f32>
    %start = mo.constant {
      value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
    %step = mo.constant {
      value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
    %res = mo.range(%start, %limit, %step) : (
      !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
    ) -> !mo.tensor<[?], f32>

    %startInt = mo.constant {
      value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
    %stepInt = mo.constant {
      value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
    %limitInt = mo.constant {
      value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
    %oddNumbersBelowTen = mo.range(%startInt, %limitInt, %stepInt) : (
      !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
    ) -> !mo.tensor<[5], si32>
    ```
  }];

  let arguments = (ins
    MO_RankedScalarTensor:$start,
    MO_RankedScalarTensor:$limit,
    MO_RankedScalarTensor:$step,
    MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_VectorIfRanked:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// Matmul related ops
//===----------------------------------------------------------------------===//

class MO_DotBase<string name, list<Trait> traits = []> :
    MO_TensorOp<name, traits # [
    MO_MatmulLike,
    DeclareOpInterfaceMethods<MO_Parameterization>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
]> {
  let description = [{
    Base class for all generic dot products (i.e. tensor contractions with two
    inputs), for example matmuls.
  }];

  let results = (outs MO_Tensor:$result);

  let hasVerifier = 1;
}

def MO_MatmulOp : MO_DotBase<"matmul"> {
  let summary = "Matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two 2D tensors.

    Example:

    ```mlir
    %lhs: ... !mo.tensor<[10, 20], f32>
    %rhs: ... !mo.tensor<[20, 5], f32>
    %res = mo.matmul(%lhs, %rhs) : (
      !mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>
    ) -> !mo.tensor<[10, 5], f32>
    ```
  }];

  let arguments = (ins MO_Tensor:$input_a, MO_Tensor:$input_b);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

def MO_BatchMatmulOp : MO_DotBase<"batch_matmul", [
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
  MO_SameOperandsAndResultElementType
]> {
  let summary = "Batched matrix-matrix multiplication";
  let description = [{
    Perform matrix multiplication on two batches of matrices, represented by
    two N-dimensional tensors.

    The last two dimensions of each input are the matrix dimensions.

    Example:

    ```mlir
    %lhs: ... !mo.tensor<[3, 4, 5], f32>
    %rhs: ... !mo.tensor<[3, 5, 6], f32>
    %res = mo.batch_matmul(%lhs, %rhs) :
      mo.tensor<[3, 4, 5], f32>, !mo.tensor<[3, 5, 6], f32>
    ) -> !mo.tensor<[3, 4, 6], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input_a,
    MO_Tensor:$input_b,
    OptionalAttr<BoolAttr>:$transpose_b,
    MO_ParamDecls:$outputParamDecls);

  let builders = [
    /// Create matmul with default attribute flags off.
    OpBuilder<(ins "Type":$type, "Value":$input_a, "Value":$input_b)>,
    OpBuilder<(ins "Type":$type, "Value":$input_a, "Value":$input_b, "ArrayRef<KGEN::ParamDeclAttr>":$decls)>,
  ];

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Linear Algebra ops
//===----------------------------------------------------------------------===//

class MO_LinalgOp<string mnemonic, list<Trait> traits = []>
    : MO_TensorOp<"linalg." # mnemonic, traits # [Pure]> {
  string cppNamespace = MODialect.cppNamespace # "::LinAlg";
}

// TODO: Use an AllElementTypesMatch<["input", "result"]> trait for MO_Tensor.
def MO_BandPartOp : MO_LinalgOp<"band_part", [
    DeclareOpInterfaceMethods<MO_SinkStaticBroadcastTo>,
    DeclareOpInterfaceMethods<MO_Parameterization>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let summary = "Matrix Band Part operation";
  let description = [{
    Copy a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assume the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

    Please explicitly note that with negative values, this kernel returns the
    entire lower or uppper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
    %arg: !mo.tensor<[3, 2, 3], f32>
    %num_lower = mo.constant {
      value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %num_upper = mo.constant {
      value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
    %exclude = mo.constant {
      value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
    %res = mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
      !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
      !mo.tensor<[], bool>
      ) -> !mo.tensor<[3, 2, 3], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_RankedScalarIndexTensor:$num_lower,
    MO_RankedScalarIndexTensor:$num_upper,
    MO_CondLike:$exclude,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def MO_SolveOp : MO_LinalgOp<"solve", [
    MO_SameOperandsAndResultElementType,
    DeclareOpInterfaceMethods<MO_Parameterization>,
    DeclareOpInterfaceMethods<MO_ParamDeclaration>,
    DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>]> {
  let summary = "Solves systems of Linear Equations";
  let description = [{
    Solve the system of linear equations described by

      AX = B

    Where input A is a batch of square matrices and has shape [I, J, ..., M, M]
    which are assumed to be full rank/invertible and rhs B is a batch of
    results of shape [I, J, ..., M, K]. The result, X, will be a tensor of
    shape [I, J, ..., M, K]

    Example:

    ```mlir
    %arg: !mo.tensor<[3, 3, 3], f32>
    %rhs: !mo.tensor<[3, 3, 2], f32>
    %res = mo.linalg.band_part(%arg, %rhs) : (
      !mo.tensor<[3, 3, 3], f32>, !mo.tensor<[3, 3, 2], f32>, !mo.tensor<[]
      ) -> !mo.tensor<[3, 3, 2], f32>
    ```
  }];

  let arguments = (ins
    MO_Tensor:$input,
    MO_Tensor:$rhs,
    MO_ParamDecls:$outputParamDecls);

  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Shape function invocation
//===----------------------------------------------------------------------===//

def MO_InvokeShapeFuncOp : MO_TensorOp<"invoke_shape_func", [
    DeclareOpInterfaceMethods<MO_ParamDeclaration>]> {
  let summary = "Invokes a shape function to declare the shape parameters";
  let description = [{
    Invokes a shape function with given name and bind its return values to the
    output parameters. Each argument must be a shape-like or integer scalar
    tensor. The `kgenParams` will be passed along to the shapep function as is.

    Example:

    ```mlir
    %input : !mo.tensor<[D1, D2], f32>
    %axis  : !mo.tensor<[], si64>
    %input_shape = mo.shape_of(%input)
    : (!mo.tensor<[D1, D2], f32>) -> !mo.tensor<[2], si64>

    mo.invoke_shape_func["shape_func_name"]<() -> D1, D2>(%input_shape, %axis)
    : (!mo.tensor<[2], si64>, !mo.tensor<[], si64>)
    ```

    `dataDeptTensors` Is an instruction to the lowering that the inputs at
    these indices should be treated as data dependent tensors. These tensors
    include things like dyanmic "axis" in reductions or "steps" in slice. Any
    tensor which needs to have its values read to compute the shape for the op
    falls into this category.
  }];

  let arguments = (ins
    Variadic<AnyTypeOf<[MO_Tensor, MO_ShapeLike, MO_RankedScalarIndexTensor]>>:$inputs,
    StrAttr:$shapeFuncName,
    MO_ParamDecls:$outputParamDecls,
    OptionalAttr<DictionaryAttr>:$kgenParams,
    OptionalAttr<ArrayAttr>:$dataDeptTensors
  );

  let assemblyFormat = [{
    `[` $shapeFuncName `]` `` custom<ParamDecls>($outputParamDecls)
    `(` $inputs `)` attr-dict `:` `(` type($inputs) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Layout Transform
//===----------------------------------------------------------------------===//

def MO_LayoutTransformOp : MO_TensorOp<"layout.transform", [Pure]> {
  let summary = "Transform the layout of input tensor";
  let description = [{
    This op will transform the layout of input tensor to the layout specified in
    the output type.

    It requires both the input and output tensor types to contain layout
    annotations.
  }];

  let arguments = (ins MO_Tensor:$input);
  let results = (outs MO_Tensor:$result);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` qualified(type($input)) `->` qualified(type($result))
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// Temp layer norm ops
//===----------------------------------------------------------------------===//

def MO_LayerNormOp : MO_TensorOp<"layer_norm"> {
  let summary = "Layer norm";
  let description = [{
    Layer norm target for the pattern matcher
  }];

  let arguments = (ins MO_Tensor:$input, MO_Tensor:$gamma, MO_Tensor:$beta, MO_Tensor:$epsilon);
  let results = (outs MO_Tensor:$result);
}

def MO_LayerNormSkipOp : MO_TensorOp<"layer_norm_skip"> {
  let summary = "Layer norm with skip";
  let description = [{
    Layer norm with skip target for the pattern matcher
  }];

  let arguments = (ins MO_Tensor:$input, MO_Tensor:$gamma, MO_Tensor:$beta, MO_Tensor:$epsilon, MO_Tensor:$skip);
  let results = (outs MO_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Print
//===----------------------------------------------------------------------===//

def DebugTensorPrintOp : MO_TensorOp<"debug.tensor.print"> {
  let summary = "Prints a debug representation of argument input";
  let description = [{
    Prints a debug representation of argument input. If a label attribute
    is supplied the tensor contents is printed with that label. Otherwise
    just the tensor metadata is printed. For debugging and testing only.

    Example:
    ```mlir
    %arg: !mo.tensor<[5], f32>
    mo.debug.tensor.print(%arg) {label = "label"} : !mo.tensor<[5], f32>
    ```
  }];

  let arguments = (ins MO_Tensor: $input,
                   DefaultValuedAttr<StrAttr, "\"\"">: $label);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` qualified(type($input))
  }];
}

def MO_TopKOp : MO_TensorOp<"top_k", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Top K values and indices along an axis";
  let description = [{
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis. Returned values along the axis are always sorted
    (stable).

    Example:
    ```mlir
    %in = "mo.constant"() {value =
      #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>} :
      () -> !mo.tensor<[2, 6], si64>
    %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () ->
      !mo.tensor<[], si64>
    %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () ->
      !mo.tensor<[], si64>
    %values, %indices = mo.top_k(%in, %K, %axis) : (
      !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) ->
        (!mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>)
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_RankedScalarIndexTensor:$K,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$values,
                      MO_I64Tensor:$indices);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}

def MO_BottomKOp : MO_TensorOp<"bottom_k", [
  DeclareOpInterfaceMethods<MO_DefaultParameterization>,
  DeclareOpInterfaceMethods<MO_ShapeInferenceMaterialization>,
]> {
  let summary = "Bottom K values and indices along an axis";
  let description = [{
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
    %in = "mo.constant"() {value =
      #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>} :
      () -> !mo.tensor<[2, 6], si64>
    %K = "mo.constant"() {value = #M.dense_array<3> : tensor<si64>} : () ->
      !mo.tensor<[], si64>
    %axis = "mo.constant"() {value = #M.dense_array<1> : tensor<si64>} : () ->
      !mo.tensor<[], si64>
    %values, %indices = mo.bottom_k(%in, %K, %axis) : (
      !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>) ->
        (!mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>)
    ```
  }];

  let arguments = (ins MO_Tensor:$input,
                       MO_RankedScalarIndexTensor:$K,
                       MO_RankedScalarIndexTensor:$axis,
                       MO_ParamDecls:$outputParamDecls);
  let results = (outs MO_Tensor:$values,
                      MO_I64Tensor:$indices);
  let assemblyFormat = [{
    `` custom<ParamDecls>($outputParamDecls) `(` operands `)` attr-dict `:`
    functional-type(operands, results)
  }];
}


#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOOPS_TD
