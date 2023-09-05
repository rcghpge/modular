//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the MEF dialect.
//
//===----------------------------------------------------------------------===//

include "GraphRT/MEFDialect/MEFDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

// Base class for the operation in this dialect.
class MEFOp<string mnemonic, list<Trait> traits = []> :
    Op<MEFDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// InvokableOp
//===----------------------------------------------------------------------===//

def InvokableOp
  : MEFOp<"invokable", [HasParent<"ModuleOp">, FunctionOpInterface,
                        Symbol, IsolatedFromAbove,
                        SingleBlockImplicitTerminator<"OutputOp">]> {
  let summary = "Named invokable in a MEF file";
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getFunctionBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getFunctionBody().front();
    }

    OutputOp getOutputOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getFunctionBody(); }
  }];

  let builders = [
    /// Create a new mef.invokable with an empty body.
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$resultTypes,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins CArg<"ArrayRef<NamedAttribute>">:$attrs)>,
  ];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// OpaqueOp
//===----------------------------------------------------------------------===//

def OpaqueOp : MEFOp<"opaque"> {
  let summary = "opaque MEF operation";
  let description = [{
    The "mef.opaque" operation represents an operation parsed from a MEF file
    without sufficient type information to form the proper MLIR operation.  This
    allows us to load and round trip arbitrary MEF files even when they don't
    have additional metadata (e.g. MLIR types, attribute names etc) required to
    properly form original operations.
  }];

  let arguments = (ins StrAttr:$opName, ArrayAttr:$attrs, UnitAttr:$nonstrict,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$body);
  let assemblyFormat = [{
    $opName (`nonstrict` $nonstrict^)? $attrs `(` $operands `)` attr-dict `:`
            functional-type($operands, $results) $body
  }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def OutputOp : MEFOp<"output", [Terminator, ReturnLike]> {
  let summary = "MEF output operation";
  let description = [{
    The `mef.output` operation specifies the returned values for invokables in
    a MEF MLIR file.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the `mef.invokable`
    (or enclosing region) that contains the operation. For example:

    ```mlir
      mef.invokable @foo() : i32, f8 {
        ...
        mef.output %0, %1 : i32, f8
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
