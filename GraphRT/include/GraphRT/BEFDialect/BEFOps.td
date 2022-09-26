//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the BEF dialect.
//
//===----------------------------------------------------------------------===//

include "GraphRT/BEFDialect/BEFDialect.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Base class for the operation in this dialect.
class BEFOp<string mnemonic, list<Trait> traits = []> :
    Op<BEFDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// InvokableOp
//===----------------------------------------------------------------------===//

def InvokableOp
  : BEFOp<"invokable", [HasParent<"ModuleOp">, FunctionOpInterface,
                        Symbol, IsolatedFromAbove,
                        SingleBlockImplicitTerminator<"OutputOp">]> {
  let summary = "Named invokable in a BEF file";
  let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getBody().front();
    }

    OutputOp getOutputOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let builders = [
    /// Create a new bef.invokable with an empty body.
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

def OpaqueOp : BEFOp<"opaque"> {
  let summary = "opaque BEF operation";
  let description = [{
    The "bef.opaque" operation represents an operation parsed from a BEF file
    without sufficient type information to form the proper MLIR operation.  This
    allows us to load and round trip arbitrary BEF files even when they don't
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

def OutputOp : BEFOp<"output", [Terminator, ReturnLike]> {
  let summary = "BEF output operation";
  let description = [{
    The `bef.output` operation specifies the returned values for invokables in
    a BEF MLIR file.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the `bef.invokable`
    (or enclosing region) that contains the operation. For example:

    ```mlir
      bef.invokable @foo() : i32, f8 {
        ...
        bef.output %0, %1 : i32, f8
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let extraClassDeclaration = [{
    /// Verify that the operands of this OutputOp match the specified set of
    /// types.
    LogicalResult checkArgumentTypes(TypeRange types);
  }];
}
