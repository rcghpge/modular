//===- BEFOps.td -------------------------------------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the BEF dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : BEFOp<"func", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                            Symbol, IsolatedFromAbove,
                            SingleBlockImplicitTerminator<"OutputOp">]> {
  let summary = "Named function in a BEF file";
  let arguments = (ins TypeAttrOf<FunctionType>:$type);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getBody().front();
    }

    mlir::FunctionType getType() { return type(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }
  }];

  let builders = [
    /// Create a new bef.func with an empty body.
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<Type>":$inputTypes,
                   "ArrayRef<Type>":$resultTypes,
                   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
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

  let arguments = (ins StrAttr:$opname, ArrayAttr:$attrs, UnitAttr:$nonstrict,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$body);
  let assemblyFormat = [{
    $opname (`nonstrict` $nonstrict^)? $attrs `(` $operands `)` attr-dict `:` 
            functional-type($operands, $results) $body
  }];
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

def OutputOp : BEFOp<"output", [Terminator, ReturnLike]> {
  let summary = "BEF output operation";
  let description = [{
    The "bef.output" operation specifies the returned values for a BEF function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the bef.func
    that contains the operation. For example:

    ```mlir
      bef.func @foo() : i32, f8 {
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
    mlir::LogicalResult checkArgumentTypes(TypeRange types);
  }];
}

