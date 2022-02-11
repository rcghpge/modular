//===- BEFOps.td - BEF operators definition ----------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the BEF dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def FuncOp : BEFOp<"func", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                            Symbol, SymbolTable, IsolatedFromAbove]> {
  let summary = "Named function in a BEF file";
  let arguments = (ins);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block *getBodyBlock() {
      return &getBody().front();
    }

    mlir::FunctionType getType() {
      return getTypeAttr().getValue().cast<mlir::FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    LogicalResult verifyType();
  }];

  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::printFuncOp(*this, p); }];
  let hasVerifier = 1;
}

def OutputOp : BEFOp<"output",
                     [Terminator, ReturnLike, HasParent<"BEF::FuncOp">]> {
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
  let hasVerifier = 1;
}

