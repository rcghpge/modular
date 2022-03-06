//===- GraphRT/GRTDialect/GRTControlFlowOps.td -------------*- tablegen -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// ChainNewOp
//===----------------------------------------------------------------------===//

def ChainNewOp : GRTOp<"chain.new"> {
  let summary = "Return a new !grt.chain";

  let arguments = (ins);
  let results = (outs ChainType:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : GRTOp<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call a BEF function";

  let description = [{
     This operation is safe to use as a 'nonstrict' op, allowing the function
     executor to dispatch computation that isn't dependent on the unavailable
     operands to the call.
  }];
    
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

def RepeatI32Op : GRTOp<"repeat.i32",
         [IsolatedFromAbove, SingleBlockImplicitTerminator<"BEF::OutputOp">]> {
  let summary = "Repeat a body N times";
  let description = [{
    The "grt.repeat.i32" operation is a repeat loop that executes its body N
    times, where N is a 32-bit value specified by its first operand.  It has one
    region which represents the loop body.

    The body region takes an arbitrary number of arguments and must return
    values with the same types, using a 'bef.output' operation at the end of the
    region.  The initial values for the loop are specified as operands (along
    with the trip count), and the types of the loop values are specified with a
    type list.  The operation returns the results of the final iteration.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever dependent arguments are resolved.

    Example:

    ```mlir
      %res1, %res2 = grt.repeat.i32 %i (%loopval1: i32, %loopval2: f32) {
        ...
        bef.output %loopval1, %loopval2 : i32, f32
      }
    ```
  }];

  let arguments = (ins I32:$tripCount, Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
