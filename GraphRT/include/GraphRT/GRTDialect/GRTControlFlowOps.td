//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "GraphRT/MEFDialect/MEFNonStrict.td"
include "GraphRT/GRTDialect/GRTOpsBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

def ChainCreateOp : GRTOp<"chain.create"> {
  let summary = "Creates a new !mef.chain";

  let arguments = (ins);
  let results = (outs ChainType:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ChainMergeOp
//===----------------------------------------------------------------------===//

def ChainMergeOp : GRTOp<"chain.merge", [Pure]> {
  let summary = "Return a !mef.chain when all inputs are ready";

  let arguments = (ins Variadic<ChainType>:$operands);
  let results = (outs ChainType:$result);
  let assemblyFormat = "$operands attr-dict";

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

def InvokeOp : GRTOp<"invoke", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                            NonStrictOp]> {
  let summary = "Invoke an invokable (non-strictly).";

  let arguments = (ins FlatSymbolRefAttr:$target, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $target `(` $operands `)` attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : GRTOp<"if",
          [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
           NonStrictOp]> {
  let summary = "if operation";
  let description = [{
    The "grt.if" operation takes an i1 condition, a 'then' body and an 'else'
    body.  If the condition is true, the 'then' body is run and the "grt.if"
    operation returns the results of that region, otherwise the "else" body is
    run and its values are returned.

    The body regions must have the same signature, but they may take an
    arbitrary number of arguments and return values with the same types as the
    op return types, using a 'mef.output' operation at the end of the
    region.  The initial values are specified as operands of "mef.if" (along
    with the condition).

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever the condition and any dependent arguments are resolved.

    Example:

    ```mlir
      %res = grt.if %cond (%x: i32, %y: f32) -> i32 {
        %v1 = "use"(%x, %y) : (i32, f32) -> i32
        mef.output %v1 : i32
      } else {
        %v2 = "abuse"(%x, %y) : (i32, f32) -> i32
        mef.output %v2 : i32
      }
    ```
  }];

  let arguments = (ins I1:$cond, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

def RepeatI32Op : GRTOp<"repeat.i32",
         [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
          NonStrictOp]> {
  let summary = "Repeat a body N times";
  let description = [{
    The "grt.repeat.i32" operation is a repeat loop that executes its body N
    times, where N is a 32-bit value specified by its first operand.  It has one
    region which represents the loop body.

    The body region takes an arbitrary number of arguments and must return
    values with the same types, using a 'mef.output' operation at the end of the
    region.  The initial values for the loop are specified as operands (along
    with the trip count), and the types of the loop values are specified with a
    type list.  The operation returns the results of the final iteration.

    This operation is safe to use as a 'nonstrict' op, which dispatches its body
    whenever dependent arguments are resolved.

    Example:

    ```mlir
      %res1, %res2 = grt.repeat.i32 %i (%loopval1: si32, %loopval2: f32) {
        ...
        mef.output %loopval1, %loopval2 : si32, f32
      }
    ```
  }];

  let arguments = (ins SI32:$tripCount, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
