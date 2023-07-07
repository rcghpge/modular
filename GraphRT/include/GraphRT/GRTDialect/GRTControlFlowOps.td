//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines the control flow operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "GraphRT/MEFDialect/MEFNonStrictInterface.td"
include "GraphRT/GRTDialect/GRTOpsBase.td"
include "GraphRT/GRTDialect/GRTControlFlowInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ChainCreateOp
//===----------------------------------------------------------------------===//

def ChainCreateOp : GRTOp<"chain.create", [Pure, Commutative]> {
  let summary = "Return a ready !mef.chain when all inputs (if any) are ready";

  let description = [{
    This operation returns a ready !mef.chain when all of its inputs are ready.
    If there are no inputs a ready chain is returned immediately. The operands
    may be of any type, not just !mef.chain. However, non !mef.chain operands
    must be paired with a type annotation.

    Example:

    ```mlir
      %ch0 = grt.chain.create ()
      %idx: index
      %int: i32
      %ch1 = grt.chain.create (%idx: index, %ch0, %int: i32)
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs ChainType:$result);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins)>
  ];
}

//===----------------------------------------------------------------------===//
// ChainFanoutOp
//===----------------------------------------------------------------------===//

def ChainFanoutOp : GRTOp<"chain.fanout", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "";

  let description = [{
    This operation returns any number of !mef.chain results which will be ready
    when the argument !mef.chain is ready. However, if a result chain makes
    a dependent primitive ready to run then that primitive will be run
    asynchronously.

    This can be used to ensure high-fanout values to synchronous primitives
    do not all run sequentially on the same thread.

    Example:

    ```mlir
      // %in_ch will be ready when %x is ready to be read
      %in_ch = "some.writing.op"(%x)
      %out_ch0, %out_ch1 = grt.chain.fanout %in_ch, 2
      // op{1,2} and op{3,4} will begin execution on separate threads
      ... = "some.reading.op1"(%out_ch0, %x)
      ... = "some.reading.op2"(%out_ch0, %x)
      ... = "some.reading.op3"(%out_ch1, %x)
      ... = "some.reading.op4"(%out_ch1, %x)
    ```
  }];

  let arguments = (ins ChainType:$inChain, UI32Attr:$numResults);
  let results = (outs Variadic<ChainType>:$outChains);
  let assemblyFormat = [{
    $inChain `,` $numResults attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

def InvokeOp : GRTOp<"invoke", [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                                NonStrictInterface]> {
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
           NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
           DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "if control flow operation";
  let description = [{
    The "grt.if" operation takes an i1 condition, a 'then' block and an 'else'
    block. If the condition is true, the 'then' block is executed and the
    "grt.if" operation returns the results of that block. Otherwise the 'else'
    block is executed and its values are returned.

    Both blocks have access to the outer SSA values listed in the "grt.if"
    lhs signature. Both must return results with the same types as the "grt.if"
    rhs signature, using a "mef.output" operation at the end of each block.

    This operation is 'nonstrict'. When the condition is ready the 'then' or
    'else' blocks will be merged into the overall dataflow graph without
    waiting for any additional arguments to become ready.

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

  let extraClassDefinition = [{
    GRT::ControlFlowFlavor IfOp::getFlavor() { return GRT::kBranchLike; }
    llvm::ArrayRef<mlir::OpOperand> IfOp::getInputOpOperands() {
      return getOperation()->getOpOperands().drop_front();
    }
    mlir::Operation::operand_range IfOp::getStrictExtraOperands() {
      return getOperation()->getOperands().take_front();
    }
    llvm::SmallVector<llvm::StringRef> IfOp::getRegionNames() { return {"then", "else"}; }
  }];
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : GRTOp<"while",
          [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
           NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
           DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "while control flow operation";
  let description = [{
    The "grt.while" operation takes 'cond' and 'body' blocks. While the
    'cond' block evaluates to the i1 true condition, the 'body' block is
    executed.

    Both the 'cond' and 'body' blocks have access to the outer SSA values listed
    in the "grt.while" signature. The 'cond' block must return results with the
    same types as the "grt.while" signatures, followed by an i1 indicating the
    condition result, using a "mef.output" op. The 'body' block must return
    results with the same types as the "grt.while" signature, again using the
    "mef.output" operation. Each execution of the 'cond' or 'body' block
    effectively 'updates' the arguments given in the "grt.while" signature,
    and the results of the "grt.while" op are the last 'updated' arguments.

    Note we allow the 'cond' block to 'update' arguments so as to support
    side-effecting condition code. In particular, it is possible to thread a
    chain through all 'cond' and 'body' block execution.

    This operation is 'nonstrict'. The 'cond' block is merged into the overall
    dataflow graph without waiting for any arguments to become ready. When its
    condition result is true, the 'body' block is similarly merged into the
    overall dataflow graph, followed by the 'cond' for the next iteration. In
    this way the 'cond' block may control how much asynchrony is permitted in
    the loop.

    Example:

    ```mlir
      %res1, %res2 = grt.while (%x: i32, %y: f32) {
        %v1 = "compare"(%x, %y) : (i32, f32) -> i1
        mef.output %x, %y, %v1 : (i32, f32, i1)
      } do {
        %v2, %v3 = "use"(%x, %y) : (i32, f32) -> (i32, f32)
        mef.output %v2, %v3 : (i32, f32)
      }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condRegion,
                        SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDefinition = [{
    GRT::ControlFlowFlavor WhileOp::getFlavor() { return GRT::kLoopLike; }
    llvm::ArrayRef<mlir::OpOperand> WhileOp::getInputOpOperands() {
      return getOperation()->getOpOperands();
    }
    mlir::Operation::operand_range WhileOp::getStrictExtraOperands() {
      return getOperation()->getOperands().take_front(0);
    }
    llvm::SmallVector<llvm::StringRef> WhileOp::getRegionNames() { return {"cond", "body"}; }
  }];
}

//===----------------------------------------------------------------------===//
// RepeatI32Op
//===----------------------------------------------------------------------===//

def RepeatI32Op : GRTOp<"repeat.i32",
         [IsolatedFromAbove, SingleBlockImplicitTerminator<"MEF::OutputOp">,
          NonStrictInterface, RegionKindInterface, HasOnlyGraphRegion,
          DeclareOpInterfaceMethods<GRTControlFlowInterface>]> {
  let summary = "Repeat a body N times";
  let description = [{
    The "grt.repeat.i32" operation is a repeat loop that executes its 'body'
    block 'tripCount' times, where 'tripCount' is an i32 value specified by the
    first operand.

    The 'body' block has access to the outer SSA values listed in the
    "grt.repeat.i32" signature. The block must return values with the
    same types as the signature, using a "mef.output" operation at the end of
    the block. The result of the "grt.repeat.i32" op is the result of the last
    executed 'body' block, or the "grt.repeat.i32" signature arguments directly
    if 'tripCount' is <= 0.

    This operation is 'nonstrict'. Once the 'tripCount' operand is ready and
    > 0 then the 'body' block is merged into the overall dataflow graph.
    However, each subsequent execution of the 'body' block is only merged
    once all results of the previous iteration are ready. (This is a policy
    decision to avoid unrolling the loop body into a potentially very large
    dataflow graph.)

    Example:

    ```mlir
      %res1, %res2 = grt.repeat.i32 %i (%loopval1: si32, %loopval2: f32) {
        %v1, v2 = "process"(%loopval1, %loopval2) : (si32, f32) -> (s132, f32)
        mef.output %v1, %v2 : si32, f32
      }
    ```
  }];

  let arguments = (ins SI32:$tripCount, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDefinition = [{
    GRT::ControlFlowFlavor RepeatI32Op::getFlavor() { return GRT::kLoopLike; }
    llvm::ArrayRef<mlir::OpOperand> RepeatI32Op::getInputOpOperands() {
      return getOperation()->getOpOperands().drop_front();
    }
    mlir::Operation::operand_range RepeatI32Op::getStrictExtraOperands() {
      return getOperation()->getOperands().take_front();
    }
    llvm::SmallVector<llvm::StringRef> RepeatI32Op::getRegionNames() { return {"body"}; }
  }];
}
