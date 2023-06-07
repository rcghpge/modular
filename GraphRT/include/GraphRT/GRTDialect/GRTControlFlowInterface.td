//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD
#define GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD

include "mlir/IR/OpBase.td"

def GRTControlFlowInterface : OpInterface<"GRTControlFlowInterface"> {
  let description = [{
    This interface describes common properties of GRT control flow operators.
  }];

  let cppNamespace = "::M::GRT";

  let methods = [
    InterfaceMethod<
      [{
        Returns the N control flow op 'input' operands which are implicitly
        passed as the first N operands to each of the op's control flow blocks.
        The control flow op may take additional operands before these inputs
        to control its operation, such as a condition or trip count.
      }],
      "::mlir::Operation::operand_range", "getInputs", (ins)
    >,
    InterfaceMethod<
      [{
        As for getInputs, but returned as an array of OpOperands.
      }],
      "llvm::ArrayRef<::mlir::OpOperand>", "getInputOpOperands", (ins)
    >,
    InterfaceMethod<
      [{
        Returns the 'flavor' of the control flow op.

        For a `kLoopLike` op (eg grt.{while, repeat.i32}):
         - The op inputs match the block arguments.
         - The first N results of each of the op's control flow blocks
           represent updated versions of the N op inputs. Blocks may return
           additional outputs, such as a condition value.
         - The updated values are chained from the op inputs through each
           of the control flow op's blocks and back to the inputs or the
           control flow op's results.

        For a 'kBranchLike` op (eg grt.if):
         - The op inputs match the block arguments.
         - Each of the control flow op's blocks are assumed to be distinct
           control flow paths which all agree on their result types.
      }],
      "M::GRT::ControlFlowFlavor", "getFlavor", (ins)
    >,
    InterfaceMethod<
      [{
        Returns the human-readable names for each of the control flow
        ops nested regions.
      }],
      "llvm::SmallVector<llvm::StringRef>", "getRegionNames", (ins)
    >,
  ];

  let verify = [{
    return M::GRT::verifyControlFlowInterfaceOp($_op);
  }];
}

#endif // GRAPHRT_GRTDIALECT_GRTCONTROLFLOWINTERFACE_TD
