//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GenericML/MGPTransforms/MGPTransforms.h"

#include "GenericML/MGPDialect/MGPOps.h"
#include "GenericML/MGPDialect/MGPTensorInterface.h"
#include "Support/AssertStream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include "./Utils.h"

using namespace mlir;

#define DEBUG_TYPE "mgp-defer-allocs"

namespace M::MGP {

// Generate support code for DeferAllocsPass
#define GEN_PASS_DEF_DEFERALLOCSPASS
#include "GenericML/MGPTransforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// DeferAllocsPass
//===----------------------------------------------------------------------===//

/// This pass defers buffer allocations until the operations which write to
/// them are ready to execute.
class DeferAllocsPass : public impl::DeferAllocsPassBase<DeferAllocsPass> {
  void runOnOperation() override {
    ModelOp model = getOperation();
    ExecuteState execState(model.getContext(), model.getExecuteBlock());
    // Recover type and buffer builders appropriate for the model.
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(model);

    // This helps prevent infinite loop in case an op write to multiple buffers
    DenseSet<Operation *> movedAllocOps;

    auto defer = [&](Operation *op) {
      if (movedAllocOps.contains(op))
        return;
      if (auto allocOp = dyn_cast<MGPBufferAllocOpInterface>(op)) {
        // For the unique mutator of the allocated buffer...
        if (OpOperand *mutator = getUniqueMutator(allocOp.getOutBuffer())) {
          // For the input chain(s) to that mutator....
          for (OpOperand *mutatorChainOpOperand :
               getInputChainOpOperands(mutator->getOwner())) {
            // Move the allocation to just before the mutating op.
            // (Note that this may mean we revisit the same allocation again.)
            allocOp->moveBefore(mutator->getOwner());
            movedAllocOps.insert(allocOp);
            // The allocation must only be triggered once the mutating op's
            // input chain is ready.
            Value mutatorInChain = mutatorChainOpOperand->get();
            LLVM_DEBUG(llvm::dbgs() << "adding chain:\n  " << mutatorInChain
                                    << "\nto op:\n  " << allocOp << "\n");
            execState.builder.setInsertionPoint(allocOp);
            auto res = alsoWaitForChains(execState.builder, builderContext,
                                         allocOp, {mutatorInChain});
            ASSERT_STREAM(succeeded(res),
                          << "all allocation ops must accept an input "
                             "chain to control their timing");
          }
        }
      }
    };

    // For each mgp.buffer.alloc.*...
    model.getExecute().walk(defer);
    model.getInit().walk(defer);
  }
};

} // namespace
} // namespace M::MGP
